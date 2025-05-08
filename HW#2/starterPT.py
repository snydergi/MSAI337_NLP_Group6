from transformers import AutoTokenizer, BertModel
import torch.optim as optim
import torch.nn as nn
import torch
import math
import time
import sys
import json
import numpy as np
from torch.utils.data import Dataset, DataLoader

answers = ['A', 'B', 'C', 'D']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MCQADataset(Dataset):
    def __init__(self, data, tokenizer, max_len=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        instance = self.data[idx]
        texts = [item[0] for item in instance]
        labels = [item[1] for item in instance]

        encodings = self.tokenizer(
            texts, padding='max_length', truncation=True, max_length=self.max_len, return_tensors="pt"
        )

        return {
            'input_ids': encodings['input_ids'],  # shape: [4, max_len]
            'attention_mask': encodings['attention_mask'],
            'label': torch.tensor(labels.index(1)),  # index of the correct answer
        }


class BERTMCQAModel(nn.Module):
    def __init__(self, model_name='bert-base-uncased'):
        super(BERTMCQAModel, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.classifier = nn.Linear(self.bert.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        B, C, L = input_ids.size()  # Batch, Choices, Seq_len
        input_ids = input_ids.view(B * C, L)
        attention_mask = attention_mask.view(B * C, L)

        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]  # [B*C, hidden]
        logits = self.classifier(cls_output)  # [B*C, 1]
        logits = logits.view(B, C)  # [B, C]
        return logits


def prepare_data(file_name):
    data = []
    with open(file_name) as json_file:
        for line in json_file:
            result = json.loads(line)
            base = result['fact1'] + ' [SEP] ' + result['question']['stem']
            ans = answers.index(result['answerKey'])

            obs = []
            for j in range(4):
                text = f"[CLS]{base} {result['question']['choices'][j]['text']}[END]"
                label = 1 if j == ans else 0
                obs.append([text, label])
            data.append(obs)
    return data


def train_model(model, dataloader, optimizer, criterion, epoch):
    model.train()
    total_loss = 0
    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch} - Loss: {total_loss/len(dataloader):.4f}")


def evaluate_model(model, dataloader):
    model.eval()
    total, correct = 0, 0
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            logits = model(input_ids, attention_mask)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    accuracy = correct / total
    print(f"Accuracy: {accuracy:.4f}")
    return accuracy


def main():
    torch.manual_seed(0)

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    # Prepare datasets
    train_data = prepare_data('train_complete.jsonl')
    valid_data = prepare_data('dev_complete.jsonl')
    test_data = prepare_data('test_complete.jsonl')

    train_dataset = MCQADataset(train_data, tokenizer)
    valid_dataset = MCQADataset(valid_data, tokenizer)
    test_dataset = MCQADataset(test_data, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=8)
    test_loader = DataLoader(test_dataset, batch_size=8)

    model = BERTMCQAModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=3e-5)
    criterion = nn.CrossEntropyLoss()

    # Train
    for epoch in range(3):  # Adjust epoch count
        train_model(model, train_loader, optimizer, criterion, epoch)
        print("Validation:")
        evaluate_model(model, valid_loader)

    print("Test:")
    evaluate_model(model, test_loader)


if __name__ == "__main__":
    main()

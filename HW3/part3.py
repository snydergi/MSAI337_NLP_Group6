from transformers import AutoTokenizer, BertModel
import torch.optim as optim
import torch
import json
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import time
from datetime import timedelta

answers = ['A', 'B', 'C', 'D']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LoRALinear(nn.Module):
    def __init__(self, original_linear, r=4, alpha=16, dropout=0.0):
        super().__init__()
        self.original = original_linear  # Keep original Linear layer
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        in_features = original_linear.in_features
        out_features = original_linear.out_features

        # Freeze original layer
        for param in self.original.parameters():
            param.requires_grad = False

        self.lora_A = nn.Parameter(torch.randn((r, in_features)) * 0.01)
        self.lora_B = nn.Parameter(torch.randn((out_features, r)) * 0.01)

    def forward(self, x):
        return self.original(x) + (self.dropout(x) @ self.lora_A.T @ self.lora_B.T * self.scaling)


def apply_lora_to_self_attention(attention, r=4, alpha=16, dropout=0.0):
    for proj_name in ['query', 'value']:
        original_proj = getattr(attention, proj_name)
        lora_wrapped = LoRALinear(original_proj, r=r, alpha=alpha, dropout=dropout)
        setattr(attention, proj_name, lora_wrapped)


class BertDataset(Dataset):
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
            'input_ids': encodings['input_ids'],
            'attention_mask': encodings['attention_mask'],
            'label': torch.tensor(labels.index(1)),
        }


class BertClassifier(nn.Module):
    def __init__(self, r=4, alpha=16, dropout=0.1, lora_dropout=0.05):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(768, 1)

        # Freeze all pretrained BERT parameters
        for param in self.bert.parameters():
            param.requires_grad = False

        # Inject LoRA into attention layers
        for layer in self.bert.encoder.layer:
            apply_lora_to_self_attention(layer.attention.self, r=r, alpha=alpha, dropout=lora_dropout)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits.squeeze(-1)


def print_trainable_params(model):
    print("Trainable parameters:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)


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
    start_time = time.time()
    for batch in dataloader:
        input_ids = batch['input_ids'].view(-1, batch['input_ids'].size(-1)).to(device)
        attention_mask = batch['attention_mask'].view(-1, batch['attention_mask'].size(-1)).to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)

        logits = logits.view(-1, 4)

        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    epoch_time = time.time() - start_time
    print(f"Epoch {epoch} - Loss: {total_loss/len(dataloader):.4f} - Time: {timedelta(seconds=epoch_time)}")
    return epoch_time


def evaluate_model(model, dataloader):
    model.eval()
    total, correct = 0, 0
    start_time = time.time()
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].view(-1, batch['input_ids'].size(-1)).to(device)
            attention_mask = batch['attention_mask'].view(-1, batch['attention_mask'].size(-1)).to(device)
            labels = batch['label'].to(device)

            logits = model(input_ids, attention_mask)
            logits = logits.view(-1, 4)

            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    inference_time = time.time() - start_time
    accuracy = correct / total
    avg_inference_time = inference_time / len(dataloader)
    print(f"Accuracy: {accuracy:.4f}")
    print(f'Inference Time: {timedelta(seconds=inference_time)} - Avg per batch: {avg_inference_time:.4f} seconds')
    return accuracy, inference_time


def main():
    torch.manual_seed(0)

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    tokenizer.add_special_tokens = False

    # Prepare datasets
    train_data = prepare_data('train_complete.jsonl')
    valid_data = prepare_data('dev_complete.jsonl')
    test_data = prepare_data('test_complete.jsonl')

    train_dataset = BertDataset(train_data, tokenizer)
    valid_dataset = BertDataset(valid_data, tokenizer)
    test_dataset = BertDataset(test_data, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=8)
    test_loader = DataLoader(test_dataset, batch_size=8)

    model = BertClassifier().to(device)
    print_trainable_params(model)

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=3e-5)
    criterion = nn.CrossEntropyLoss()

    # Train
    print("Pre Train test")
    evaluate_model(model, test_loader)

    print("Pre train valid")
    evaluate_model(model, valid_loader)

    total_train_time = 0
    for epoch in range(3):
        epoch_time = train_model(model, train_loader, optimizer, criterion, epoch)
        total_train_time += epoch_time
        print("Test Accuracy:")
        test_acc, test_time = evaluate_model(model, test_loader)

    print("Validation Accuracy:")
    valid_acc, valid_time = evaluate_model(model, valid_loader)
    print(f"Total Training Time: {timedelta(seconds=total_train_time)}")
    print(f'Test Time: {timedelta(seconds=test_time)}')
    print(f'Validation Time: {timedelta(seconds=valid_time)}')


if __name__ == "__main__":
    main()

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


class PrefixEncoder(nn.Module):
    def __init__(self, config, prefix_length):
        super().__init__()
        self.prefix_length = prefix_length
        self.hidden_size = config.hidden_size
        self.num_layers = config.num_hidden_layers
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads

        self.prefix_embeddings = nn.Embedding(self.prefix_length, self.hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Tanh(),
            nn.Linear(self.hidden_size, 2 * self.num_layers * self.hidden_size),
        )

    def forward(self, batch_size):
        prefix_tokens = torch.arange(self.prefix_length, device=device).unsqueeze(0).expand(batch_size, -1)
        prefix_embed = self.prefix_embeddings(prefix_tokens)  # (B, prefix_length, H)
        past_key_values = self.mlp(prefix_embed)  # (B, prefix_length, 2 * L * H)

        # Split and reshape into tuples of (key, value) for each layer
        past_key_values = (
            past_key_values.view(batch_size, self.prefix_length, self.num_layers * 2, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
            .split(2)
        )

        past_key_values = tuple((kv[0], kv[1]) for kv in past_key_values)  # length = num_layers
        return past_key_values


class PrefixTunedBERT(nn.Module):
    def __init__(self, prefix_length=5):
        super().__init__()
        self.prefix_length = prefix_length
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.config = self.bert.config
        for param in self.bert.parameters():
            param.requires_grad = False  # freeze BERT

        self.prefix_embeddings = nn.Embedding(prefix_length, self.config.hidden_size)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        batch_size = input_ids.size(0)

        # Create prefix token ids: (batch_size, prefix_length)
        prefix_tokens = torch.arange(self.prefix_length, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)

        # Get prefix embeddings: (batch_size, prefix_length, hidden_size)
        prefix_embed = self.prefix_embeddings(prefix_tokens)

        # Get input embeddings: (batch_size, seq_len, hidden_size)
        inputs_embeds = self.bert.embeddings(input_ids)

        # Concatenate: (batch_size, prefix_length + seq_len, hidden_size)
        inputs_embeds = torch.cat([prefix_embed, inputs_embeds], dim=1)

        # Update attention mask: (batch_size, prefix_length + seq_len)
        prefix_attention_mask = torch.ones(batch_size, self.prefix_length, device=attention_mask.device)
        attention_mask = torch.cat([prefix_attention_mask, attention_mask], dim=1)

        # Pass through BERT using inputs_embeds directly
        outputs = self.bert(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, self.prefix_length, :]  # skip prefix, use real CLS

        logits = self.classifier(self.dropout(cls_output))
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

    train_dataset = BertDataset(train_data, tokenizer, max_len=128 - 5)
    valid_dataset = BertDataset(valid_data, tokenizer, max_len=128 - 5)
    test_dataset = BertDataset(test_data, tokenizer, max_len=128 - 5)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=8)
    test_loader = DataLoader(test_dataset, batch_size=8)

    model = PrefixTunedBERT(prefix_length=5).to(device)
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

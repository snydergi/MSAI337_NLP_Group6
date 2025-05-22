import argparse
import os
import sys
import shutil
import random
import numpy as np
import time
import copy
import math
import pickle
import json
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2TokenizerFast, GPT2LMHeadModel


class Embedder(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embed(x.int())


class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len=4096, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        # create constant 'pe' matrix with values dependant on
        # pos and i
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i) / d_model)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # make embeddings relatively larger
        x = x * math.sqrt(self.d_model)
        # add constant to embedding
        seq_len = x.size(1)
        pe = Variable(self.pe[:, :seq_len], requires_grad=False)
        if x.is_cuda:
            pe.cuda()
        x = x + pe
        return self.dropout(x)


class Norm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()

        self.size = d_model

        # create two learnable parameters to calibrate normalisation
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))

        self.eps = eps

    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm


############################### Harrison Bounds ############################
def euclidean_distance(q, k):
    """Calculates the Euclidean distance between query and key."""
    return torch.sqrt(torch.sum((q.unsqueeze(-2) - k.unsqueeze(-3)) ** 2, dim=-1))


def distance_based_attention(q, k, v, d_k, mask=None, dropout=None):
    """Attention based on Euclidean distance."""
    distances = euclidean_distance(q, k)  # Shape: (batch_size, num_heads, seq_len_q, seq_len_k)

    # Convert distances to scores. Smaller distance means higher score.
    # You might need to experiment with the scaling factor (-1/sqrt(d_k) is common)
    scores = -distances / math.sqrt(d_k)  # Invert and scale

    if mask is not None:
        mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask == 0, -1e9)

    # Apply softmax to get attention weights
    weights = F.softmax(scores, dim=-1)

    if dropout is not None:
        weights = dropout(weights)

    output = torch.matmul(weights, v)
    return output


############################### End_Citation  #############################


def attention(q, k, v, d_k, mask=None, dropout=None):

    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        mask = mask.unsqueeze(1).unsqueeze(2)  # Add head dimension
        scores = scores.masked_fill(mask == 0, float('-inf'))

    scores = F.softmax(scores, dim=-1)

    if dropout is not None:
        scores = dropout(scores)

    output = torch.matmul(scores, v)
    return output


class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout=0.1):
        super().__init__()

        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):

        bs = q.size(0)

        # perform linear operation and split into N heads
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        # transpose to get dimensions bs * N * sl * d_model
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        # calculate attention using function we will define next
        scores = attention(q, k, v, self.d_k, mask, self.dropout)
        # concatenate heads and put through final linear layer
        concat = scores.transpose(1, 2).contiguous().view(bs, -1, self.d_model)
        output = self.out(concat)

        return output


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout=0.1):
        super().__init__()

        # We set d_ff as a default to 2048
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x


def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class CosineWithRestarts(torch.optim.lr_scheduler._LRScheduler):
    """
    Cosine annealing with restarts.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer

    T_max : int
        The maximum number of iterations within the first cycle.

    eta_min : float, optional (default: 0)
        The minimum learning rate.

    last_epoch : int, optional (default: -1)
        The index of the last epoch.

    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        T_max: int,
        eta_min: float = 0.0,
        last_epoch: int = -1,
        factor: float = 1.0,
    ) -> None:
        # pylint: disable=invalid-name
        self.T_max = T_max
        self.eta_min = eta_min
        self.factor = factor
        self._last_restart: int = 0
        self._cycle_counter: int = 0
        self._cycle_factor: float = 1.0
        self._updated_cycle_len: int = T_max
        self._initialized: bool = False
        super(CosineWithRestarts, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        """Get updated learning rate."""
        # HACK: We need to check if this is the first time get_lr() was called, since
        # we want to start with step = 0, but _LRScheduler calls get_lr with
        # last_epoch + 1 when initialized.
        if not self._initialized:
            self._initialized = True
            return self.base_lrs

        step = self.last_epoch + 1
        self._cycle_counter = step - self._last_restart

        lrs = [
            (
                self.eta_min
                + ((lr - self.eta_min) / 2)
                * (np.cos(np.pi * ((self._cycle_counter) % self._updated_cycle_len) / self._updated_cycle_len) + 1)
            )
            for lr in self.base_lrs
        ]

        if self._cycle_counter % self._updated_cycle_len == 0:
            # Adjust the cycle length.
            self._cycle_factor *= self.factor
            self._cycle_counter = 0
            self._updated_cycle_len = int(self._cycle_factor * self.T_max)
            self._last_restart = step

        return lrs


class QADataset(Dataset):
    def __init__(self, filename, tokenizer, max_length=256, is_train=True):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_train = is_train  # Different handling for train vs test
        with open(filename) as f:
            self.examples = [json.loads(line) for line in f]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]
        # During training: Include answer
        if self.is_train:
            prompt = (
                f"Question: {example['question']['stem']}\n"
                f"Fact: {example['fact1']}\n"
                f"Options:\n"
                f"A) {example['question']['choices'][0]['text']}\n"
                f"B) {example['question']['choices'][1]['text']}\n"
                f"C) {example['question']['choices'][2]['text']}\n"
                f"D) {example['question']['choices'][3]['text']}\n"
                f"The correct answer is: {example['answerKey']}"
            )
        # During testing: Exclude answer
        else:
            prompt = (
                f"Question: {example['question']['stem']}\n"
                f"Fact: {example['fact1']}\n"
                f"Options:\n"
                f"A) {example['question']['choices'][0]['text']}\n"
                f"B) {example['question']['choices'][1]['text']}\n"
                f"C) {example['question']['choices'][2]['text']}\n"
                f"D) {example['question']['choices'][3]['text']}\n"
                f"The correct answer is:"  # No answer provided!
            )
        
        tokenized = self.tokenizer(
            prompt, 
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return tokenized["input_ids"].squeeze(0), tokenized["attention_mask"].squeeze(0)


def train_model(model, opt, tokenizer, train_loader, test_loader, test_examples):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    
    for epoch in range(opt.epochs):
        total_loss = 0
        for batch_idx, (input_ids, attention_mask) in enumerate(train_loader):
            input_ids = input_ids.to(opt.device)
            attention_mask = attention_mask.to(opt.device)
            
            # Create labels (shifted input_ids)
            labels = input_ids.clone()
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            
            # Check for NaN loss
            if torch.isnan(loss):
                print("Warning: NaN loss detected!")
                continue
                
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % opt.printevery == 0:
                print(f"Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        # Calculate perplexity safely
        try:
            ppl = math.exp(min(total_loss/len(train_loader), 20))  # Clipped to prevent overflow
        except:
            ppl = float('inf')
            
        print(f"Epoch {epoch}, Loss: {total_loss/len(train_loader):.4f}, Perplexity: {ppl:.4f}")
        test_model(model, opt, tokenizer, test_loader, test_examples)


def test_model(model, opt, tokenizer, test_loader, examples):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_idx, (input_ids, attention_mask) in enumerate(test_loader):
            input_ids = input_ids.to(opt.device)
            attention_mask = attention_mask.to(opt.device)
            
            # Generate just the answer portion
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=3,  # Just generate A/B/C/D
                pad_token_id=tokenizer.eos_token_id,
                do_sample=False,
                temperature=0.7
            )
            
            # Get the generated portion only
            for i in range(input_ids.shape[0]):
                generated = outputs[i][input_ids.shape[1]:]  # Only new tokens
                pred_answer = tokenizer.decode(generated, skip_special_tokens=True).strip()[0]
                
                # Get true answer from original data (not from input!)
                example_idx = batch_idx * opt.batchsize + i
                true_answer = examples[example_idx]['answerKey']
                
                if pred_answer in ["A", "B", "C", "D"]:
                    total += 1
                    if pred_answer == true_answer:
                        correct += 1
                        print(f"Correct: Predicted {pred_answer}, True {true_answer}")
                    else:
                        print(f"Incorrect: Predicted {pred_answer}, True {true_answer}")
    
    if total > 0:
        accuracy = correct / total
        print(f"\nTest Accuracy: {accuracy:.4f} ({correct}/{total})")
        return accuracy
    else:
        print("No valid answers found in test set")
        return 0


def main():

    random.seed(10)

    parser = argparse.ArgumentParser()
    parser.add_argument('-no_cuda', action='store_true')
    parser.add_argument('-SGDR', action='store_true')
    parser.add_argument('-epochs', type=int, default=20)
    parser.add_argument('-d_model', type=int, default=512)
    parser.add_argument('-n_layers', type=int, default=6)
    parser.add_argument('-heads', type=int, default=8)
    parser.add_argument('-dropout', type=float, default=0.1)
    parser.add_argument('-batchsize', type=int, default=8)
    parser.add_argument('-printevery', type=int, default=100)
    parser.add_argument('-lr', type=int, default=3e-5)
    parser.add_argument('-seqlen', type=int, default=512)
    parser.add_argument('-threshold', type=int, default=3)
    parser.add_argument('-savename', type=str, default='modelWeights')
    parser.add_argument('-loadname', type=str, default='./epoch20.pth')
    parser.add_argument('-tied', type=int, default=1)
    parser.add_argument('-dir_name', type=str, default='model')
    parser.add_argument('-norm', type=float, default=2.0)

    opt = parser.parse_args()
    opt.verbose = False

    opt.device = 0 if opt.no_cuda is False else -1
    if opt.device == 0:
        assert torch.cuda.is_available()
    opt.device = torch.device("cuda:0")

    time_name = time.strftime("%y%m%d_%H%M%S")
    opt.time_name = time_name
    dir_name = "saved/%s" % (opt.dir_name)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    source_name = sys.argv[0]
    dir_name = dir_name + "//"
    opt.dir_name = dir_name
    shutil.copy(source_name, dir_name + source_name)
    opt.log_file = dir_name + "log_file.txt"

    print(str(opt), flush=True)

    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    special_tokens = {"additional_special_tokens": ["[START]", "[SEP]", "[ANSWER]", "[A]", "[B]", "[C]", "[D]"]}
    tokenizer.add_special_tokens(special_tokens)
    # Load datasets
    train_dataset = QADataset("train_complete.jsonl", tokenizer, is_train=True)
    valid_dataset = QADataset("dev_complete.jsonl", tokenizer, is_train=False)  
    test_dataset = QADataset("test_complete.jsonl", tokenizer, is_train=False)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=opt.batchsize, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=opt.batchsize)
    test_loader = DataLoader(test_dataset, batch_size=opt.batchsize)

    # Need to load examples separately for evaluation
    with open("test_complete.jsonl") as f:
        test_examples = [json.loads(line) for line in f]

    opt.vocab_size = tokenizer.vocab_size
    print('Vocab size: ', opt.vocab_size)
    temp = []
    for i in range(opt.vocab_size):
        temp.append(i)
    opt.indices = torch.tensor(temp)
    opt.indices = opt.indices.cuda()

    # model = get_model(opt, len(tokenizer))
    model = GPT2LMHeadModel.from_pretrained("models/gpt2/")
    model.resize_token_embeddings(len(tokenizer))
    model.to(opt.device)

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    text = 'total params: %d' % (params)
    print(text, flush=True)
    opt.train_len = len(train_loader)
    opt.optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.98), eps=1e-9)
    if opt.SGDR == True:
        opt.sched = CosineWithRestarts(opt.optimizer, T_max=opt.train_len)

    if opt.savename is not None:
        try:
            os.mkdir(opt.savename)
        except:
            nothing = 1
    opt.src_pad = 0
    opt.trg_pad = 0

    train_model(model, opt, tokenizer, train_loader, test_loader, test_examples)

    test_model(model, opt, tokenizer, test_loader, test_examples)


if __name__ == "__main__":
    main()

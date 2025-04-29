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

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2TokenizerFast

savedTestPs = []

def read_corpus(filename, tokenizer):
    seq = []
    with open(filename, 'rt') as f:
        for line in f:
            line = line.replace('\n', '')
            tokens = tokenizer(line)
            for t in tokens['input_ids']:
                seq.append(t)
    return seq


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


def attention(q, k, v, d_k, mask=None, dropout=None):

    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask == 0, -1e9)

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


# build a decoder layer with two multi-head attention layers and
# one feed-forward layer
class DecoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)

        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

        self.attn_1 = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.ff = FeedForward(d_model, dropout=dropout)

    def forward(self, x, trg_mask):
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn_1(x2, x2, x2, trg_mask))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.ff(x2))
        return x


class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads, dropout):
        super().__init__()
        self.N = N
        self.embed = Embedder(vocab_size, d_model)
        self.pe = PositionalEncoder(d_model, dropout=dropout)
        self.layers = get_clones(DecoderLayer(d_model, heads, dropout), N)
        self.norm = Norm(d_model)

    def forward(self, trg, trg_mask):
        x = self.embed(trg)
        x = self.pe(x)
        for i in range(self.N):
            x = self.layers[i](x, trg_mask)
        return self.norm(x)


class Transformer(nn.Module):
    def __init__(self, trg_vocab, d_model, N, heads, dropout):
        super().__init__()
        self.decoder = Decoder(trg_vocab, d_model, N, heads, dropout)
        self.out = nn.Linear(d_model, trg_vocab)

    def forward(self, trg, trg_mask):
        # print("DECODER")
        d_output = self.decoder(trg, trg_mask)
        output = self.out(d_output)
        return output


# class TokenDataset(Dataset):
#     def __init__(self, tokens, seq_len):
#         self.tokens = tokens
#         self.seq_len = seq_len

#     def __len__(self):
#         return len(self.tokens) - self.seq_len

#     def __getitem__(self, idx):
#         X = torch.tensor(self.tokens[idx : idx + self.seq_len])
#         y = torch.tensor(self.tokens[idx + 1 : idx + self.seq_len + 1])
#         return X, y


def data_generator(data, batch_size, seq_len, device, tokenizer):
    """Generates batches of data with padding.

    Args:
        data (list):  List of token IDs (output of read_corpus).
        batch_size (int): The desired batch size.
        seq_len (int):  Maximum sequence length.
        tokenizer:   The tokenizer.

    Yields:
        torch.Tensor:  Padded batch of input sequences (shape: batch_size, seq_len).
        torch.Tensor:  Padded batch of target sequences (shape: batch_size, seq_len).
    """
    for i in range(0, len(data) - seq_len, seq_len * batch_size):  # modified the loop
        batch_data = data[i : i + seq_len * batch_size]

        # Create input and target sequences
        inputs = []
        targets = []
        for j in range(0, len(batch_data), seq_len):
            inputs.append(batch_data[j : j + seq_len - 1])
            targets.append(batch_data[j + 1 : j + seq_len])

        # Pad sequences to seq_len
        padded_inputs = []
        padded_targets = []

        for seq in inputs:
            padding_len = seq_len - len(seq)
            padded_seq = seq + [tokenizer.pad_token_id] * padding_len
            padded_inputs.append(padded_seq)

        for seq in targets:
            padding_len = seq_len - len(seq)
            padded_seq = seq + [tokenizer.pad_token_id] * padding_len
            padded_targets.append(padded_seq)

        input_tensor = torch.tensor(padded_inputs, dtype=torch.long).to(device)  # global opt
        target_tensor = torch.tensor(padded_targets, dtype=torch.long).to(device)

        yield input_tensor, target_tensor


def get_model(opt, trg_vocab):

    assert opt.d_model % opt.heads == 0
    assert opt.dropout < 1

    model = Transformer(trg_vocab, opt.d_model, opt.n_layers, opt.heads, opt.dropout)
    model.to(opt.device)

    if opt.loadname is not None:
        print("loading pretrained weights...", flush=True)
        model.load_state_dict(torch.load(opt.loadname))
    else:
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    return model


def train_model(model, opt, tokenizer):

    print("training model...", flush=True)
    model.train()
    # dataset = TokenDataset(opt.train, opt.seqlen)  # or whatever length you want
    # opt.train = DataLoader(dataset, opt.batchsize, shuffle=True)
    savedPs = []
    trg_mask = torch.tril(torch.ones(1, opt.d_model, opt.d_model, device=opt.device)).bool()
    for i in range(opt.epochs):
        startTime = time.time()
        totalTokens = 0
        for batch, (input_batch, target_batch) in enumerate(
            data_generator(opt.train, opt.batchsize, opt.seqlen, opt.device, tokenizer)
        ):
            pred = model(input_batch, trg_mask)
            loss = F.cross_entropy(pred.view(-1, opt.vocab_size), target_batch.view(-1))
            loss.backward()
            opt.optimizer.step()
            totalTokens += input_batch.numel() + target_batch.numel()
            if batch % 100 == 0:
                print(
                    f"Loss: {loss.item():.4f}, Perplexity: {ppl:.4f}",
                    flush=True,
                )
        ppl = math.exp(loss.item())
        savedPs.append(ppl)
        with open('perps.pickle', 'wb') as file:
            pickle.dump(savedPs, file)
        test_model(model, opt, i, tokenizer, trg_mask)
        torch.save(model.state_dict(), f"{opt.savename}/epoch{i+1}.pth")
        print("Epoch", i + 1, " Done", flush=True)
        print("Rate: ", totalTokens / (time.time() - startTime), " TPS")
        # test_model(model, opt, i)
    print("Final Perplexity: ", math.exp(loss.item()), flush=True)
    #  7. generate a test perplexity once per training epoch by calling test_model()
    #  8. save model weights to file specified in opt.savename
    #  SEE trainer.py for examples of each of the above


def test_model(model, opt, epoch, tokenizer, mask):
    print("testing model...", flush=True)
    model.eval()
    total_loss = 0
    total_tokens = 0
    with torch.no_grad():
        for batch, (input_batch, target_batch) in enumerate(
            data_generator(opt.test, opt.batchsize, opt.seqlen, opt.device, tokenizer)
        ):
            pred = model(input_batch, mask)

            # Reshape for cross entropy
            # pred_flat = pred.reshape(opt.batchsize * opt.seqlen, -1)
            pred_flat = pred.view(-1, pred.size(-1))  # [batch_size * seq_len, vocab_size]
            target_flat = target_batch.view(-1)  # [batch_size * seq_len]
            # target_flat = target_batch.reshape(opt.batchsize * opt.seqlen)

            # Calculate loss
            loss = F.cross_entropy(pred_flat, target_flat)
            savedTestPs.append(math.exp(loss.item()))
            total_loss += loss.item()
            total_tokens += len(target_flat)

            # Calculate accuracy
            correct = (pred_flat.argmax(1) == target_flat).sum().item()

    avg_loss = total_loss / (batch + 1)
    perplexity = math.exp(avg_loss)
    accuracy = correct / total_tokens

    with open('perpsTest.pickle', 'wb') as file:
                    pickle.dump(savedTestPs, file)

    print(
        f'Test Error for Epoch {epoch}: \n Accuracy: {(100 * accuracy):>0.1f}%, Avg loss: {avg_loss:>8f}, Perplexity: {perplexity:>8f}\n',
        flush=True,
    )
    model.train()


def main():

    random.seed(10)

    parser = argparse.ArgumentParser()
    parser.add_argument('-no_cuda', action='store_true')
    parser.add_argument('-SGDR', action='store_true')
    parser.add_argument('-epochs', type=int, default=20)
    parser.add_argument('-d_model', type=int, default=512)
    parser.add_argument('-n_layers', type=int, default=6)
    parser.add_argument('-heads', type=int, default=8)
    parser.add_argument('-dropout', type=int, default=0.1)
    parser.add_argument('-batchsize', type=int, default=32)
    parser.add_argument('-printevery', type=int, default=100)
    parser.add_argument('-lr', type=int, default=0.00005)
    parser.add_argument('-seqlen', type=int, default=512)
    parser.add_argument('-threshold', type=int, default=3)
    parser.add_argument('-savename', type=str, default='modelWeights')
    parser.add_argument('-loadname', type=str)
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
    opt.train = read_corpus('wiki2.train.txt', tokenizer)
    opt.valid = read_corpus('wiki2.valid.txt', tokenizer)
    opt.test = read_corpus('wiki2.test.txt', tokenizer)

    opt.vocab_size = tokenizer.vocab_size
    print('Vocab size: ', opt.vocab_size)
    temp = []
    for i in range(opt.vocab_size):
        temp.append(i)
    opt.indices = torch.tensor(temp)
    opt.indices = opt.indices.cuda()

    model = get_model(opt, opt.vocab_size)

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    text = 'total params: %d' % (params)
    print(text, flush=True)

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

    train_model(model, opt, tokenizer)


if __name__ == "__main__":
    main()

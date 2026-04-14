import torch
import torch.nn as nn
import torch.optim as optim
import math
import time
import os
import json
import urllib.request
from collections import Counter
from torch.utils.data import Dataset, DataLoader

# ============================================================
# Device Setup
# ============================================================

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ============================================================
# Download and Load CoNLL-2003 Dataset
# ============================================================

os.makedirs("data", exist_ok=True)

base_url = "https://raw.githubusercontent.com/synalp/NER/master/corpus/CoNLL-2003/"

files = {
    "train.txt": "eng.train",
    "valid.txt": "eng.testa",
    "test.txt": "eng.testb"
}

for local_name, remote_name in files.items():
    path = f"data/{local_name}"
    if not os.path.exists(path):
        print(f"Downloading {remote_name}...")
        urllib.request.urlretrieve(base_url + remote_name, path)

print("Dataset ready!")

def load_conll(filepath):
    sentences = []
    labels = []
    current_tokens = []
    current_labels = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line == "" or line.startswith("-DOCSTART-"):
                if current_tokens:
                    sentences.append(current_tokens)
                    labels.append(current_labels)
                    current_tokens = []
                    current_labels = []
                continue
            parts = line.split()
            current_tokens.append(parts[0])
            current_labels.append(parts[3])
        if current_tokens:
            sentences.append(current_tokens)
            labels.append(current_labels)
    return sentences, labels

train_sentences, train_labels = load_conll("data/train.txt")
val_sentences, val_labels = load_conll("data/valid.txt")
test_sentences, test_labels = load_conll("data/test.txt")

print(f"Train: {len(train_sentences)} sentences")
print(f"Val:   {len(val_sentences)} sentences")
print(f"Test:  {len(test_sentences)} sentences")

# ============================================================
# Convert IOB1 to IOB2
# ============================================================

def convert_to_iob2(labels):
    new_labels = []
    for i, label in enumerate(labels):
        if label.startswith("I-"):
            if i == 0 or labels[i-1] == "O" or labels[i-1].split("-")[1] != label.split("-")[1]:
                new_labels.append("B-" + label.split("-")[1])
            else:
                new_labels.append(label)
        else:
            new_labels.append(label)
    return new_labels

train_labels = [convert_to_iob2(labels) for labels in train_labels]
val_labels = [convert_to_iob2(labels) for labels in val_labels]
test_labels = [convert_to_iob2(labels) for labels in test_labels]

# ============================================================
# Label Mappings
# ============================================================

label_list = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC"]
label2idx = {label: idx for idx, label in enumerate(label_list)}
idx2label = {idx: label for label, idx in label2idx.items()}
NUM_LABELS = len(label_list)

# ============================================================
# Vocabulary
# ============================================================

PAD_IDX = 0
UNK_IDX = 1

def build_vocab(sentences, min_freq=1):
    counter = Counter()
    for sent in sentences:
        counter.update(sent)
    token2idx = {'<pad>': 0, '<unk>': 1}
    for word, freq in counter.items():
        if freq >= min_freq and word not in token2idx:
            token2idx[word] = len(token2idx)
    idx2token = {idx: tok for tok, idx in token2idx.items()}
    return token2idx, idx2token

token2idx, idx2token = build_vocab(train_sentences, min_freq=1)
print(f"Vocab size: {len(token2idx)}")

# ============================================================
# Dataset and DataLoader
# ============================================================

LABEL_PAD_IDX = -100

class NERDataset(Dataset):
    def __init__(self, sentences, labels, token2idx, label2idx):
        self.sentences = sentences
        self.labels = labels
        self.token2idx = token2idx
        self.label2idx = label2idx

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        label = self.labels[idx]
        token_indices = [self.token2idx.get(word, UNK_IDX) for word in sentence]
        label_indices = [self.label2idx[lab] for lab in label]
        return torch.tensor(token_indices, dtype=torch.long), torch.tensor(label_indices, dtype=torch.long)

def collate_fn(batch):
    tokens_batch, labels_batch = zip(*batch)
    max_len = max(len(seq) for seq in tokens_batch)
    tokens_padded = []
    labels_padded = []
    for tokens, labels in zip(tokens_batch, labels_batch):
        token_padding = torch.full((max_len - len(tokens),), PAD_IDX, dtype=torch.long)
        label_padding = torch.full((max_len - len(labels),), LABEL_PAD_IDX, dtype=torch.long)
        tokens_padded.append(torch.cat([tokens, token_padding]))
        labels_padded.append(torch.cat([labels, label_padding]))
    return torch.stack(tokens_padded), torch.stack(labels_padded)

BATCH_SIZE = 64

train_dataset = NERDataset(train_sentences, train_labels, token2idx, label2idx)
val_dataset = NERDataset(val_sentences, val_labels, token2idx, label2idx)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

# ============================================================
# Model Components
# ============================================================

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn_weights = torch.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, V)
        return output

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)
        Q = Q.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.W_o(attn_output)
        return output

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x

class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, n_heads)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask):
        attn_output = self.self_attention(x, x, x, mask)
        attn_output = self.dropout1(attn_output)
        x = self.norm1(x + attn_output)
        ff_output = self.feed_forward(x)
        ff_output = self.dropout2(ff_output)
        x = self.norm2(x + ff_output)
        return x

class NERTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, d_ff, n_layers, num_labels, dropout=0.1):
        super().__init__()
        self.token_embedding = TokenEmbedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout=dropout)
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.classifier = nn.Linear(d_model, num_labels)

    def forward(self, src, mask):
        x = self.token_embedding(src)
        x = self.positional_encoding(x)
        for layer in self.layers:
            x = layer(x, mask)
        output = self.classifier(x)
        return output

# ============================================================
# Masking
# ============================================================

def create_mask(src):
    mask = (src != PAD_IDX).unsqueeze(1).unsqueeze(2)
    return mask

# ============================================================
# Training and Evaluation
# ============================================================

def train_one_epoch(model, dataloader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0
    total_tokens = 0
    for tokens, labels in dataloader:
        tokens = tokens.to(device)
        labels = labels.to(device)
        mask = create_mask(tokens).to(device)
        output = model(tokens, mask)
        output = output.reshape(-1, output.size(-1))
        labels = labels.reshape(-1)
        loss = loss_fn(output, labels)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        non_pad = (labels != LABEL_PAD_IDX).sum().item()
        total_loss += loss.item() * non_pad
        total_tokens += non_pad
    return total_loss / total_tokens

def evaluate(model, dataloader, loss_fn, device):
    model.eval()
    total_loss = 0
    total_tokens = 0
    with torch.no_grad():
        for tokens, labels in dataloader:
            tokens = tokens.to(device)
            labels = labels.to(device)
            mask = create_mask(tokens).to(device)
            output = model(tokens, mask)
            output = output.reshape(-1, output.size(-1))
            labels = labels.reshape(-1)
            loss = loss_fn(output, labels)
            non_pad = (labels != LABEL_PAD_IDX).sum().item()
            total_loss += loss.item() * non_pad
            total_tokens += non_pad
    return total_loss / total_tokens

def compute_accuracy(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for tokens, labels in dataloader:
            tokens = tokens.to(device)
            labels = labels.to(device)
            mask = create_mask(tokens).to(device)
            output = model(tokens, mask)
            predictions = output.argmax(dim=-1)
            non_pad_mask = (labels != LABEL_PAD_IDX)
            correct += (predictions[non_pad_mask] == labels[non_pad_mask]).sum().item()
            total += non_pad_mask.sum().item()
    return correct / total

# ============================================================
# Hyperparameters
# ============================================================

D_MODEL = 256
N_HEADS = 8
D_FF = 1024
N_LAYERS = 3
DROPOUT = 0.1
LEARNING_RATE = 3e-4
EPOCHS = 20

# ============================================================
# Create Model
# ============================================================

model = NERTransformer(
    vocab_size=len(token2idx),
    d_model=D_MODEL,
    n_heads=N_HEADS,
    d_ff=D_FF,
    n_layers=N_LAYERS,
    num_labels=NUM_LABELS,
    dropout=DROPOUT
).to(device)

loss_fn = nn.CrossEntropyLoss(ignore_index=LABEL_PAD_IDX)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.98), eps=1e-9)

total_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {total_params:,}")

# ============================================================
# Training Loop
# ============================================================

print("\nStarting training...\n")

for epoch in range(EPOCHS):
    start_time = time.time()
    train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
    val_loss = evaluate(model, val_loader, loss_fn, device)
    val_acc = compute_accuracy(model, val_loader, device)
    elapsed = time.time() - start_time
    print(f"Epoch {epoch+1:02d}/{EPOCHS} | "
          f"Train Loss: {train_loss:.4f} | "
          f"Val Loss: {val_loss:.4f} | "
          f"Val Acc: {val_acc:.4f} | "
          f"Time: {elapsed:.1f}s")

# ============================================================
# Save Model
# ============================================================

save_dir = "ner_model"
os.makedirs(save_dir, exist_ok=True)

torch.save(model.state_dict(), f"{save_dir}/model.pt")

with open(f"{save_dir}/token2idx.json", "w") as f:
    json.dump(token2idx, f)

with open(f"{save_dir}/label2idx.json", "w") as f:
    json.dump(label2idx, f)

with open(f"{save_dir}/idx2label.json", "w") as f:
    json.dump({str(k): v for k, v in idx2label.items()}, f)

config = {
    "vocab_size": len(token2idx),
    "d_model": D_MODEL,
    "n_heads": N_HEADS,
    "d_ff": D_FF,
    "n_layers": N_LAYERS,
    "num_labels": NUM_LABELS,
    "dropout": DROPOUT
}

with open(f"{save_dir}/config.json", "w") as f:
    json.dump(config, f)

print(f"\nModel saved to {save_dir}/")

# ============================================================
# Test Predictions
# ============================================================

import re

def tokenize(text):
    text = text.strip()
    tokens = re.findall(r"[\w]+|[^\s\w]", text)
    return tokens

def predict_ner(sentence):
    model.eval()
    tokens = tokenize(sentence)
    token_indices = [token2idx.get(tok, UNK_IDX) for tok in tokens]
    src = torch.tensor(token_indices, dtype=torch.long).unsqueeze(0).to(device)
    mask = create_mask(src).to(device)
    with torch.no_grad():
        output = model(src, mask)
    predictions = output.argmax(dim=-1).squeeze(0)
    results = []
    for tok, pred_idx in zip(tokens, predictions):
        label = idx2label[pred_idx.item()]
        results.append((tok, label))
    return results

print("\n=== Sample Predictions ===\n")

test_sentences = [
    "Barack Obama visited Google in New York.",
    "The European Union rejected the proposal.",
    "Apple CEO Tim Cook announced a new product in California.",
    "Manchester United signed a player from Brazil.",
]

for sent in test_sentences:
    results = predict_ner(sent)
    print(f"Sentence: {sent}")
    for word, label in results:
        if label != "O":
            print(f"  {word:20s} -> {label}")
    print()

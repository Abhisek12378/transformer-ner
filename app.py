import torch
import torch.nn as nn
import math
import json
import re
import gradio as gr

# ============================================================
# Model Architecture
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

PAD_IDX = 0
UNK_IDX = 1

def create_mask(src):
    mask = (src != PAD_IDX).unsqueeze(1).unsqueeze(2)
    return mask

# ============================================================
# Load Model
# ============================================================

device = torch.device('cpu')

with open("config.json", "r") as f:
    config = json.load(f)

with open("token2idx.json", "r") as f:
    token2idx = json.load(f)

with open("idx2label.json", "r") as f:
    idx2label = {int(k): v for k, v in json.load(f).items()}

model = NERTransformer(
    vocab_size=config["vocab_size"],
    d_model=config["d_model"],
    n_heads=config["n_heads"],
    d_ff=config["d_ff"],
    n_layers=config["n_layers"],
    num_labels=config["num_labels"],
    dropout=config["dropout"]
)

model.load_state_dict(torch.load("model.pt", map_location=device))
model.eval()

# ============================================================
# Tokenizer
# ============================================================

def tokenize(text):
    text = text.strip()
    tokens = re.findall(r"[\w]+|[^\s\w]", text)
    return tokens

# ============================================================
# Color mapping for entities
# ============================================================

ENTITY_COLORS = {
    "PER": "#FF6B6B",
    "ORG": "#4ECDC4",
    "LOC": "#45B7D1",
    "MISC": "#96CEB4",
}

# ============================================================
# Prediction Function
# ============================================================

def predict_ner(sentence):
    if not sentence.strip():
        return "Please enter a sentence."

    tokens = tokenize(sentence)
    token_indices = [token2idx.get(tok, UNK_IDX) for tok in tokens]
    src = torch.tensor(token_indices, dtype=torch.long).unsqueeze(0)
    mask = create_mask(src)

    with torch.no_grad():
        output = model(src, mask)

    predictions = output.argmax(dim=-1).squeeze(0)

    # Build formatted output
    entities = []
    result_parts = []

    for tok, pred_idx in zip(tokens, predictions):
        label = idx2label[pred_idx.item()]
        if label != "O":
            entity_type = label.split("-")[1]
            entities.append(f"{tok} → {label}")
            result_parts.append((tok, entity_type))
        else:
            result_parts.append((tok, None))

    # Build highlighted text
    highlighted = []
    for tok, entity_type in result_parts:
        if entity_type:
            highlighted.append((tok, entity_type))
        else:
            highlighted.append((tok, None))

    # Build entity summary
    if entities:
        summary = "Detected entities:\n" + "\n".join(entities)
    else:
        summary = "No entities detected."

    return highlighted, summary

# ============================================================
# Gradio Interface
# ============================================================

examples = [
    "Barack Obama visited Google in New York.",
    "The European Union rejected the proposal.",
    "Apple CEO Tim Cook announced a new product in California.",
    "Manchester United signed a player from Brazil.",
    "The United Nations held a meeting in Geneva about climate change.",
]

demo = gr.Interface(
    fn=predict_ner,
    inputs=gr.Textbox(label="Enter a sentence", placeholder="Type an English sentence..."),
    outputs=[
        gr.HighlightedText(label="NER Tags", combine_adjacent=True,
                          color_map={"PER": "#FF6B6B", "ORG": "#4ECDC4", "LOC": "#45B7D1", "MISC": "#96CEB4"}),
        gr.Textbox(label="Detected Entities")
    ],
    title="Named Entity Recognition (NER)",
    description="Transformer encoder model built from scratch using PyTorch. Trained on CoNLL-2003 dataset (~14K sentences). Architecture: 3 encoder layers, 8 attention heads, d_model=256. Detects: PER (persons), ORG (organizations), LOC (locations), MISC (miscellaneous).",
    examples=examples,
    theme=gr.themes.Soft()
)

if __name__ == "__main__":
    demo.launch()

import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
import random
import pickle
from transformers import GPT2Tokenizer

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

model_to_save = "model-01.pkl" # change this based on how you want to save your model

###############################################################################
# Hyperparameters
###############################################################################
batch_size = 16               
block_size = 512               
max_iters = 50000
learning_rate = 3e-4
eval_iters = 500
n_embd = 768
n_head = 4
n_layer = 8
dropout = 0.2

###############################################################################
# Load GPT-2 Tokenizer
###############################################################################
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
vocab_size = tokenizer.vocab_size
print(f"Using GPT-2 Tokenizer with vocab size: {vocab_size}")

###############################################################################
# Preload and Pre-tokenize Entire Data from train_split.txt and val_split.txt
###############################################################################
def load_and_tokenize(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    tokens = tokenizer.encode(text)
    return torch.tensor(tokens, dtype=torch.long)

train_data = load_and_tokenize("train_split.txt")
val_data = load_and_tokenize("val_split.txt")
print(f"Total training tokens: {train_data.size(0)}")
print(f"Total validation tokens: {val_data.size(0)}")

###############################################################################
# Batch Sampling from Pre-tokenized Data
###############################################################################
def get_batch(data):
    # Randomly choose starting indices such that there is room for a block.
    i = torch.randint(0, len(data) - block_size, (batch_size,))
    x = torch.stack([data[j:j+block_size] for j in i])
    y = torch.stack([data[j+1:j+block_size+1] for j in i])
    return x.to(device), y.to(device)

###############################################################################
# Model Components (Attention, Feed-Forward, Transformer Block)
###############################################################################
class Head(nn.Module):
    """One head of self-attention"""
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * (1.0 / (k.shape[-1] ** 0.5))
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    """Multiple heads of self-attention in parallel"""
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    """Fully connected feed-forward network"""
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """Transformer block: self-attention followed by feed-forward"""
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

###############################################################################
# GPTLanguageModel Definition
###############################################################################
class GPTLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.apply(self._init_weights)
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
    def forward(self, idx, targets=None):
        B, T = idx.shape
        if idx.max().item() >= vocab_size:
            raise ValueError(f"Token index out of range: {idx.max().item()} >= {vocab_size}")
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= block_size else idx[:, -block_size:]
            logits, _ = self.forward(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

###############################################################################
# Training Loop
###############################################################################
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split, data in zip(['train', 'val'], [train_data, val_data]):
        losses = torch.zeros(eval_iters, device=device)
        for k in range(eval_iters):
            X, Y = get_batch(data)
            _, loss = model(X, Y)
            losses[k] = loss
        out[split] = losses.mean().item()
    model.train()
    return out

model = GPTLanguageModel(vocab_size).to(device)

# Optionally, load an existing checkpoint
try:
    with open(model_to_save, 'rb') as f:
        model = pickle.load(f).to(device)
    print("Loaded existing model-final.pkl!")
except FileNotFoundError:
    print("No existing model file found. Training from scratch.")

optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_iters, eta_min=1e-5)

print(f"Starting training with batch_size={batch_size}, block_size={block_size}, lr={learning_rate}, max_iters={max_iters}")

for step in range(max_iters):
    if step % eval_iters == 0:
        losses = estimate_loss()
        print(f"step: {step}, train loss: {losses['train']:.3f}, val loss: {losses['val']:.3f}")
    xb, yb = get_batch(train_data)
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    scheduler.step()
    if step % 500 == 0:
        with open('model-checkpoint.pkl', 'wb') as f:
            pickle.dump(model, f)
        print(f"Checkpoint saved at step {step}")

print(f"Final training step {max_iters}, last training loss = {loss.item():.4f}")

final_model_path = model_to_save
with open(final_model_path, 'wb') as f:
    pickle.dump(model, f)

print(f"✅ Training complete! Final model saved to {final_model_path}")

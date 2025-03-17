import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
import mmap
import random
import pickle
from transformers import GPT2Tokenizer


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

model_to_save = "model-01.pkl" # change this based on how you want to save your model

###############################################################################
# Hyperparameters
###############################################################################
batch_size = 8                     # bigger batch size for smoother updates
block_size = 256                   # length of the sequence to train on
max_iters = 20_000                 # how many total steps to train
learning_rate = 3e-4
eval_iters = 500
n_embd = 512
n_head = 4
n_layer = 6
dropout = 0.2

###############################################################################
# Load vocabulary
###############################################################################
with open("vocab.txt", 'r', encoding='utf-8') as f:
    vocab_text = f.read()
chars = sorted(list(set(vocab_text)))
vocab_size = len(chars)

string_to_int = { ch: i for i, ch in enumerate(chars) }
int_to_string = { i: ch for i, ch in enumerate(chars) }

encode = lambda s: [string_to_int[c] for c in s]
decode = lambda l: ''.join([int_to_string[i] for i in l])

print(f"Vocab size: {vocab_size}")

###############################################################################
# Random chunk loader
###############################################################################
def get_random_chunk(split):
    """ 
    Reads a random chunk from train_split.txt or val_split.txt, 
    filters out-of-vocab chars, returns a tensor of token indices.
    """
    filename = "train_split.txt" if split == 'train' else "val_split.txt"
    with open(filename, 'rb') as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)

        file_size = len(mm)
        # We might need multiple draws if data near the end is short or invalid
        while True:
            # pick a random start position that can (hopefully) fit block_size*batch_size
            if file_size < block_size * batch_size:
                raise ValueError(f"File {filename} is too small to read {block_size*batch_size} bytes.")
            
            start_pos = random.randint(0, file_size - block_size * batch_size)
            mm.seek(start_pos)
            # Read enough bytes for block_size*batch_size
            block = mm.read(block_size * batch_size)

            # Decode and filter unknown chars
            decoded_block = block.decode('utf-8', errors='ignore').replace('\r', '')
            filtered_block = ''.join(ch for ch in decoded_block if ch in string_to_int)

            data_tensor = torch.tensor(encode(filtered_block), dtype=torch.long)

            # Only return if we have at least block_size+1 tokens
            if data_tensor.size(0) > block_size:
                mm.close()
                return data_tensor
            # Otherwise, loop again to pick another chunk

def get_batch(split):
    """
    Returns x, y each of shape (batch_size, block_size).
    Ensures no out-of-bounds indexing by redrawing if chunk is too short.
    """
    while True:
        data = get_random_chunk(split)
        if data.size(0) <= block_size:
            continue  # skip short chunk

        try:
            ix = torch.randint(0, data.size(0) - block_size, (batch_size,))
            x = torch.stack([data[i : i + block_size] for i in ix])
            y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
            x, y = x.to(device), y.to(device)

            # optional debug check for out-of-range
            if x.max().item() >= vocab_size or x.min().item() < 0:
                print("Found token index out of range! Redrawing chunk...")
                continue

            return x, y
        except Exception as e:
            # if for some reason stacking fails, try again
            print("Error stacking batch:", e, " -- Retrying...")
            continue

###############################################################################
# Model definition
###############################################################################
class Head(nn.Module):
    """ one head of self-attention """
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)   # (B,T,hs)
        q = self.query(x) # (B,T,hs)

        # compute attention scores
        wei = q @ k.transpose(-2, -1) * (1.0 / (k.shape[-1] ** 0.5)) # (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        # weighted aggregation
        v = self.value(x)
        out = wei @ v  # (B,T,hs)
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """
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
    """ a simple linear layer followed by a non-linearity """
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
    """ Transformer block: communication followed by computation """
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

class GPTLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
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
        # Debug check for out-of-range
        if idx.max() >= vocab_size:
            raise ValueError(f"Token index out of range: max={idx.max().item()}, vocab_size={vocab_size}")

        tok_emb = self.token_embedding_table(idx)               # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))  # (T,C)
        x = tok_emb + pos_emb                                   # (B,T,C)
        x = self.blocks(x)                                      # (B,T,C)
        x = self.ln_f(x)                                        # (B,T,C)
        logits = self.lm_head(x)                                # (B,T,vocab_size)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # if sequence grows beyond block_size, we can slice off the last block_size tokens
            idx_condensed = idx if idx.size(1) <= block_size else idx[:, -block_size:]

            logits, _ = self.forward(idx_condensed)
            logits = logits[:, -1, :] # take the last time step
            probs = F.softmax(logits, dim=-1) # (B, vocab_size)
            idx_next = torch.multinomial(probs, num_samples=1)  # (B,1)
            idx = torch.cat((idx, idx_next), dim=1)             # (B, T+1)
        return idx

###############################################################################
# Training
###############################################################################
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters, device=device)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss
        out[split] = losses.mean().item()
    model.train()
    return out

model = GPTLanguageModel(vocab_size).to(device)

# Optionally, load a pre-trained checkpoint if you have a model-02.pkl
# with the same block_size, n_embd, etc.:
try:
    with open(model_to_save, 'rb') as f:
        model = pickle.load(f).to(device)
    print("Loaded existing model-02.pkl!")
except FileNotFoundError:
    print("No existing model file found. Training from scratch.")

optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

# Example: Cosine learning-rate scheduler
# Let's say we do a full cosine cycle over `max_iters` steps
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_iters, eta_min=1e-5)

print(f"Starting training with batch_size={batch_size}, block_size={block_size}, "
      f"lr={learning_rate}, max_iters={max_iters}")

for step in range(max_iters):
    # Evaluate loss on train/val set periodically
    if step % eval_iters == 0:
        losses = estimate_loss()
        print(f"step: {step}, train loss: {losses['train']:.3f}, val loss: {losses['val']:.3f}")

    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    scheduler.step()  # update LR

    if step % 500 == 0:
        with open('model-checkpoint.pkl', 'wb') as f:
            pickle.dump(model, f)
        print(f"Checkpoint saved at step {step}")

# final loss
print(f"Final training step {max_iters}, last training loss = {loss.item():.4f}")

# Save the trained model
with open(model_to_save, 'wb') as f:
    pickle.dump(model, f)

print("Model saved to model-02.pkl")

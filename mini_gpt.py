# =============================================================================
# MINI GPT — Built from scratch
# Run this on Kaggle (T4 GPU). Should train in ~5 minutes.
# Dataset: Tiny Shakespeare (~1MB of text)
# What it learns: given a sequence of characters, predict the next one.
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import urllib.request
import os

# ── 0. DEVICE SETUP ──────────────────────────────────────────────────────────
# PyTorch can run on CPU or GPU. "cuda" = NVIDIA GPU (what Kaggle gives you).
# .to(device) moves tensors/models onto that hardware.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using: {device}")


# ── 1. DATASET ───────────────────────────────────────────────────────────────
# We use Tiny Shakespeare — one long string of ~1M characters.
# The model's only job: given the last N characters, guess the next one.

url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
if not os.path.exists("input.txt"):
    urllib.request.urlretrieve(url, "input.txt")

text = open("input.txt").read()
print(f"Dataset size: {len(text):,} characters")
print(f"Sample: {text[:80]!r}")


# ── 2. TOKENIZER (character-level) ───────────────────────────────────────────
# Real models use subword tokenizers (BPE). We use characters — simpler to
# understand. Each unique character gets an integer ID.
#
# vocab_size is the size of the "output layer" — at each step the model
# outputs one logit (z_i) per character in the vocabulary.

chars = sorted(set(text))
vocab_size = len(chars)
print(f"Vocabulary: {vocab_size} unique characters")
print(f"Chars: {''.join(chars)!r}")

# Two lookup tables: char → int, int → char
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

encode = lambda s: [stoi[c] for c in s]   # "hi" → [32, 37]
decode = lambda l: "".join(itos[i] for i in l)  # [32, 37] → "hi"

# Encode the entire dataset as a 1D tensor of integers
data = torch.tensor(encode(text), dtype=torch.long)  # shape: [1_115_394]


# ── 3. TRAIN / VAL SPLIT ─────────────────────────────────────────────────────
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]


# ── 4. HYPERPARAMETERS ───────────────────────────────────────────────────────
# Keep these small so it trains fast on T4. You can scale up on the L4.

block_size  = 64    # context length: how many characters the model sees at once
batch_size  = 64    # how many sequences to process in parallel
n_embd      = 128   # embedding dimension (d_model) — size of each token vector
n_heads     = 4     # number of attention heads (n_embd must be divisible by n_heads)
n_layers    = 4     # how many transformer blocks to stack
dropout     = 0.1   # randomly zero out 10% of activations during training (regularization)
learning_rate = 3e-4
max_iters   = 3000
eval_interval = 300
eval_iters  = 100


# ── 5. BATCHING ───────────────────────────────────────────────────────────────
# A "batch" is a random sample of (input, target) pairs.
#
# For each sequence of length block_size, the targets are the same sequence
# shifted by 1. This way we get block_size training examples per sequence:
#
#   input:  [h, e, l, l, o]
#   target: [e, l, l, o, !]
#   ↑ at position 0, given "h", predict "e"
#   ↑ at position 1, given "h,e", predict "l"   ... etc.

def get_batch(split):
    data_ = train_data if split == "train" else val_data
    # Pick batch_size random starting positions
    ix = torch.randint(len(data_) - block_size, (batch_size,))
    # Stack into matrices of shape [batch_size, block_size]
    x = torch.stack([data_[i:i+block_size] for i in ix])
    y = torch.stack([data_[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)


# ── 6. SINGLE ATTENTION HEAD ─────────────────────────────────────────────────
# This is the core of the transformer. Read this carefully.
#
# Every token produces three vectors by multiplying its embedding by learned
# weight matrices:
#   Query (Q): "what am I looking for?"
#   Key   (K): "what do I contain?"
#   Value (V): "what do I actually send to others?"
#
# Attention score between token i and token j = dot(Q_i, K_j) / sqrt(head_size)
# Softmax converts scores → weights → weighted sum of Values.
#
# The division by sqrt(head_size) prevents dot products from growing too large
# (which would push softmax into flat regions where gradients vanish).
#
# "Causal" masking (tril): token i can only attend to tokens 0..i.
# This is what makes it a language model — no peeking at future tokens.

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        # Three linear projections — no bias (standard practice)
        self.key   = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        # tril is not a parameter, just a constant mask — register as buffer
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x shape: [batch, seq_len, n_embd]
        B, T, C = x.shape
        k = self.key(x)    # [B, T, head_size]
        q = self.query(x)  # [B, T, head_size]
        v = self.value(x)  # [B, T, head_size]

        # Attention scores: Q @ K^T / sqrt(head_size)
        # Result shape: [B, T, T] — every token vs every token
        head_size = k.shape[-1]
        scores = q @ k.transpose(-2, -1) * (head_size ** -0.5)

        # Causal mask: fill future positions with -inf so softmax → 0
        scores = scores.masked_fill(self.tril[:T, :T] == 0, float("-inf"))

        # Softmax over the last dim → attention weights that sum to 1
        weights = F.softmax(scores, dim=-1)  # [B, T, T]
        weights = self.dropout(weights)

        # Weighted sum of values
        out = weights @ v  # [B, T, head_size]
        return out


# ── 7. MULTI-HEAD ATTENTION ───────────────────────────────────────────────────
# Run n_heads attention heads in parallel, each with head_size = n_embd/n_heads.
# Each head learns to attend to different kinds of relationships.
# Concatenate their outputs and project back to n_embd.

class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(n_heads)])
        # Projection back to n_embd after concatenation
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Run all heads, concat on last dim: [B, T, n_heads * head_size] = [B, T, n_embd]
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


# ── 8. FEED-FORWARD BLOCK ────────────────────────────────────────────────────
# After attention, each token is processed independently through a small MLP.
# The 4x expansion (n_embd → 4*n_embd → n_embd) is standard from the original
# "Attention is All You Need" paper.
#
# Why do we need this after attention?
# Attention mixes information across tokens (communication).
# FFN processes each token individually (computation / reasoning on that info).

class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),           # smoother than ReLU, used in GPT-2/3/4
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


# ── 9. TRANSFORMER BLOCK ─────────────────────────────────────────────────────
# One block = attention + feed-forward, each wrapped with:
#   - LayerNorm: normalizes each token's vector to mean=0, std=1.
#     Stabilizes training. Applied *before* each sub-layer (Pre-LN style).
#   - Residual connection: output = x + sublayer(x)
#     Lets gradients flow directly from output to input during backprop.
#     Without this, deep networks fail to train (vanishing gradients).

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        head_size = n_embd // n_heads
        self.attn = MultiHeadAttention(n_heads, head_size)
        self.ff   = FeedForward()
        self.ln1  = nn.LayerNorm(n_embd)
        self.ln2  = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))  # residual around attention
        x = x + self.ff(self.ln2(x))    # residual around feed-forward
        return x


# ── 10. THE FULL MODEL ────────────────────────────────────────────────────────
# Putting it all together:
#   token_embedding:    int → vector of size n_embd
#   position_embedding: position index → vector of size n_embd
#   Both are learned. Added together → each token knows what it is AND where it is.
#   Then N transformer blocks.
#   Then LayerNorm + linear to vocab_size → the logits z_i we discussed.

class MiniGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb   = nn.Embedding(block_size, n_embd)
        self.blocks    = nn.Sequential(*[Block() for _ in range(n_layers)])
        self.ln_final  = nn.LayerNorm(n_embd)
        self.head      = nn.Linear(n_embd, vocab_size)  # produces logits z_i

    def forward(self, idx, targets=None):
        # idx: [B, T] — integer token IDs
        B, T = idx.shape

        tok  = self.token_emb(idx)                             # [B, T, n_embd]
        pos  = self.pos_emb(torch.arange(T, device=device))   # [T, n_embd]
        x    = tok + pos                                       # broadcast over batch

        x    = self.blocks(x)      # N transformer blocks
        x    = self.ln_final(x)
        logits = self.head(x)      # [B, T, vocab_size] — these are z_i values!

        if targets is None:
            return logits, None

        # Cross-entropy loss: how wrong were our predicted probabilities?
        # Internally: softmax(logits) → probs, then -log(prob of correct token)
        # PyTorch expects [B*T, vocab_size] and [B*T]
        B, T, C = logits.shape
        loss = F.cross_entropy(logits.view(B*T, C), targets.view(B*T))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0):
        """
        Autoregressively generate text.
        At each step:
          1. Run forward pass → logits z_i for the last position
          2. Divide by temperature (what we discussed!)
          3. Softmax → probability distribution
          4. Sample one token
          5. Append it and repeat
        """
        for _ in range(max_new_tokens):
            # Crop context to block_size (can't exceed what positional emb knows)
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            # Take logits at the last time step only
            logits = logits[:, -1, :]           # [B, vocab_size] → z_i values
            logits = logits / temperature        # temperature scaling
            probs  = F.softmax(logits, dim=-1)   # → probability distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # sample
            idx = torch.cat([idx, idx_next], dim=1)
        return idx


# ── 11. TRAINING ─────────────────────────────────────────────────────────────

model = MiniGPT().to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"\nModel parameters: {n_params:,}")

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

@torch.no_grad()
def estimate_loss():
    model.eval()
    losses = {}
    for split in ["train", "val"]:
        L = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            L[k] = loss.item()
        losses[split] = L.mean().item()
    model.train()
    return losses

print("\nTraining...\n")
for step in range(max_iters):
    if step % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {step:4d} | train loss {losses['train']:.4f} | val loss {losses['val']:.4f}")

    X, Y = get_batch("train")
    logits, loss = model(X, Y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print("\nDone training!")


# ── 12. GENERATION ───────────────────────────────────────────────────────────
# Try different temperatures — notice how the output changes.
# This is exactly what we visualized in the slider earlier.

print("\n" + "="*60)
for temp in [0.2, 0.8, 1.0]:
    print(f"\n--- Temperature = {temp} ---")
    context = torch.zeros((1, 1), dtype=torch.long, device=device)  # start with token 0
    generated = model.generate(context, max_new_tokens=200, temperature=temp)
    print(decode(generated[0].tolist()))
    print()

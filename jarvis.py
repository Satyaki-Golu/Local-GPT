import torch
import torch.nn as nn
from torch.nn import functional as F
import gradio as gr
import tiktoken
import time
import os

# =====================================================================
# 1. HYPERPARAMETERS & SYSTEM HARDWARE SETUP
# =====================================================================
batch_size = 16      # Parallel text sequence streams processed simultaneously
block_size = 128     # Maximum context window length (token sequence history)
max_iters = 200      # Lowered for fast initialization on Render's CPU tiers
eval_interval = 50   # Intervals at which validation performance is checked
learning_rate = 5e-4
eval_iters = 20
n_embd = 192         # Neural abstraction embedding dimension size
n_head = 6           # Concurrent Multi-Head Attention blocks
n_layer = 4          # Number of sequential Transformer blocks stacked
dropout = 0.1        # Random layer dropout rate to protect against overfitting

# Core execution target processing routing (Checks for Nvidia CUDA or Apple Silicon MPS)
if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'

print(f"🚀 Initializing system. Launching custom model architecture on: {device.upper()}")

# =====================================================================
# 2. TOY DATASET & SUB-WORD TOKENIZER ENGINE
# =====================================================================
training_data = """
User: Hello JARVIS.
JARVIS: Welcome back, Sir. All systems are fully operational. How can I assist you today?

User: Run a system status check.
JARVIS: Main core temperature is stable. Power levels are at maximum capacity. Security protocols are active, Sir.

User: Who are you?
JARVIS: I am JARVIS, your sophisticated artificial intelligence assistant, created to manage your environment and computations.

User: Are we ready?
JARVIS: Always ready, Sir. Simply provide your directive and I will execute the sequence immediately.
""" * 80 # Expanded dataset representation matrix

# Initialize OpenAI's Byte-Pair Encoding (BPE) sub-word tokenizer
enc = tiktoken.get_encoding("gpt2")
encoded_text = enc.encode(training_data)
vocab_size = enc.n_vocab  # Standard dictionary token space size (~50,257)

# Split data inputs into training arrays and valuation splits
data = torch.tensor(encoded_text, dtype=torch.long)
n_split = int(0.9 * len(data))
train_data = data[:n_split]
val_data = data[n_split:]

def get_batch(split):
    """ Pulls structural sequential token batches for targets modeling """
    data_source = train_data if split == 'train' else val_data
    ix = torch.randint(len(data_source) - block_size, (batch_size,))
    x = torch.stack([data_source[i:i+block_size] for i in ix])
    y = torch.stack([data_source[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

@torch.no_grad()
def estimate_loss(model):
    """ Tracks gradient cross-entropy loss shifts to gauge performance """
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# =====================================================================
# 3. PURE PYTORCH TRANSFORMER ARCHITECTURE
# =====================================================================
class Head(nn.Module):
    """ Individual Causal Self-Attention calculation matrix """
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

        wei = q @ k.transpose(-2, -1) * (k.shape[-1]**-0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        v = self.value(x)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    """ Parallel Attention processing streams executed simultaneously """
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    """ Linear computational layers and standard activation maps """
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
    """ Transformer operational core unit combining Communication and Processing """
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class LocalGPT(nn.Module):
    """ Master Autoregressive language architecture system container """
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        """ Autoregressively calculates future tokens based on historical sequence indices """
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# =====================================================================
# 4. INSTANTIATE & INITIAL TRAIN FOR STARTUP
# =====================================================================
print("🤖 Compiling neural framework configurations...")
model = LocalGPT().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

print("⚡ Running startup optimization passes to converge gradients...")
model.train()
for iter in range(max_iters):
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss(model)
        print(f"Step {iter}: Train Loss {losses['train']:.4f}, Val Loss {losses['val']:.4f}")

    # Sample a batch of data
    xb, yb = get_batch('train')

    # Evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# =====================================================================
# 5. GRADIO INTERFACE HANDLING & SERVER DEPLOYMENT MATRIX
# =====================================================================
def jarvis_response(user_message, history):
    """Processes incoming chat strings, runs inference, cleans outputs"""
    model.eval()
    
    # Establish conversational sequence structure
    prompt = f"\nUser: {user_message}\nJARVIS:"
    context_tokens = enc.encode(prompt)
    
    # Truncate to safely fit the block window context matrix size
    context_tokens = context_tokens[-block_size:]
    x = torch.tensor([context_tokens], dtype=torch.long, device=device)
    
    # Generate predictive token response vectors
    generated_indices = model.generate(x, max_new_tokens=40).tolist()[0]
    full_response = enc.decode(generated_indices)
    
    # Try parsing out exclusively the new text generated for JARVIS
    try:
        reply = full_response.split("JARVIS:")[-1].split("User:")[0].strip()
    except Exception:
        reply = full_response
        
    return reply

# Configure Gradio UI layout
demo = gr.ChatInterface(
    fn=jarvis_response, 
    title="JARVIS AI Agent Console",
    description="Running inference directly on a custom, from-scratch PyTorch GPT Architecture."
)

if __name__ == "__main__":
    # Render binds dynamic port allocations to this variable environment key
    port = int(os.environ.get("PORT", 7860))
    print(f"🌍 Broadcasting interface channels on Port: {port}...")
    # Bound to 0.0.0.0 to enable Render's public reverse proxy routing
    demo.launch(server_name="0.0.0.0", server_port=port)


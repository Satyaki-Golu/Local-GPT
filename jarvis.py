
import torch
import torch.nn as nn
from torch.nn import functional as F
import gradio as gr
import tiktoken
import time

# =====================================================================
# 1. HYPERPARAMETERS & SYSTEM HARDWARE SETUP
# =====================================================================
batch_size = 16      # Parallel text sequence streams processed simultaneously
block_size = 128     # Maximum context window length (token sequence history)
max_iters = 1200     # Gradient optimization steps
eval_interval = 200  # Intervals at which validation performance is checked
learning_rate = 5e-4
eval_iters = 100
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
# Training text data used to teach our network conversational paradigms
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
        # Create an explicit look-back tracking matrix mask lower-triangle block
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)

        # Calculate dot-product relationship attention scores scaled by dimension roots
        wei = q @ k.transpose(-2, -1) * (k.shape[-1]**-0.5)
        # Causal mask filtering intercepts token projection from reading information out of order
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
        # Adding residual connections scales gradient flow safety bounds
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
# 4. MODEL INITIALIZATION AND LOCAL DEVICE TRAINING
# =====================================================================
model = LocalGPT().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

print("⚡ Starting localized tensor training passes...")
for iter in range(max_iters):
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss(model)
        print(f"Iteration: {iter:4d} | Training Loss: {losses['train']:.4f} | Validation Loss: {losses['val']:.4f}")

    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print("\n🎉 Core model pipeline trained successfully! Deploying server canvas...")

# =====================================================================
# 5. GRADIO STREAMING LOCAL CHAT USER WEB INTERFACE
# =====================================================================
def generate_web_response(message, history):
    # Formulate conversational prompt patterns matching structural templates
    context_str = f"User: {message}\nJARVIS:"

    # Tokenize input context string using tiktoken encoder
    encoded_input = enc.encode(context_str)
    input_tensor = torch.tensor([encoded_input], dtype=torch.long, device=device)

    # Process text tensor generation passes
    model.eval()
    generated_tensor = model.generate(input_tensor, max_new_tokens=45)
    decoded_output = enc.decode(generated_tensor[0].tolist())

    # Strip contextual parsing layout anomalies from user views
    answer = decoded_output.replace(context_str, "")
    if "User:" in answer:
        answer = answer.split("User:")[0]
    answer = answer.strip()

    if not answer:
        answer = "Awaiting structural telemetry configurations, Sir."

    # Sequential character generation loops to mimic real-time response streams
    partial_response = ""
    for char in answer:
        partial_response += char
        time.sleep(0.01)
        yield partial_response

# Compile layout elements
with gr.Blocks(theme=gr.themes.Default(primary_hue="blue", secondary_hue="slate")) as demo:
    gr.Markdown("# 🖥️ From-Scratch Local PyTorch GPT Interface")
    gr.Markdown("An end-to-end custom Transformer running predictions on your device's native hardware processor arrays.")
    gr.ChatInterface(
        fn=generate_web_response,
        textbox=gr.Textbox(placeholder="Talk to your standalone local model...", container=False, scale=7),
        retry_btn=None,
        undo_btn=None,
        clear_btn="Reset History"
    )

if __name__ == "__main__":
    demo.launch()

"""## Model Resources and Configuration"""
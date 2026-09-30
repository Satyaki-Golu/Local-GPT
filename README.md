# LocalGPT: From-Scratch Custom Transformer with PyTorch

An educational implementation of a **Causal, Autoregressive GPT-style Transformer** built entirely from scratch using pure PyTorch. This model utilizes OpenAI's `tiktoken` byte-pair encoding (BPE) engine and includes an implementation of Multi-Head Self-Attention, residual connections, layer normalization, and a dynamic hardware routing matrix.

---

## 📌 Features
* **Pure PyTorch Implementation**: Build blocks, attention mechanisms, and embeddings without high-level abstraction wrappers.
* **Smart Device Allocation**: Automatic fallback and detection across **NVIDIA CUDA**, **Apple Silicon (MPS)**, and standard **CPU** engines.
* **BPE Tokenization**: Leverages `tiktoken` (`gpt2` configuration) for token mapping rather than character-level parsing.
* **Causal Masking**: Explicit lower-triangle look-back masking prevents future token leakage during training.

---

## ⚙️ Hyperparameters & System Config

The architectural dimensions are optimized to allow rapid training cycles on consumer-grade hardware:

| Hyperparameter | Value | Description |
| :--- | :--- | :--- |
| `batch_size` | 16 | Number of parallel sequence streams processed simultaneously |
| `block_size` | 128 | Maximum context window length (Token sequence capacity) |
| `max_iters` | 1200 | Maximum gradient optimization steps |
| `learning_rate`| 5e-4 | Step size modifier for AdamW optimizer |
| `n_embd` | 192 | Embedding vector dimensionality size |
| `n_head` | 6 | Concurrent Multi-Head Attention heads (`head_size` = 32) |
| `n_layer` | 4 | Number of sequential Transformer blocks stacked |
| `dropout` | 0.1 | Random dropout rate applied to control overfitting |

---

## 🛠️ Architecture Breakdown

### 1. Data Processing Engine
The framework feeds on a conversational matrix mapping out responses for a toy virtual assistant named **JARVIS**. It splits the tokenized tensor matrix into a `90%` training split and a `10%` verification split.
* **Context Shifting**: Target windows are dynamically shifted forward by exactly t+1 to allow text prediction sequence modeling.

### 2. Attention Matrix (`Head` & `MultiHeadAttention`)
Computes dot-product scaled self-attention scoring vectors. 
\[\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V\]
A lower-triangular matrix (`tril`) acts as a causal mask, zeroing out future probabilities by evaluating them as -∞ before applying the Softmax layer.

### 3. Structural Layer Blocks (`Block`)
Each block acts as a standard Transformer layer leveraging:
* **Pre-LN Architecture**: Layer Normalization is applied *before* the Multi-Head Attention and Feed-Forward networks to stabilize gradient flows.
* **Residual Connections**: Skip connections add inputs directly to the block outputs, expanding computational safety bounds.

---

## 🚀 Setup & Execution

### Prerequisites
Ensure you have Python 3.8+ installed along with the required libraries:
```bash
pip install torch gradio tiktoken
```

### Running the Script
Save the provided code to a file (e.g., `train.py`) and run it via your terminal:
```bash
python train.py
```

---

## 📝 Code Implementation Reference

```python
# Save your script core logic inside your main training execution pipeline.
# The network will automatically detect your underlying silicon architecture:
# CUDA -> MPS -> CPU
```


# TOPIC 1 — Neural Networks from Scratch → PyTorch (Tutorial Sheet)

```
CURRENT PROGRESS
----------------
Completed: none
Current:   Neural networks, backprop, PyTorch training loop
Next:      Deep learning basics (CNN, normalization, init) → Tokenization & language modeling → Embeddings → Attention → Transformer / tiny GPT
Depends on: calculus (chain rule), linear algebra (matmul, shapes), basic probability, Python/NumPy
```

**Outcome:** you can train an MLP with hand-written backprop, verify it with gradient checking, reproduce it with autograd, write a clean PyTorch training loop, and debug a model that won't learn.

**Time budget:** ~8–12 focused hours across 5 sessions.

---

## 0. Setup (15 min)

```bash
python -m venv .venv && source .venv/bin/activate
pip install numpy matplotlib torch torchvision
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Folder: `nn-foundations/` with `np_mlp.py`, `value.py`, `torch_loop.py`, `mnist.py`.

---

## Roadmap

| Step | Do | Done when |
|---|---|---|
| 1 | Tensors & shapes | You predict output shapes of 10 ops without running them |
| 2 | One neuron by hand | Hand gradient == numeric gradient |
| 3 | MLP in NumPy (forward + manual backprop) | Spiral accuracy > 95%, gradient check rel_err < 1e-6 |
| 4 | Mini autograd engine (`Value`) | Matches PyTorch on a small expression |
| 5 | Same MLP in PyTorch autograd | Gradients match NumPy version to ~1e-12 (float64) |
| 6 | `nn.Module` + training loop | Train/val curves plotted |
| 7 | MNIST + overfitting experiments | > 97% val acc; you can demonstrate overfit and fix it |
| 8 | Debug checklist drills | You can overfit a single batch on demand |
| 9 | Final project | Deployed endpoint (see §Project) |

---

## Step 1 — Tensors & shapes

**Intuition:** a tensor is an n-dimensional array; ML is mostly batched matrix multiplication where shape discipline is everything.

**Engineering:** convention is `(batch, features)` for MLPs, `(batch, seq, dim)` for sequence models.

**Internals:** PyTorch tensors are a strided view over contiguous storage. `view`/`transpose` change strides, not data; `reshape` may copy; `.contiguous()` forces a copy.

Practice (predict first, then run):

```python
import torch
a = torch.randn(32, 10); W = torch.randn(10, 4); b = torch.randn(4)
(a @ W + b).shape
a.sum(0).shape, a.sum(1, keepdim=True).shape
torch.randn(8, 1, 5) + torch.randn(3, 1)
a.T.is_contiguous()
a.argmax(1).shape
```

Learn: broadcasting rule (align from the right; dims must match or be 1), `keepdim`, `dtype` (float32 default, int64 for class labels), `.to(device)`, `.detach()`, `.item()`.

---

## Step 2 — One neuron, by hand

Neuron: `z = w·x + b`, `a = σ(z)`. Loss `L = (a − y)²`.

Chain rule: `dL/dw = dL/da · da/dz · dz/dw`.

Task: pick `x=2, w=-1, b=0.5, y=1` with σ = tanh. Compute `dL/dw` and `dL/db` on paper, then confirm numerically:

```python
import math
def f(w, b, x=2.0, y=1.0): return (math.tanh(w * x + b) - y) ** 2
eps = 1e-6
num_dw = (f(-1 + eps, 0.5) - f(-1 - eps, 0.5)) / (2 * eps)
```

Central difference `(f(w+ε) − f(w−ε)) / 2ε` is the ground truth you will use to test every gradient you ever derive.

---

## Step 3 — MLP in NumPy with manual backprop

### Intuition
Stack of linear maps with nonlinearities. Without the nonlinearity, any depth collapses into one linear map. Training = repeatedly nudge every weight against the gradient of the loss.

### Math (shapes: N batch, D in, H hidden, C classes)

```
X:(N,D)  W1:(D,H) b1:(H)  W2:(H,C) b2:(C)

z1 = X W1 + b1          (N,H)
a1 = ReLU(z1)           (N,H)
z2 = a1 W2 + b2         (N,C)   logits
p  = softmax(z2)        (N,C)
L  = -(1/N) Σ log p[i, y_i]
```

Backward (the whole trick):

```
dz2 = (p − onehot(y)) / N        (N,C)   softmax+CE gradient
dW2 = a1ᵀ dz2                    (H,C)
db2 = Σ_rows dz2                 (C)
da1 = dz2 W2ᵀ                    (N,H)
dz1 = da1 ⊙ 1[z1 > 0]            (N,H)
dW1 = Xᵀ dz1                     (D,H)
db1 = Σ_rows dz1                 (H)
```

Rule of thumb: the gradient of a parameter always has the same shape as the parameter. Use that to check your matmul orientation.

Numerical stability: subtract row max before `exp` in softmax. He init `N(0, 2/fan_in)` for ReLU keeps activation variance stable across layers.

### Code (tested)

```python
import numpy as np

def make_spiral(n=100, k=3, seed=0):
    rng = np.random.default_rng(seed)
    X = np.zeros((n * k, 2)); y = np.zeros(n * k, dtype=int)
    for j in range(k):
        ix = slice(n * j, n * (j + 1))
        r = np.linspace(0.0, 1, n)
        t = np.linspace(j * 4, (j + 1) * 4, n) + rng.normal(0, 0.2, n)
        X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
        y[ix] = j
    return X, y

class MLP:
    def __init__(s, i, h, o, seed=0):
        r = np.random.default_rng(seed)
        s.W1 = r.normal(0, np.sqrt(2 / i), (i, h)); s.b1 = np.zeros(h)
        s.W2 = r.normal(0, np.sqrt(2 / h), (h, o)); s.b2 = np.zeros(o)

    def forward(s, X):
        s.X = X
        s.z1 = X @ s.W1 + s.b1
        s.a1 = np.maximum(0, s.z1)
        s.z2 = s.a1 @ s.W2 + s.b2
        return s.z2

    def loss(s, logits, y):
        z = logits - logits.max(1, keepdims=True)
        p = np.exp(z); p /= p.sum(1, keepdims=True)
        s.p = p
        return -np.log(p[np.arange(len(y)), y] + 1e-12).mean()

    def backward(s, y):
        N = len(y)
        dz2 = s.p.copy(); dz2[np.arange(N), y] -= 1; dz2 /= N
        s.dW2 = s.a1.T @ dz2; s.db2 = dz2.sum(0)
        dz1 = (dz2 @ s.W2.T) * (s.z1 > 0)
        s.dW1 = s.X.T @ dz1; s.db1 = dz1.sum(0)

    def step(s, lr):
        s.W1 -= lr * s.dW1; s.b1 -= lr * s.db1
        s.W2 -= lr * s.dW2; s.b2 -= lr * s.db2

def grad_check(model, X, y, eps=1e-5):
    model.loss(model.forward(X), y); model.backward(y)
    for name in ["W1", "b1", "W2", "b2"]:
        P = getattr(model, name); G = getattr(model, "d" + name)
        idx = tuple(np.random.randint(0, d) for d in P.shape)
        old = P[idx]
        P[idx] = old + eps; lp = model.loss(model.forward(X), y)
        P[idx] = old - eps; lm = model.loss(model.forward(X), y)
        P[idx] = old
        num = (lp - lm) / (2 * eps)
        rel = abs(num - G[idx]) / max(1e-8, abs(num) + abs(G[idx]))
        print(f"{name}: analytic={G[idx]:.6f} numeric={num:.6f} rel_err={rel:.2e}")

if __name__ == "__main__":
    X, y = make_spiral()
    m = MLP(2, 64, 3)
    grad_check(m, X, y)
    for ep in range(2000):
        loss = m.loss(m.forward(X), y); m.backward(y); m.step(1.0)
        if ep % 400 == 0: print(ep, round(loss, 4))
    print("acc", (m.forward(X).argmax(1) == y).mean())
```

Observed when run: rel_err ≈ 1e-8 to 1e-11 on all four params, loss 1.08 → 0.04, accuracy ≈ 99% (training set).

### Tasks
1. Re-derive each backward line yourself before reading mine.
2. Change ReLU to tanh; update backward (`1 − a1²`). Re-run gradient check.
3. Add L2 regularization to loss and gradients.
4. Add a third layer. If gradient check passes, you understand backprop.
5. Plot decision boundary on a meshgrid.

---

## Step 4 — Mini autograd engine

**Intuition:** backprop = chain rule applied over a computation graph in reverse topological order. Each op only needs to know its *local* derivative.

**Internals:** every `Value` stores `data`, `grad`, parents, and a closure that pushes `out.grad` into parents' `grad`. Note `+=` — a node used twice (e.g. `a*a`) must accumulate.

```python
import math

class Value:
    def __init__(s, data, _prev=()):
        s.data = data; s.grad = 0.0
        s._prev = _prev; s._backward = lambda: None

    def __add__(s, o):
        o = o if isinstance(o, Value) else Value(o)
        out = Value(s.data + o.data, (s, o))
        def _b():
            s.grad += out.grad; o.grad += out.grad
        out._backward = _b
        return out

    def __mul__(s, o):
        o = o if isinstance(o, Value) else Value(o)
        out = Value(s.data * o.data, (s, o))
        def _b():
            s.grad += o.data * out.grad; o.grad += s.data * out.grad
        out._backward = _b
        return out

    def tanh(s):
        t = math.tanh(s.data)
        out = Value(t, (s,))
        def _b():
            s.grad += (1 - t * t) * out.grad
        out._backward = _b
        return out

    def backward(s):
        topo, seen = [], set()
        def build(v):
            if v not in seen:
                seen.add(v)
                for p in v._prev: build(p)
                topo.append(v)
        build(s)
        s.grad = 1.0
        for v in reversed(topo): v._backward()

    __radd__ = __add__
    __rmul__ = __mul__

x, w, b = Value(0.5), Value(-1.5), Value(0.8)
y = (x * w + b).tanh(); y.backward()
print(y.data, x.grad, w.grad, b.grad)
```

Verified output: `0.04996, -1.4963, 0.4988, 0.9975`. Also `a=3; c=a*a+a; c.backward()` → `a.grad == 7`.

Tasks: add `__sub__`, `__neg__`, `__pow__`, `exp`, `relu`, `log`. Build a neuron and a 2-layer MLP out of `Value`s; train on XOR. Compare every gradient with PyTorch.

Why this matters: PyTorch autograd is the same idea over tensors (`grad_fn` graph, `.backward()` walks it in reverse). Tensor ops store their backward rules; memory for saved activations is why training needs far more memory than inference.

---

## Step 5 — Same MLP, PyTorch autograd

```python
import numpy as np, torch
from np_mlp import make_spiral, MLP

X, y = make_spiral()
m = MLP(2, 64, 3)
Xt = torch.tensor(X, dtype=torch.float64); yt = torch.tensor(y)
p = {n: torch.tensor(getattr(m, n), dtype=torch.float64, requires_grad=True) for n in ["W1", "b1", "W2", "b2"]}

logits = torch.relu(Xt @ p["W1"] + p["b1"]) @ p["W2"] + p["b2"]
loss = torch.nn.functional.cross_entropy(logits, yt)
loss.backward()

m.loss(m.forward(X), y); m.backward(y)
for n in p:
    print(n, np.abs(p[n].grad.numpy() - getattr(m, "d" + n)).max())
```

Expected: max diff ~1e-17 to 1e-15. Note: `cross_entropy` takes **raw logits** (it fuses log-softmax + NLL) and **class indices** (int64), not one-hot, not softmax output.

*(PyTorch snippets in this sheet were not executed in my environment; NumPy and `Value` code was. If something errors, paste the traceback and I'll fix it.)*

---

## Step 6 — `nn.Module` + a proper training loop

```python
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, random_split
from np_mlp import make_spiral

torch.manual_seed(0)
X, y = make_spiral(n=300)
ds = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y))
tr, va = random_split(ds, [0.8, 0.2], generator=torch.Generator().manual_seed(0))
tl = DataLoader(tr, batch_size=64, shuffle=True)
vl = DataLoader(va, batch_size=256)

dev = "cuda" if torch.cuda.is_available() else "cpu"
model = nn.Sequential(nn.Linear(2, 64), nn.ReLU(), nn.Linear(64, 3)).to(dev)
opt = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=1e-4)

def evaluate():
    model.eval(); L = c = n = 0
    with torch.no_grad():
        for xb, yb in vl:
            xb, yb = xb.to(dev), yb.to(dev)
            out = model(xb)
            L += F.cross_entropy(out, yb, reduction="sum").item()
            c += (out.argmax(1) == yb).sum().item(); n += len(yb)
    return L / n, c / n

for ep in range(60):
    model.train()
    for xb, yb in tl:
        xb, yb = xb.to(dev), yb.to(dev)
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(xb), yb)
        loss.backward()
        opt.step()
    if ep % 10 == 0: print(ep, *evaluate())
```

The five-line core (memorize): `zero_grad → forward → loss → backward → step`.

Learn each:
- `model.train()` vs `model.eval()` (dropout, batchnorm behave differently)
- `torch.no_grad()` (no graph, less memory) vs `.detach()`
- `parameters()`, `state_dict()`, `torch.save(model.state_dict(), ...)`, `load_state_dict`
- Why gradients accumulate (`.grad += `) and hence need zeroing
- Optimizers: SGD → momentum → Adam (per-parameter adaptive step from first/second moment estimates) → AdamW (decoupled weight decay; the default for transformers)
- LR is the most important hyperparameter; try `1e-4, 1e-3, 1e-2, 1e-1` and observe

Write a custom `nn.Module` version (class with `__init__` and `forward`) and confirm identical behavior.

---

## Step 7 — MNIST and overfitting experiments

```python
import torch, torch.nn as nn, torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
full = datasets.MNIST("data", train=True, download=True, transform=tf)
test = datasets.MNIST("data", train=False, transform=tf)
tr, va = random_split(full, [55000, 5000], generator=torch.Generator().manual_seed(0))
tl = DataLoader(tr, batch_size=128, shuffle=True, num_workers=2)
vl = DataLoader(va, batch_size=512)

class Net(nn.Module):
    def __init__(s, h=256, p=0.0):
        super().__init__()
        s.f = nn.Sequential(nn.Flatten(), nn.Linear(784, h), nn.ReLU(), nn.Dropout(p), nn.Linear(h, 10))
    def forward(s, x): return s.f(x)
```

Experiments (record val loss/acc for each in a table):
1. Baseline, 10 epochs.
2. Train on only 1,000 samples for 100 epochs → watch train acc → 100%, val stalls (overfitting).
3. Add dropout 0.3; add weight decay; add early stopping. Compare.
4. Learning rate sweep; batch size sweep (32 / 128 / 1024).
5. Remove input normalization; observe.
6. Initialize all weights to 0; explain why it fails (symmetry).
7. Report final number on the **test set once**, only at the end.

Concept checkpoints: bias–variance, train/val/test split purpose, why val set tunes hyperparameters and test set is touched once, generalization gap.

---

## Step 8 — Debug checklist (the most valuable skill here)

| Symptom | Likely cause |
|---|---|
| Loss is `nan` | LR too high, `log(0)`, missing softmax stabilization |
| Loss doesn't move | LR too small, forgot `opt.step()`, params not in optimizer, frozen layers |
| Loss stuck at ≈ ln(C) | Predicting uniform: dead ReLUs, bad init, labels shuffled vs inputs |
| Train great / val bad | Overfit: more data, dropout, weight decay, smaller model |
| Both bad | Underfit: bigger model, train longer, tune LR |
| Val acc suspiciously high | Data leakage |
| Shape/dtype errors | `long` labels, float32 inputs, batch dim missing |
| Different results each run | Seeds, nondeterministic ops |

Drills:
1. Initial loss for C classes should be ≈ `ln(C)`; ln(10) ≈ 2.303 for MNIST. Verify before training.
2. **Overfit a single batch** of 32 examples to ~0 loss. If you can't, there is a bug, not a tuning problem.
3. Inject each bug above deliberately and diagnose it.
4. Print gradient norms per layer.

---

## Under the hood (read once, revisit later)

- **Why ReLU:** sigmoid/tanh saturate → gradient ≈ 0 → vanishing gradients; ReLU has gradient 1 for positive inputs. Cost: dead neurons. Modern LLMs use GELU/SwiGLU variants.
- **Why softmax+CE together:** gradient simplifies to `p − y`; separately it is numerically unstable. That is why PyTorch wants logits.
- **Complexity:** forward for a dense layer is `O(N·D·H)` FLOPs; backward ≈ 2× forward; training ≈ 3× forward. Parameter memory `D·H`, activation memory `N·H` per layer (stored for backward).
- **Training memory** = params + gradients + optimizer state (Adam: 2 extra copies) + activations. This breakdown returns in LoRA/QLoRA and quantization chapters.
- **Minibatch SGD:** noisy gradient estimate of the full-data gradient; noise helps generalization and fits in memory.

---

## Common mistakes

- Applying `softmax` before `cross_entropy`
- Forgetting `zero_grad`
- Forgetting `model.eval()` / `torch.no_grad()` at validation
- One-hot labels into `cross_entropy`
- Using test data for tuning
- Not normalizing inputs
- Comparing runs without fixing seeds
- Reading training accuracy as model quality
- Tuning architecture before verifying single-batch overfit

---

## Interview questions

1. **Why do we need nonlinearities?** Without them, stacked linear layers equal one linear layer.
2. **Derive the gradient of softmax + cross-entropy w.r.t. logits.** `p − y`.
3. **What does `loss.backward()` do?** Walks the autograd graph in reverse topological order, applying each op's local derivative and accumulating into `.grad` of leaf tensors.
4. **Why call `zero_grad()`?** `.grad` accumulates; accumulation is useful for gradient accumulation but must be reset per step otherwise.
5. **SGD vs Adam?** Adam keeps running mean and variance of gradients per parameter, giving adaptive per-parameter step sizes; AdamW decouples weight decay from that adaptive update.
6. **Vanishing/exploding gradients — cause and fixes?** Repeated multiplication of Jacobians; use ReLU-family, good init (He/Xavier), normalization, residual connections, gradient clipping.
7. **Why can't we initialize weights to zero?** All neurons in a layer receive identical gradients and stay identical.
8. **What is the expected loss at initialization for C classes?** ≈ `ln(C)`.
9. **`model.eval()` vs `torch.no_grad()`?** `eval` changes layer behavior (dropout/BN); `no_grad` disables graph construction. You typically use both.
10. **Why do training and inference differ in memory?** Training stores activations plus gradients and optimizer state.

---

## Project

**"nn-from-scratch" repo + deployed digit classifier**

1. `micrograd`-style tensor-free engine (`Value`) with MLP, trained on `make_moons`/XOR.
2. NumPy MLP with gradient checking, trained on MNIST (target > 95%).
3. PyTorch version with config (hidden size, LR, dropout), logging curves to CSV, and early stopping with checkpointing of the best `state_dict`.
4. A short README comparing: from-scratch vs autograd vs `nn.Module`, with plots and the overfit experiments.
5. Serve it: FastAPI `POST /predict` taking a 28×28 array/PNG, returning probabilities; Dockerize it.

```python
from fastapi import FastAPI
from pydantic import BaseModel
import torch

app = FastAPI()
model = torch.load("model.pt", weights_only=False).eval()

class Req(BaseModel):
    pixels: list[float]

@app.post("/predict")
def predict(r: Req):
    x = torch.tensor(r.pixels, dtype=torch.float32).view(1, 1, 28, 28)
    with torch.no_grad():
        p = torch.softmax(model(x), -1)[0]
    return {"label": int(p.argmax()), "probs": p.tolist()}
```

(Prefer saving `state_dict` and rebuilding the class on load in real code; shown here for brevity. Apply the same Normalize transform you trained with.)

---

## Connection to what comes next

| This topic | Becomes |
|---|---|
| Matmul + shapes | Attention (`QKᵀ`) |
| Softmax + CE | Next-token prediction loss |
| Training loop | Fine-tuning, SFT, RL loops |
| AdamW, LR | Transformer training recipes |
| Memory breakdown | KV cache, quantization, LoRA |
| Overfit-one-batch | Debugging any model |

---

## What to memorize

- Training loop: `zero_grad → forward → loss → backward → step`
- Softmax+CE gradient: `p − y`; initial loss ≈ `ln(C)`
- Gradient shape == parameter shape
- `dW = inputᵀ · dOut`, `dInput = dOut · Wᵀ`, `db = Σ rows dOut`
- He init `std = sqrt(2/fan_in)`
- `cross_entropy(logits, int64 class indices)`
- Train/val/test roles; seeds; `train()` vs `eval()`; `no_grad`
- Training memory = params + grads + optimizer state + activations

---

## One-page revision

```
Data → [Linear → ReLU]×k → Linear → logits → softmax → CE loss
                                                   ↓
     update ← optimizer(AdamW) ← grads ← backward (chain rule, reverse order)

Shapes: X(N,D) W(D,H) → (N,H)    grads match param shapes
Loss at init ≈ ln(C)   Overfit 1 batch first   Normalize inputs
Debug: nan→LR  |  flat→optimizer/params  |  train≫val→regularize
Autograd = graph of ops + local derivatives, accumulate with +=
```

**If I understand everything here, what can I build?** Train and debug image/tabular classifiers from scratch, write your own autograd, serve a model via FastAPI, and explain exactly what the training loop in every later chapter (language models, fine-tuning) is doing.

---

## Recommended resources

- Andrej Karpathy — *Neural Networks: Zero to Hero*, lectures 1 (micrograd) and 2–3 (makemore MLP, activations/gradients/BatchNorm). Free on YouTube.
- Stanford CS231n notes — Module 1: backprop, neural networks parts 1–3 (cs231n.github.io).
- PyTorch official tutorials — "Learn the Basics" (Tensors, Datasets & DataLoaders, Autograd, Optimization, Save/Load).
- Goodfellow et al., *Deep Learning*, chapters 6 and 8 (free at deeplearningbook.org) — skim for backprop and optimization.
- Karpathy's blog post "A Recipe for Training Neural Networks" — the debugging mindset behind Step 8.

---

```
SKILL TREE
----------
Covered (not yet mastered): none until you complete the tasks above.
Unlocks when done: tensors/broadcasting · MLP forward/backward · softmax+CE · gradient checking ·
autograd internals · PyTorch training loop · optimizers (SGD/Adam/AdamW) · overfitting & regularization ·
single-batch debugging · model serving basics
```

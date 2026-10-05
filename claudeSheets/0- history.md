# TOPIC 0 — History & Foundations: What is AI, DL, Backprop, NLP, and How We Got to Transformers

```
CURRENT PROGRESS
----------------
Completed: none
Current:   Topic 0 — History/foundations (AI → ML → DL → NLP → Transformers)
Next:      Topic 1 — Neural networks from scratch → PyTorch  (tutorial_sheet.md)
           → Deep learning basics → Tokenization & language modeling → Embeddings → Attention → Transformer/tiny GPT
Depends on: nothing. Read this first or alongside Topic 1.
```

**Outcome:** you can explain, without notes, why each generation of AI/NLP replaced the previous one, what problem each step solved, and why the Transformer became the architecture that everything modern (LLMs, agents, RAG) is built on.

**Time budget:** ~4–6 hours across 2–3 sessions.

---

## 1. TL;DR

- **AI** = building systems that perform tasks needing intelligence. **ML** = learn the function from data instead of hand-coding rules. **DL** = ML with many-layer neural networks that learn their own features. **LLM** = a very large Transformer trained to predict the next token.
- Progress came from three levers moving together: **algorithms** (backprop, attention), **data** (web-scale text), **compute** (GPUs/TPUs).
- Backprop made multi-layer networks trainable. Embeddings made words numeric. LSTMs handled sequences. Attention removed the fixed-size bottleneck. Transformers removed recurrence and made training parallel, which unlocked scale.
- Scale + next-token prediction produced models that generalize across tasks; post-training (instruction tuning, RLHF/RL) made them usable; tools, retrieval and agents made them useful in systems.

---

## 2. Why this history matters for an engineer

Every component in a modern AI system exists to fix a specific failure of an earlier one. If you know the failure, you know when the component is needed:

| Failure | Fix | Where it shows up today |
|---|---|---|
| Hand-written rules don't scale | Learn from data | Everything |
| Single-layer models can't represent XOR | Hidden layers + backprop | Every network |
| Manual feature engineering | Learned representations | Embeddings, hidden states |
| Words as IDs have no similarity | Dense embeddings | Semantic search, RAG |
| Counts-based LMs are sparse | Neural LMs | All LLMs |
| RNNs forget & can't parallelize | Attention → Transformer | GPT/Claude/Llama |
| Pretrained models are raw text predictors | SFT, preference tuning, RL | Chat/assistant models |
| Models hallucinate, know nothing recent | Retrieval, tools | RAG, agents, MCP |

---

## 3. Intuition — the taxonomy

```
Artificial Intelligence
 ├─ Symbolic / rule-based (expert systems, search, logic)
 └─ Machine Learning
     ├─ Classical ML (linear/logistic regression, trees, SVM, k-means)
     └─ Deep Learning (multi-layer neural networks)
         ├─ CNNs (images)
         ├─ RNN/LSTM (sequences)
         └─ Transformers
             └─ LLMs (decoder-only, next-token prediction)
                 └─ Assistants, RAG, agents (systems built around LLMs)
```

Three learning paradigms: **supervised** (input → label), **unsupervised/self-supervised** (structure in raw data; LLMs predict the next token from raw text, no human labels), **reinforcement learning** (learn from reward).

**Training vs inference:** training adjusts parameters using gradients (expensive, offline); inference runs the frozen model (what your API call does).

**Parameters vs tokens:** parameters are the learned weights (model size); tokens are pieces of text the model reads/writes (data size and context size). Mixing these up is a common interview slip.

---

## 4. Timeline (the part to learn cold)

### Era 1 — Symbolic AI and the first neurons (1943–1980s)

| Year | Event | Why it matters |
|---|---|---|
| 1943 | McCulloch–Pitts artificial neuron | First mathematical neuron model |
| 1950 | Turing, "Computing Machinery and Intelligence" | The imitation game / Turing test |
| 1956 | Dartmouth workshop; "artificial intelligence" coined (McCarthy) | Birth of the field |
| 1958 | Rosenblatt's perceptron | First learning rule for a neuron |
| 1969 | Minsky & Papert, *Perceptrons* | Showed single-layer limits (XOR); contributed to funding collapse |
| 1970s | First "AI winter" | Overpromising vs compute/data reality |
| 1980s | Expert systems boom, then bust (second winter, late 80s–90s) | Rules don't scale; knowledge is hard to hand-code |

### Era 2 — Backprop and statistical learning (1970s–2000s)

| Year | Event | Why it matters |
|---|---|---|
| 1970 / 1974 | Linnainmaa (reverse-mode autodiff), Werbos (applied to NNs) | Mathematical roots of backprop |
| 1986 | Rumelhart, Hinton & Williams popularize backprop | Multi-layer nets become trainable |
| 1989 / 1998 | LeCun: CNN for digits → LeNet-5 | Convolutions, weight sharing |
| 1990 | Elman RNN | Neural nets for sequences |
| 1997 | LSTM (Hochreiter & Schmidhuber) | Gates fix vanishing gradients over time |
| 1997 | Deep Blue beats Kasparov | Search + hand-crafted evaluation, not learning |
| 1990s–2000s | SVMs, random forests dominate | Strong with small data and engineered features |
| 2003 | Bengio neural probabilistic LM | Learned word embeddings + neural next-word prediction |

### Era 3 — Deep learning revolution (2012–2016)

| Year | Event |
|---|---|
| 2012 | AlexNet wins ImageNet by a large margin; GPUs + big data + ReLU/dropout |
| 2013 | word2vec — cheap, high-quality word embeddings |
| 2014 | GloVe; seq2seq (Sutskever et al.); **Bahdanau attention**; Adam; GANs |
| 2015 | ResNet (residual connections), BatchNorm |
| 2016 | AlphaGo (deep RL + search) |

### Era 4 — Transformers and pretraining (2017–2019)

| Year | Event |
|---|---|
| 2017 | **"Attention Is All You Need"** — the Transformer |
| 2018 | ELMo, **GPT-1**, **BERT** — pretrain on unlabeled text, then fine-tune |
| 2019 | GPT-2, T5 |

### Era 5 — Scaling and LLMs (2020–2022)

| Year | Event |
|---|---|
| 2020 | **GPT-3** (few-shot in-context learning); Kaplan et al. scaling laws |
| 2021 | CLIP; LoRA; Codex |
| 2022 | Chain-of-thought prompting; Chinchilla (compute-optimal: scale data with params); InstructGPT (RLHF); ChatGPT (Nov) |

### Era 6 — Systems around models (2023 → now)

Open-weight models (LLaMA family, Mistral, etc.), long context, multimodality, RAG as standard practice, function/tool calling, agents, Model Context Protocol (late 2024), reasoning models trained with RL on verifiable rewards (o1 in 2024, DeepSeek-R1 in early 2025). This is the territory of the rest of your curriculum. Because this part moves fast, check current model names and capabilities in official docs rather than trusting any static list.

---

## 5. Core concepts

| Term | Meaning |
|---|---|
| Feature | An input signal the model uses. Classical ML: hand-designed. DL: learned. |
| Representation | Internal numeric encoding the model builds (hidden states) |
| Loss | Scalar measuring wrongness; training minimizes it |
| Gradient descent | Move parameters opposite the gradient of the loss |
| Backpropagation | Efficient chain-rule computation of all gradients (reverse-mode autodiff) |
| Language model | Model of `P(next token | previous tokens)` |
| Token | Unit of text (word piece), not a word |
| Embedding | Learned dense vector for a token (a row of a lookup table) |
| Hidden state | Context-dependent vector produced by a layer for a position |
| Attention | Weighted lookup: each position reads from other positions |
| Pretraining | Self-supervised training on large raw corpora |
| Fine-tuning | Further training on narrower data |
| Perplexity | `exp(average cross-entropy)`; effective branching factor of a LM |

**Embeddings vs hidden states:** an embedding is a static lookup for a token ID (same for "bank" in any sentence). A hidden state is computed in context (differs for river bank vs money bank). Do not use the terms interchangeably.

---

## 6. How the pieces evolved — the NLP storyline

```
Rules/regex ──► Bag-of-words / TF-IDF + linear model
                   │  (no word order, no meaning of words)
                   ▼
              n-gram language models
                   │  (sparse counts, tiny context)
                   ▼
              Word embeddings (word2vec/GloVe)
                   │  (one vector per word, context-blind)
                   ▼
              RNN / LSTM / GRU language models
                   │  (sequential, forget long range, no parallelism)
                   ▼
              Seq2seq encoder–decoder
                   │  (whole input squeezed into ONE vector)
                   ▼
              + Attention (Bahdanau)
                   │  (decoder looks back at all encoder states)
                   ▼
              Transformer (attention only, no recurrence)
                   │
                   ▼
              Pretrain (BERT: encoder / GPT: decoder) → fine-tune
                   │
                   ▼
              Scale → in-context learning (GPT-3)
                   │
                   ▼
              Instruction tuning + RLHF → assistants
                   │
                   ▼
              RAG, tools, agents, memory, MCP
```

### Why Transformers won (the key exam question)

| Property | RNN/LSTM | Transformer |
|---|---|---|
| Sequential steps per layer | O(T) (can't parallelize across time) | O(1) |
| Path length between two tokens | O(T) | O(1) |
| Per-layer compute | O(T · d²) | O(T² · d) |
| Training on GPUs | poor utilization | excellent |

The Transformer trades quadratic attention cost for parallelism and direct long-range connections. Parallel training is what allowed training on trillions of tokens. The quadratic cost is why long-context, KV cache, and efficient attention become engineering problems later.

### Transformer at 10,000 feet (details come in the Attention/Transformer chapters)

```
tokens → tokenizer → IDs → embedding lookup (+ position info)
      → [ self-attention → MLP ] × L   (with residuals + normalization)
      → final projection to vocabulary → logits → softmax
      → next-token distribution → sample → append → repeat
```

---

## 7. Mathematics (only what is needed here)

**Supervised learning:** find parameters θ minimizing average loss over data:
`θ* = argmin (1/N) Σ L(f_θ(xᵢ), yᵢ)`

**Gradient descent:** `θ ← θ − η ∇θ L` (η is the learning rate).

**Chain rule (backprop's core):** if `L = f(g(h(x)))`, then `dL/dx = f'(g(h(x))) · g'(h(x)) · h'(x)`. Backprop reuses intermediate results so all gradients cost about 2× one forward pass.

Numeric example: `a = 3, c = a·a + a`. `dc/da = 2a + 1 = 7`. A node used twice accumulates gradient from both paths.

**Language model factorization:** 
`P(w₁…w_T) = Π P(w_t | w₁…w_{t−1})`
n-gram approximation: condition only on previous n−1 tokens.

**Softmax:** `p_i = exp(z_i) / Σ_j exp(z_j)` turns logits into a distribution.

**Cross-entropy / perplexity:**
`CE = −(1/T) Σ log P(w_t | context)`, `PPL = exp(CE)`.
A uniform model over vocabulary V has PPL = V. Lower is better. GPT-style training loss is exactly this CE, per token.

**Why single-layer fails XOR:** a perceptron's decision boundary is one line `w·x + b = 0`. XOR's positive points `(0,1),(1,0)` can't be separated from `(0,0),(1,1)` by one line. One hidden layer lets the network combine two lines.

---

## 8. From scratch — three tiny experiments

### 8.1 Perceptron: learns AND/OR, fails XOR (tested)

```python
import numpy as np

def train(X, y, epochs=50, lr=0.1):
    w = np.zeros(X.shape[1]); b = 0.0
    for _ in range(epochs):
        for xi, yi in zip(X, y):
            pred = int(w @ xi + b > 0)
            w += lr * (yi - pred) * xi
            b += lr * (yi - pred)
    return w, b

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
for name, y in [("AND", [0, 0, 0, 1]), ("OR", [0, 1, 1, 1]), ("XOR", [0, 1, 1, 0])]:
    w, b = train(X, np.array(y))
    print(name, ((X @ w + b > 0).astype(int) == y).mean())
```

Output: AND 1.0, OR 1.0, XOR 0.5. Fixing XOR is your first task in Topic 1 (the MLP).

### 8.2 n-gram language model: generation + perplexity (tested)

Put any English text in `corpus.txt` (e.g. a few thousand words).

```python
import random, math
from collections import defaultdict, Counter

tokens = open("corpus.txt").read().lower().split()
cut = int(len(tokens) * 0.9)
train, test = tokens[:cut], tokens[cut:]
V = len(set(train)) + 1

def fit(toks, n):
    c = defaultdict(Counter)
    for i in range(len(toks) - n + 1):
        c[tuple(toks[i:i + n - 1])][toks[i + n - 1]] += 1
    return c

def generate(c, seed, n, k=30):
    out = list(seed)
    for _ in range(k):
        d = c.get(tuple(out[-(n - 1):]))
        if not d: break
        out.append(random.choices(list(d), weights=list(d.values()))[0])
    return " ".join(out)

def perplexity(c, toks, n, alpha=0.1):
    ll = 0.0
    for i in range(n - 1, len(toks)):
        d = c.get(tuple(toks[i - n + 1:i]), {})
        p = (d.get(toks[i], 0) + alpha) / (sum(d.values()) + alpha * V)
        ll += math.log(p)
    return math.exp(-ll / (len(toks) - n + 1))

for n in (2, 3, 4):
    c = fit(train, n)
    print(n, "contexts:", len(c), "ppl:", round(perplexity(c, test, n), 1))
print(generate(fit(train, 3), train[100:102], 3))
```

Run on a ~5.6k-word text, test perplexity was ≈ 546 (bigram), 935 (trigram), 1106 (4-gram): **longer context got worse**. That is data sparsity: most long contexts were never seen, so smoothing dominates. This is the exact problem neural LMs (and embeddings) solve by generalizing across similar contexts. Try it on a larger corpus and watch the crossover point move.

### 8.3 Backprop from scratch

Do Step 4 of `tutorial_sheet.md` (`Value` autograd class). That is the history's central algorithm made concrete.

---

## 9. Real-world: seeing the endpoint with Hugging Face (not executed in my environment)

```python
from transformers import pipeline
gen = pipeline("text-generation", model="gpt2")
print(gen("The history of artificial intelligence began", max_new_tokens=40, do_sample=True)[0]["generated_text"])
```

GPT-2 (2019) is a small decoder-only Transformer with ~124M parameters at the smallest size. Compare its output with your n-gram model: same objective (predict next token), vastly different model class. That comparison is the whole story of this chapter in one experiment.

Optional: load GloVe/word2vec vectors with `gensim.downloader` and verify `king − man + woman ≈ queen`, then look at where analogies fail.

---

## 10. Under the hood

- **Why depth works:** each layer composes simpler features into more abstract ones (edges → shapes → objects; characters → words → syntax → meaning). Hidden layers + nonlinearity are what let a network represent XOR-type functions.
- **Why GPUs mattered:** neural nets are dominated by large matrix multiplications, which are massively parallel. RNNs force sequential steps; Transformers don't.
- **Why attention helps:** in a vanilla seq2seq model the encoder compresses a whole sentence into a single fixed vector; attention lets each decoder step compute a weighted average over all encoder states, so information doesn't pass through one bottleneck.
- **Why pretraining works:** predicting the next token forces a model to learn grammar, facts, and some reasoning patterns, using unlabeled data at massive scale. Labels are the next token itself.
- **Why post-training exists:** a pretrained model continues text; it doesn't follow instructions. SFT shows it desired responses; preference optimization/RLHF/RL shape which responses it prefers.
- **Scaling laws:** loss falls predictably as a power law in parameters, data, and compute. Chinchilla showed many early large models were undertrained relative to their size. In practice modern models often train on far more tokens than "compute-optimal" because inference cost favors smaller, longer-trained models.

---

## 11. Complexity cheat table

| Model | Training parallelism | Context handling | Cost |
|---|---|---|---|
| n-gram | trivial counting | n−1 tokens | memory grows with n (sparse) |
| RNN/LSTM | sequential over time | compressed state | O(T·d²) |
| Transformer | parallel over positions | full window | O(T²·d) attention |

---

## 12. Production perspective

Even at this introductory level, build the habit of asking:

- **Is a model needed?** Regex, SQL, or a classifier may beat an LLM on cost, latency, determinism. The old methods in the timeline remain valid engineering choices.
- **Which generation of tool fits the problem?** Keyword/BM25 search, embedding search, and LLM reranking are different generations with different costs. Hybrid systems are normal.
- **Cost drivers:** tokens in and out, model size, context length.
- **Failure modes inherited from history:** hallucination (the model is a next-token predictor, not a database), stale knowledge (training cutoff), sensitivity to phrasing, quadratic cost on long context.

---

## 13. Common mistakes and misconceptions

- "AI = ChatGPT." LLMs are one narrow slice of AI.
- "Deep learning was invented in 2012." The ideas are older; 2012 was when data + GPUs + tricks made them win.
- "Backprop is the same as gradient descent." Backprop computes gradients; gradient descent/Adam uses them to update parameters.
- "Transformers understand like humans." They model token statistics; capabilities are empirical, not proven understanding.
- "More parameters is always better." Data quantity/quality and training compute matter (Chinchilla).
- Conflating embeddings and hidden states; parameters and tokens; training and inference.
- Treating the transformer as having memory. Between API calls, an LLM has no state; "memory" is context you re-send.
- Assuming the newest technique replaces the old one everywhere.

---

## 14. Interview questions

1. **AI vs ML vs DL?** AI is the broad goal; ML learns functions from data; DL is ML with deep neural networks that learn features.
2. **Why did neural networks stall in the 1970s and revive later?** Perceptron limits plus insufficient compute/data; backprop for multi-layer nets, then GPUs and large datasets.
3. **What does backprop compute?** Gradients of the loss with respect to all parameters via the chain rule in reverse over the computation graph.
4. **Why can't a single perceptron learn XOR?** XOR isn't linearly separable.
5. **What problem does attention solve in seq2seq?** The fixed-size context-vector bottleneck.
6. **Why are Transformers faster to train than LSTMs?** No sequential dependency across positions in a layer, so work parallelizes on GPUs.
7. **What's the cost of that choice?** Self-attention is O(T²) in sequence length.
8. **What is the pretraining objective of GPT vs BERT?** GPT: causal next-token prediction. BERT: masked-token prediction (bidirectional).
9. **Why did bigger n in an n-gram model hurt on small data?** Sparsity: most contexts unseen.
10. **Why isn't a pretrained LLM an assistant?** It only learned to continue text; instruction tuning and preference/RL post-training add instruction-following and alignment.

---

## 15. Hands-on exercises (no tutorial needed)

1. Write your own one-page timeline (from memory) with *problem → fix* for each step. Compare to §4.
2. Run §8.1; then add a hidden layer by hand-picking weights that solve XOR (e.g., compute OR and NAND, then AND them).
3. Run §8.2 on three corpora of different sizes. Plot perplexity vs n. Explain the crossover.
4. Add Laplace vs add-k vs simple backoff smoothing to the n-gram model; compare perplexity.
5. Implement the same sentiment task with (a) keyword rules, (b) TF-IDF + logistic regression (scikit-learn). Note where each breaks.

---

## 16. Project — "NLP Evolution Benchmark"

Pick one dataset (e.g. SST-2 or IMDB sentiment, or a small topic-classification set). Solve it with each generation, same train/test split, same metric:

| Generation | Implementation |
|---|---|
| Rules | Keyword lexicon |
| Classical ML | TF-IDF + logistic regression |
| Embeddings | Averaged pretrained word vectors + logistic regression |
| RNN | Small LSTM in PyTorch (after Topic 1) |
| Transformer | Fine-tune a small pretrained encoder (e.g. DistilBERT) |
| LLM | Zero/few-shot prompt through an API |

Report accuracy, training time, inference latency, cost per 1k predictions, failure examples. Output: a README with a results table and short analysis. This becomes a reusable evaluation harness later (evals chapter) and trains the habit of justifying model choice with numbers.

---

## 17. Connection to the next topics

| This chapter introduces | Developed in |
|---|---|
| Backprop, gradient descent | Topic 1 (NN from scratch/PyTorch) |
| Embeddings | Embeddings chapter → semantic search/RAG |
| Language modeling, perplexity | Tokenization & LM chapter |
| Attention | Attention chapter (Q, K, V, shapes) |
| Transformer block | Transformer/tiny GPT chapter |
| Pretraining → post-training | SFT, LoRA, RLHF chapters |
| LLM limits (stale, hallucinate) | RAG, tools, agents, memory, MCP |

---

## 18. What to memorize

- Taxonomy: AI ⊃ ML ⊃ DL ⊃ (Transformers ⊃ LLMs)
- Dates: 1943 neuron · 1956 AI coined · 1958 perceptron · 1969 *Perceptrons* · 1986 backprop · 1997 LSTM · 2012 AlexNet · 2013 word2vec · 2014 attention · 2017 Transformer · 2018 BERT/GPT-1 · 2020 GPT-3 + scaling laws · 2022 InstructGPT/ChatGPT
- LM objective: `P(w_t | w_<t)`, loss = cross-entropy, PPL = exp(CE)
- Transformer vs RNN: O(1) path & parallel vs O(T); cost O(T²d)
- Pipeline: pretrain → SFT → preference/RL → deploy with retrieval/tools
- Mental models: representation learning; bottleneck → attention; scale + data + compute

---

## 19. One-page revision

```
Rules ─► ML (learn f from data) ─► DL (learn features too)
Perceptron (1958) ─ XOR fails (1969) ─► hidden layers + backprop (1986)
CNN (images) · RNN/LSTM (sequences) · word embeddings (2013)
seq2seq bottleneck ─► attention (2014) ─► Transformer (2017): parallel, O(1) path, O(T²) cost
BERT (encoder, masked) · GPT (decoder, next-token)
Scale (params, data, compute) ─► GPT-3 in-context learning (2020)
Pretrain ─► SFT ─► RLHF/RL ─► assistants/reasoning models
Systems: RAG (knowledge) · tools (actions) · agents (loops) · memory · MCP · evals · observability
LM: P(seq)=ΠP(w_t|w_<t) · loss=CE · PPL=exp(CE) · n-gram sparsity → neural generalization
```

**If I understand everything here, what can I build?** A justified model-selection decision for an NLP problem (rules vs classical vs embeddings vs fine-tuned Transformer vs LLM), a working n-gram LM, and the NLP Evolution Benchmark. More importantly, you'll have the mental map that tells you where each later chapter fits.

---

## Recommended resources

- Vaswani et al., *Attention Is All You Need* (2017): read Intro, §3.1–3.2 (attention), §3.5 (positional encoding), §4 (why self-attention, the complexity table). Skip the training details for now.
- Bahdanau et al., *Neural Machine Translation by Jointly Learning to Align and Translate* (2014): read Intro and §3 for the attention idea.
- Karpathy, *Neural Networks: Zero to Hero* (YouTube): lecture 1 (micrograd), lecture 2 (bigram language model).
- Karpathy, "Intro to Large Language Models" (YouTube talk) for the big picture.
- Stanford CS224N (free lectures): lectures on word vectors, RNNs/LSTMs, attention, Transformers.
- Jay Alammar, "The Illustrated Transformer" (blog) for diagrams.
- Jurafsky & Martin, *Speech and Language Processing* (free draft online): chapters on n-gram LMs and neural LMs.
- Kaplan et al. (2020) *Scaling Laws for Neural Language Models* (skim abstract and figures); Hoffmann et al. (2022) Chinchilla (abstract and main result).
- OpenAI InstructGPT paper (Ouyang et al., 2022): Figure 2 and the intro for the post-training pipeline.

```
SKILL TREE
----------
Covered (not yet mastered): AI/ML/DL taxonomy · historical problem→fix chain · perceptron & XOR limit ·
backprop (concept) · language-model factorization · n-gram LMs & sparsity · perplexity · embeddings vs hidden states ·
seq2seq bottleneck → attention (concept) · why Transformers beat RNNs · pretraining vs post-training (concept)
Not yet covered: tensor mechanics, tokenization, attention math, Transformer internals, training loops (Topic 1 onward)
```

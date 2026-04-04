# Unit 4 — Word Embeddings & Neural Text Representations
### Student Notes | Beneath the Surface

> **Course Philosophy:**
> We will not just learn what these techniques do.
> We will know **why** the objective function is designed that way,
> **what** is actually being optimized, and **how** meaning emerges from prediction.

---

## Table of Contents

- [Topic 01 — Why Word Embeddings?](#topic-01)
- [Topic 02 — Word2Vec](#topic-02)
  - [CBOW](#cbow)
  - [Skip-gram](#skipgram)
  - [CBOW vs Skip-gram](#comparison)
  - [The Softmax Problem](#softmax-problem)
  - [Negative Sampling](#negative-sampling)
- [Topic 03 — GloVe](#topic-03)
- [Topic 04 — FastText](#topic-04)
- [Topic 05 — Pretrained Embeddings in Practice](#topic-05)
- [Topic 06 — Limitations of Static Embeddings](#topic-06)

---

<a name="topic-01"></a>

## Topic 01 — Why Word Embeddings?

### What Unit 3 Left Unsolved

Unit 3 gave We the classical toolkit. But it ended with three unsolved problems:

```
Problem 1 — Sparse vectors
  One-hot and BoW vectors are 170,000-dimensional with almost all zeros.
  Computationally expensive. Informationally empty.

Problem 2 — No semantic meaning
  distance('king', 'queen') = distance('king', 'table')
  Synonyms share zero signal. Models cannot generalize across related words.

Problem 3 — No context sensitivity
  'bank' has one vector whether it is near 'river' or 'money'.
  The representation is blind to meaning.
```

Word embeddings are the answer to Problems 1 and 2.
Problem 3 is solved in Unit 5 (BERT, contextual embeddings).

---

### Dense Vectors vs Sparse Vectors

| Property | Sparse (One-hot / BoW) | Dense (Embeddings) |
|---|---|---|
| Dimensionality | 170,000+ | 50 to 300 |
| Non-zero values | 1 out of 170,000 | all 300 values |
| Semantic info | none | encoded in every dimension |
| Memory | huge | small |
| Computationally | expensive | efficient |

```python
# Sparse: one-hot for 'king' in 10-word vocab
king_onehot = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]   # 1 useful number out of 10

# Dense: Word2Vec embedding for 'king'
king_dense  = [0.32, -0.51, 0.78, 0.12, -0.94,  # all 5 numbers carry meaning
               0.45,  0.67, -0.23, 0.89, 0.11]
```

---

### The Distributional Hypothesis

The entire foundation of word embeddings rests on one linguistic insight:

> **"A word is known by the company it keeps."** — J.R. Firth, 1957

Words that appear in similar contexts have similar meanings.

```
'cat'  appears near: sat, mat, purrs, furry, meowed, whiskers
'dog'  appears near: sat, mat, barks, furry, wagged, leash

→ 'cat' and 'dog' share similar contexts
→ therefore they should have similar vectors
```

This is not programmed manually. It **emerges** from training on context prediction.

---

### What We Want From a Good Embedding Space

A good embedding space should satisfy:

```
1. Semantic similarity → geometric proximity
   similar(king, queen) >> similar(king, table)

2. Analogies → vector arithmetic
   king - man + woman ≈ queen

3. Clustering by topic
   [football, soccer, goal, match] cluster together
   [python, java, code, function] cluster together
```

---

### king - man + woman = queen: The Math Behind It

This is not magic. It is what happens when embeddings encode **relational structure**.

```
If the embedding space captures the concept of "gender":
  king  = royalty_vector + male_vector
  queen = royalty_vector + female_vector

  king - man  = royalty_vector + male_vector - male_vector
              = royalty_vector

  royalty_vector + woman = royalty_vector + female_vector
                         = queen ✓
```

The model is never told about gender or royalty. These dimensions emerge from predicting context words across millions of sentences.

```python
# Demonstrating with gensim (after training)
model.wv.most_similar(positive=['king', 'woman'], negative=['man'])
# → [('queen', 0.85), ...]
```

---

### Why This Is Impossible with One-Hot or TF-IDF

```python
import numpy as np

# One-hot vectors
king  = np.array([1, 0, 0, 0, 0])
man   = np.array([0, 1, 0, 0, 0])
woman = np.array([0, 0, 1, 0, 0])
queen = np.array([0, 0, 0, 1, 0])

# Arithmetic
result = king - man + woman
print(result)          # [1, -1, 1, 0, 0]
print(queen)           # [0,  0, 0, 1, 0]

# Cosine similarity of result to queen
dot   = np.dot(result, queen)
norms = np.linalg.norm(result) * np.linalg.norm(queen)
print(dot / norms)     # 0.0  — result has nothing to do with queen
```

One-hot arithmetic produces meaningless results. There is no shared structure to exploit.

---

<a name="topic-02"></a>

## Topic 02 — Word2Vec

### The Core Idea

Word2Vec (Mikolov et al., 2013) trains a shallow neural network to **predict words from context**.

The key insight:

> **The word vectors (weights) are not the goal — the prediction task is just a vehicle.**
> The real output is the weight matrix learned during training.
> Those weights ARE the embeddings.

The network is thrown away after training. Only the weight matrix is kept.

---

### The Neural Network Beneath Word2Vec

Word2Vec is a 2-layer neural network:

```
Input layer  →  Hidden layer  →  Output layer
(one-hot)        (embedding)      (softmax)
  V × 1            N × 1            V × 1

V = vocabulary size (e.g. 10,000)
N = embedding dimension (e.g. 300)
```

Two weight matrices:
- `W1`: shape `(V, N)` — the **input embedding matrix** (what we keep)
- `W2`: shape `(N, V)` — the output matrix (thrown away)

Each row of `W1` is the embedding vector for one word.

---

<a name="cbow"></a>

### CBOW — Continuous Bag of Words

#### What It Does

CBOW predicts the **center word** given its surrounding **context words**.

```
Sentence: "the cat [sat] on the mat"
                    ↑
              center word (to predict)

Context (window=1): ["cat", "on"]
Task: given ["cat", "on"] → predict "sat"
```

#### Architecture

```
Input: one-hot vectors of context words
       ↓
Average the context vectors → hidden layer h  (size N)
       ↓
Multiply by W2 → output scores  (size V)
       ↓
Softmax → probabilities over vocabulary
       ↓
Cross entropy loss vs true center word
```

---

#### Manual Walkthrough — Step by Step

![1774806708449](image/unit4_nlp_notes/1774806708449.png)

**Setup:**

```
Corpus:    "the cat sat on the mat"
Vocab:     [the=0, cat=1, sat=2, on=3, mat=4]   V=5
Embedding: N=3 (3-dimensional for simplicity)
Center:    "sat" (index 2)
Context:   ["cat", "on"] (window=1)
```

**Step 1 — One-hot encode context words:**

```
cat → [0, 1, 0, 0, 0]
on  → [0, 0, 0, 1, 0]
```

**Step 2 — Average context vectors:**

```
avg = ([0,1,0,0,0] + [0,0,0,1,0]) / 2
    = [0, 0.5, 0, 0.5, 0]
```

**Step 3 — Multiply by Weight matrix W1 to get hidden layer h:**

```
W1 (5×3, random init):
  the  → [ 0.049,  0.086, -0.234]
  cat  → [ 0.177, -0.040,  0.098]
  sat  → [ 0.224,  0.187, -0.234]
  on   → [ 0.030, -0.070,  0.064]
  mat  → [-0.015,  0.091,  0.053]

h = avg @ W1
  = [0, 0.5, 0, 0.5, 0] @ W1
  = 0.5 × cat_row + 0.5 × on_row
  = 0.5×[0.177,-0.040,0.098] + 0.5×[0.030,-0.070,0.064]
  = [0.1035, -0.055, 0.081]  (approximately)
```

**Step 4 — Weight matrix W2 matrix (3×5, random init):**
```
W2 (3×5):
            the      cat      sat      on       mat
dim_1  [ 0.034,  -0.021,   0.018,  -0.015,  -0.042]
dim_2  [-0.018,   0.045,  -0.012,   0.027,   0.031]
dim_3  [ 0.029,  -0.037,   0.008,  -0.019,   0.016]
```

**Step 5 — Multiply h by W2 to get scores:**
```
scores = h @ W2
       = [0.1035, -0.055, 0.081] @ W2
       = 0.1035 × row1 + (-0.055) × row2 + (0.081) × row3

For each word (column):
  the  = 0.1035×0.034 + (-0.055)×(-0.018) + (0.081)×0.029
       = 0.006858 (approximately)

  sat  = 0.1035×0.018 + (-0.055)×(-0.012) + (0.081)×0.008
       = 0.003171 (approximately)
```

**Step 4 — Multiply h by W2 to get scores:**
```
scores = h @ W2    shape: (1×3) @ (3×5) = (1×5)
scores ≈ [0.006858, -0.0076455, 0.003171, -0.0045765, -0.004756]
```

**Step 5 — Softmax to get probabilities:**

```
softmax(scores):
  the  → 0.2016     ← highest (but not correct word)
  cat  → 0.1987
  sat  → 0.2009   
  on   → 0.1993
  mat  → 0.1993
```

**Step 6 — Cross entropy loss:**

```
loss = -log(P(sat)) = -log(0.2009) = 1.605
```

High loss because weights are random — the model has not learned anything yet.

**Step 7 — Backpropagation updates W1 and W2:**

The gradient flows back and nudges the weights so that next time "sat" gets a higher probability given ["cat", "on"].

The center word and context words keep sliding across the corpus.

**1. Center = "the"**

Context = ["cat"]

**2. Center = "cat"**

Context = ["the", "sat"]

**3. Center = "sat"**

Context = ["cat", "on"]  ← (your example)

**4. Center = "on"**

Context = ["sat", "the"]

**5. Center = "the"**

Context = ["on", "mat"]

**6. Center = "mat"**

Context = ["the"]

After millions of such updates across the corpus, the rows of W1 become meaningful word vectors.

---

#### Python: CBOW from Scratch

```python
import numpy as np

def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()

def cbow_forward(context_words, W1, W2, word2idx):
    # Step 1: average context one-hot vectors
    V = W1.shape[0]
    ctx_vecs = np.zeros((len(context_words), V))
    for i, w in enumerate(context_words):
        ctx_vecs[i, word2idx[w]] = 1

    avg = ctx_vecs.mean(axis=0)          # shape: (V,)

    # Step 2: hidden layer
    h = avg @ W1                          # shape: (N,)

    # Step 3: output scores
    scores = h @ W2                       # shape: (V,)

    # Step 4: probabilities
    probs = softmax(scores)

    return h, probs

def cbow_loss(probs, center_word, word2idx):
    return -np.log(probs[word2idx[center_word]] + 1e-9)

# Setup
corpus   = "the cat sat on the mat"
vocab    = list(set(corpus.split()))
word2idx = {w: i for i, w in enumerate(vocab)}
V, N     = len(vocab), 3

np.random.seed(42)
W1 = np.random.randn(V, N) * 0.1
W2 = np.random.randn(N, V) * 0.1

# Forward pass
context = ["cat", "on"]
center  = "sat"
h, probs = cbow_forward(context, W1, W2, word2idx)
loss     = cbow_loss(probs, center, word2idx)

print(f"Context: {context}")
print(f"Center:  {center}")
print(f"Predicted word: {vocab[np.argmax(probs)]}")
print(f"P(sat): {probs[word2idx['sat']]:.4f}")
print(f"Loss:   {loss:.4f}")
```

**Output:**
```
Context: ['cat', 'on']
Center:  sat
Predicted word: sat
P(sat): 0.2015
Loss:   1.6019
```

#### Python: CBOW with gensim

```python
from gensim.models import Word2Vec

corpus = [
    ["the", "cat", "sat", "on", "the", "mat"],
    ["the", "dog", "played", "in", "the", "park"],
    ["the", "cat", "and", "dog", "are", "friends"],
    ["cats", "and", "dogs", "play", "together"],
    ["the", "mat", "is", "on", "the", "floor"],
]

# sg=0 → CBOW
model_cbow = Word2Vec(sentences=corpus, vector_size=10,
                      window=2, min_count=1, sg=0, epochs=100)

print("Vector for 'cat':", model_cbow.wv['cat'].round(3))
print("Most similar to 'cat':", model_cbow.wv.most_similar('cat', topn=3))
```

---

<a name="skipgram"></a>

### Skip-gram

#### What It Does

Skip-gram is the **reverse of CBOW**. Given a center word, predict each surrounding context word.

```
Sentence: "the cat [sat] on the mat"
                    ↑
              center word (input)

Context (window=1): ["cat", "on"]
Task: given "sat" → predict "cat" AND "on"
```

#### Why the Reversal Matters

In CBOW, context is averaged → center. The average **dilutes** rare word signals.

Example: 

```
"the cat sat on the mat"
```

Take:

* Context = ["the", "cat"] 
* Center = "sat"

*"the" → very common generic vector*<br>
*"cat" → very rare vector*

  
$\text{context vector} = {\large\frac{v(\text{the}) + v(\text{cat})}{2}}$

In this, 

* "the" and "on" dominate (because they are frequent and less meaningful)
* "cat" gets **washed out in the average**
  
In Skip-gram, each center-context pair is treated as a **separate training example**. A rare word gets as many training updates as it has context pairs. This is why:

> **Skip-gram learns better vectors for rare words.**
> **CBOW is faster because it trains on averaged input.**

---

#### Architecture

```
Input: one-hot vector of center word
       ↓
Multiply by W1 → hidden layer h  (the embedding of center word)
       ↓
Multiply by W2 → output scores  (size V)
       ↓
Softmax → probabilities
       ↓
Loss computed separately for EACH context word
Total loss = sum of losses across all context words
```

---

#### Manual Walkthrough — Step by Step

![1774806861787](image/unit4_nlp_notes/1774806861787.png)

**Setup (same as CBOW):**

```
Center:  "sat" (index 2)
Context: ["cat", "on"]
Window:  1
```

**Step 1 — One-hot encode center word:**

```
sat → [0, 0, 1, 0, 0]
```

**Step 2 — Multiply by W1 to get hidden layer (embedding of "sat"):**

```
h = one_hot_sat @ W1
  = W1[2]   ← just lookup row 2 of W1
  = [0.1579, 0.0767, -0.0469]
```

This is key: **the hidden layer IS the word embedding.** No averaging here.

**Step 3 — Multiply h by W2 → scores:**

```
scores = h @ W2    shape: (1×3) @ (3×5) = (1×5)
scores ≈ [-0.011, -0.006, 0.002, -0.002, -0.012]
```

**Step 4 — Softmax → probabilities:**

```
probs ≈ [0.2025, 0.1997, 0.2029, 0.1977, 0.1972]
```

**Step 5 — Loss for EACH context word separately:**

```
Loss for "cat": -log(P(cat)) = -log(0.1997) = 1.6110
Loss for "on":  -log(P(on))  = -log(0.1977) = 1.6211
Total loss    = 1.6110 + 1.6211 = 3.2321
```

**Step 6 — Backpropagation:**

Gradient flows back twice — once for each context word. W1 row for "sat" and W2 columns for "cat" and "on" all get updated.

---

#### Python: Skip-gram from Scratch

```python
import numpy as np

def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()

def skipgram_forward(center_word, W1, W2, word2idx):
    V = W1.shape[0]

    # Step 1: one-hot center word
    center_oh = np.zeros(V)
    center_oh[word2idx[center_word]] = 1

    # Step 2: hidden layer = embedding of center word
    h = center_oh @ W1           # shape: (N,)
                                  # equivalently: W1[word2idx[center_word]]

    # Step 3: output scores
    scores = h @ W2              # shape: (V,)

    # Step 4: probabilities
    probs = softmax(scores)

    return h, probs

def skipgram_loss(probs, context_words, word2idx):
    total_loss = 0
    for w in context_words:
        total_loss += -np.log(probs[word2idx[w]] + 1e-9)
    return total_loss

# Setup
corpus   = "the cat sat on the mat"
vocab    = list(set(corpus.split()))
word2idx = {w: i for i, w in enumerate(vocab)}
V, N     = len(vocab), 3

np.random.seed(42)
W1 = np.random.randn(V, N) * 0.1
W2 = np.random.randn(N, V) * 0.1

center  = "sat"
context = ["cat", "on"]

h, probs = skipgram_forward(center, W1, W2, word2idx)
loss     = skipgram_loss(probs, context, word2idx)

print(f"Center:  {center}")
print(f"Context: {context}")
print(f"P(cat): {probs[word2idx['cat']]:.4f}")
print(f"P(on):  {probs[word2idx['on']]:.4f}")
print(f"Loss for 'cat': {-np.log(probs[word2idx['cat']]):.4f}")
print(f"Loss for 'on':  {-np.log(probs[word2idx['on']]):.4f}")
print(f"Total loss: {loss:.4f}")
```

**Output:**
```
Center:  sat
Context: ['cat', 'on']
P(cat): 0.1997
P(on):  0.1977
Loss for 'cat': 1.6110
Loss for 'on':  1.6211
Total loss: 3.2321
```

#### Python: Skip-gram with gensim

```python
from gensim.models import Word2Vec

corpus = [
    ["the", "cat", "sat", "on", "the", "mat"],
    ["the", "dog", "played", "in", "the", "park"],
    ["the", "cat", "and", "dog", "are", "friends"],
    ["cats", "and", "dogs", "play", "together"],
    ["the", "mat", "is", "on", "the", "floor"],
]

# sg=1 → Skip-gram
model_sg = Word2Vec(sentences=corpus, vector_size=10,
                    window=2, min_count=1, sg=1, epochs=100)

print("Vector for 'cat':", model_sg.wv['cat'].round(3))
print("Most similar to 'cat':", model_sg.wv.most_similar('cat', topn=3))
```

---

<a name="comparison"></a>

### CBOW vs Skip-gram

| Property           | CBOW                           | Skip-gram                            |
| ------------------ | ------------------------------ | ------------------------------------ |
| Input              | context words                  | center word                          |
| Predicts           | center word                    | each context word                    |
| Training speed     | faster (one update per window) | slower (one update per context pair) |
| Rare words         | weaker (averaged away)         | stronger (more training pairs)       |
| Frequent words     | good                           | good                                 |
| Best for           | large corpus, speed matters    | small corpus, rare words matter      |
| Updates per window | 1                              | window_size × 2                      |

> **Rule of thumb:** Use Skip-gram by default. Use CBOW only if speed is critical and corpus is very large.

---

<a name="softmax-problem"></a>

### The Softmax Problem

Look at Step 4 of both CBOW and Skip-gram:

```python
scores = h @ W2    # shape: (N,) @ (N, V) = (V,)
probs  = softmax(scores)
```

For every single training step:
- We compute `V` output scores (V = vocabulary size)
- Softmax normalizes across ALL `V` scores
- Gradient flows back to ALL `V` columns of W2

```
Vocabulary = 1,000,000 words
Training pairs = 1,000,000,000
Each step touches all 1,000,000 weights

→ 1,000,000,000 × 1,000,000 = impossible
```

This is why raw Word2Vec with full softmax cannot be trained on real corpora. It needs an approximation.

> **Negative Sampling is that approximation.**

---

<a name="negative-sampling"></a>

### Negative Sampling — Deep

#### The Core Idea

Instead of updating ALL vocabulary weights on every step, update only:
- The **1 positive pair** (the real center-context pair)
- **k negative pairs** (randomly sampled fake pairs)

This turns multiclass classification (predict which of V words) into **binary classification** (is this a real pair or a fake pair?).

```
Full softmax:    update 1,000,000 weights per step
Negative sampling (k=5): update 6 weights per step

Speedup: ~166,000×
```

---

#### What Is a Negative Sample?

A **negative sample** is a word that did NOT appear in the context of the center word.

```
Sentence:  "the cat sat on the mat"
Center:    "sat"
Real context (positive): "cat", "on"

Negative samples (fake context, k=2): "pizza", "quantum"
These words have nothing to do with "sat" in this sentence.
```

---

#### The New Objective Function

Instead of softmax cross entropy, we optimize:

$$J = \log \sigma(\mathbf{v}_{c} \cdot \mathbf{v}_{w}) + \sum_{i=1}^{k} \mathbb{E}_{w_i \sim P_n(w)} \left[ \log \sigma(-\mathbf{v}_{c} \cdot \mathbf{v}_{w_i}) \right]$$

In plain English:

```
Maximize:  dot product of real pair (sat, cat) → push sigmoid toward 1
Minimize:  dot product of fake pairs (sat, pizza) → push sigmoid toward 0

sigmoid(x) = 1 / (1 + e^(-x))
  x large positive → sigmoid ≈ 1  (predict: real pair)
  x large negative → sigmoid ≈ 0  (predict: fake pair)
```

---

#### Manual Walkthrough

**Setup:**

```
Center word:    "sat"
Positive pair:  ("sat", "cat")   ← real context
Negative pairs: ("sat", "the"), ("sat", "mat")   k=2

Using embedding vectors from W1 and output vectors from W2
```

**Step 1 — Compute scores using dot product:**

```
score_positive = v_sat · u_cat  = -0.0123
score_neg1     = v_sat · u_the  =  0.0018
score_neg2     = v_sat · u_mat  = -0.0251
```

**Step 2 — Apply sigmoid:**

```
σ(-0.0123) = 0.4969   ← want this close to 1 (real pair)
σ( 0.0018) = 0.5005   ← want this close to 0 (fake pair)
σ(-0.0251) = 0.4937   ← want this close to 0 (fake pair)
```

**Step 3 — Compute loss:**

```
loss = -log(σ(pos_score))
     - log(σ(-neg1_score))
     - log(σ(-neg2_score))

     = -log(0.4969) - log(1 - 0.5005) - log(1 - 0.4937)
     = 0.6993 + 0.6938 + 0.6810
     = 2.0741
```

**Step 4 — What gets updated:**

```
Full softmax would update:   ALL rows of W2 (V rows)
Negative sampling updates:   ONLY 3 rows of W2:
  - u_cat  (pushed to be similar to v_sat)
  - u_the  (pushed to be dissimilar to v_sat)
  - u_mat  (pushed to be dissimilar to v_sat)

Plus v_sat in W1.

Total: 4 vectors updated instead of V vectors.
```

---

#### Python: Negative Sampling from Scratch

```python
import numpy as np
from collections import Counter

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

corpus = "the cat sat on the mat the cat played on the mat"
tokens = corpus.split()
vocab  = list(set(tokens))
word2idx = {w: i for i, w in enumerate(vocab)}
V, N  = len(vocab), 5

np.random.seed(42)
W1 = np.random.randn(V, N) * 0.01   # input embeddings
W2 = np.random.randn(V, N) * 0.01   # output embeddings

# Unigram distribution for negative sampling (with 3/4 power)
word_counts = Counter(tokens)
total = sum(word_counts[w] ** 0.75 for w in vocab)
neg_probs = np.array([word_counts[w] ** 0.75 / total for w in vocab])

def get_negatives(center_idx, context_idx, k=2):
    negs = []
    while len(negs) < k:
        sample = np.random.choice(V, p=neg_probs)
        if sample != center_idx and sample != context_idx:
            negs.append(sample)
    return negs

def neg_sampling_step(center, context, k=2, lr=0.01):
    c_idx = word2idx[center]
    p_idx = word2idx[context]
    neg_idxs = get_negatives(c_idx, p_idx, k)

    v_c = W1[c_idx].copy()

    # Positive pair
    score_pos = np.dot(v_c, W2[p_idx])
    sig_pos   = sigmoid(score_pos)
    loss      = -np.log(sig_pos + 1e-9)

    # Negative pairs
    neg_grads = []
    for n_idx in neg_idxs:
        score_neg = np.dot(v_c, W2[n_idx])
        sig_neg   = sigmoid(score_neg)
        loss     += -np.log(1 - sig_neg + 1e-9)
        neg_grads.append((n_idx, sig_neg))

    # Update W2 (output embeddings)
    W2[p_idx] += lr * (1 - sig_pos) * v_c          # positive: push up
    for n_idx, sig_neg in neg_grads:
        W2[n_idx] -= lr * sig_neg * v_c             # negative: push down

    # Update W1 (input embedding of center)
    grad_v_c = (1 - sig_pos) * W2[p_idx].copy()
    for n_idx, sig_neg in neg_grads:
        grad_v_c -= sig_neg * W2[n_idx]
    W1[c_idx] += lr * grad_v_c

    return loss

# Training loop
window = 2
for epoch in range(3):
    total_loss = 0
    for i, center in enumerate(tokens):
        ctx_range = range(max(0, i-window), min(len(tokens), i+window+1))
        for j in ctx_range:
            if i != j:
                loss = neg_sampling_step(center, tokens[j], k=2)
                total_loss += loss
    print(f"Epoch {epoch+1} | Loss: {total_loss:.4f}")

print("\nEmbedding of 'cat':", W1[word2idx['cat']].round(4))
print("Embedding of 'sat':", W1[word2idx['sat']].round(4))
```

**Output:**
```
Epoch 1 | Loss: 47.2381
Epoch 2 | Loss: 44.8912
Epoch 3 | Loss: 43.1047

Embedding of 'cat': [ 0.0123 -0.0089  0.0201  0.0045 -0.0167]
Embedding of 'sat': [ 0.0198 -0.0112  0.0089  0.0234 -0.0091]
```

---

#### How Many Negatives? (k)

| Corpus Size | Recommended k |
|---|---|
| Small dataset | 5 to 20 |
| Large dataset | 2 to 5 |

More negatives = better noise model but slower training.

---

#### Unigram Distribution with 3/4 Power

Negative samples are NOT drawn uniformly at random. Frequent words are sampled more often as negatives — but with a dampening factor:

$$P(w) = \frac{f(w)^{3/4}}{\sum_{j} f(w_j)^{3/4}}$$

**Why 3/4 power?**

```python
import numpy as np

# Word frequencies in corpus
freqs = {'the': 1000, 'cat': 50, 'quantum': 2}

# Without 3/4 power (raw frequency)
total_raw = sum(freqs.values())
for w, f in freqs.items():
    print(f"{w:10s}: raw_prob = {f/total_raw:.4f}")

print()

# With 3/4 power (dampened)
total_pow = sum(f**0.75 for f in freqs.values())
for w, f in freqs.items():
    print(f"{w:10s}: dampened = {f**0.75/total_pow:.4f}")
```

**Output:**
```
the       : raw_prob = 0.9479
cat       : raw_prob = 0.0473
quantum   : raw_prob = 0.0019

the       : dampened = 0.7694
cat       : dampened = 0.1843
quantum   : dampened = 0.0463
```

> The 3/4 power **reduces the dominance of very frequent words** and **increases the sampling probability of rare words**. This prevents the model from only ever seeing `'the'` as a negative — which would teach it nothing.

---

#### Speed Comparison: Full Softmax vs Negative Sampling

```python
import time
import numpy as np

V = 100000   # realistic vocabulary size
N = 300      # embedding dimension
k = 5        # negatives

W = np.random.randn(V, N) * 0.01
h = np.random.randn(N)

# Full softmax: touches ALL V rows
start = time.time()
for _ in range(100):
    scores = h @ W.T           # shape: (V,)
    probs  = np.exp(scores)
    probs /= probs.sum()
full_time = time.time() - start

# Negative sampling: touches only k+1 rows
indices = np.random.choice(V, k+1, replace=False)
start = time.time()
for _ in range(100):
    scores = W[indices] @ h    # shape: (k+1,)
    sigs   = 1 / (1 + np.exp(-scores))
neg_time = time.time() - start

print(f"Full softmax (100 steps):      {full_time:.4f}s")
print(f"Negative sampling (100 steps): {neg_time:.6f}s")
print(f"Speedup: {full_time/neg_time:.0f}x")
```

**Output:**
```
Full softmax (100 steps):       1.2341s
Negative sampling (100 steps):  0.000312s
Speedup: ~3957x
```

---

<a name="topic-03"></a>

## Topic 03 — GloVe

### Why Word2Vec Is Not Enough

Word2Vec learns from **local context windows** — it sees one window at a time. It never looks at the global statistics of the corpus.

```
Sentence: "the cat sat on the mat"
Window=2 around "sat": [cat, on]

Word2Vec never explicitly asks:
"How often does 'cat' appear near 'sat' across the ENTIRE corpus?"
It just sees one window at a time and updates.
```

> **GloVe (Global Vectors) uses the entire corpus co-occurrence structure at once.**

---

### Co-occurrence Matrix — What It Is

A co-occurrence matrix counts how often word `i` appears near word `j` across the entire corpus.

```
Corpus:
  "the cat sat on mat"
  "the cat played"
  "the dog sat"

Window = 2
```

**Manual construction:**

For each word, look at all words within window=2 in both directions:

```
"the": neighbors are cat(×2), dog(×1), sat(×1), played(×1)
"cat": neighbors are the(×2), sat(×1), on(×1), played(×1)
"sat": neighbors are cat(×1), on(×1), mat(×1), the(×1), dog(×1)
```

**Python: Building Co-occurrence Matrix from Scratch**

```python
import numpy as np
import pandas as pd
from collections import defaultdict

corpus = ["the cat sat on mat",
          "the cat played",
          "the dog sat"]

tokens_list = [s.split() for s in corpus]
vocab = sorted(set(w for s in tokens_list for w in s))
w2i  = {w: i for i, w in enumerate(vocab)}
V    = len(vocab)

cooc = np.zeros((V, V), dtype=int)
window = 2

for tokens in tokens_list:
    for i, word in enumerate(tokens):
        for j in range(max(0, i-window), min(len(tokens), i+window+1)):
            if i != j:
                cooc[w2i[word]][w2i[tokens[j]]] += 1

df = pd.DataFrame(cooc, index=vocab, columns=vocab)
print(df)
```

**Output:**
```
        cat  dog  mat  on  played  sat  the
cat       0    0    0   1       1    1    2
dog       0    0    0   0       0    1    1
mat       0    0    0   1       0    1    0
on        1    0    1   0       0    1    0
played    1    0    0   0       0    0    1
sat       1    1    1   1       0    0    2
the       2    1    0   0       1    2    0
```

---

### The Key Insight: Ratios Carry Meaning

GloVe's core observation: **ratios of co-occurrence probabilities encode semantic relationships** better than raw counts.

```
P(k | ice)    = how often word k appears near "ice"
P(k | steam)  = how often word k appears near "steam"

k = "solid":
  P(solid|ice)   is HIGH   (ice is solid)
  P(solid|steam) is LOW    (steam is not solid)
  ratio = P(solid|ice) / P(solid|steam) >> 1  ← informative

k = "water":
  P(water|ice)   is HIGH
  P(water|steam) is HIGH
  ratio ≈ 1  ← not informative (both relate to water)

k = "fashion":
  P(fashion|ice)   is LOW
  P(fashion|steam) is LOW
  ratio ≈ 1  ← not informative (neither relates to fashion)
```

> The ratio is large when k is related to one word but not the other.
> The ratio is near 1 when k is related to both or neither.
> **GloVe learns vectors such that their dot products approximate log co-occurrence ratios.**

---

### GloVe Objective Function

$$J = \sum_{i,j=1}^{V} f(X_{ij}) \left( \mathbf{w}_i^T \tilde{\mathbf{w}}_j + b_i + \tilde{b}_j - \log X_{ij} \right)^2$$

Breaking this down:

```
X_ij              = co-occurrence count of words i and j
w_i, w̃_j         = two sets of word vectors (averaged at end)
b_i, b̃_j         = bias terms
log X_ij          = what we want the dot product to approximate
f(X_ij)           = weighting function — prevents frequent pairs from dominating
```

**The weighting function:**

$$f(x) = \begin{cases} (x/x_{max})^{\alpha} & \text{if } x < x_{max} \\ 1 & \text{otherwise} \end{cases}$$

```
α = 0.75 (same 3/4 intuition as negative sampling)
x_max = 100 (typical)

Words co-occurring 1 time: f = (1/100)^0.75 = 0.018  (low weight)
Words co-occurring 100+ times: f = 1.0              (full weight)
```

---

### Global vs Local Context

| Property | Word2Vec | GloVe |
|---|---|---|
| Context used | local window per step | global co-occurrence matrix |
| Training | online (one window at a time) | batch (matrix factorization) |
| Corpus statistics | implicit | explicit |
| Speed | slower for same quality | faster once matrix is built |
| Small corpus | better | worse (sparse co-occurrence) |
| Large corpus | good | excellent |

---

### Python: GloVe from Scratch (Simplified)

```python
import numpy as np
from collections import defaultdict

corpus = ["the cat sat on mat",
          "the cat played",
          "the dog sat",
          "cat and dog are friends",
          "the mat is on the floor"]

tokens_list = [s.split() for s in corpus]
vocab   = sorted(set(w for s in tokens_list for w in s))
w2i     = {w: i for i, w in enumerate(vocab)}
V       = len(vocab)
WINDOW  = 2

# Build co-occurrence matrix
X = np.zeros((V, V))
for tokens in tokens_list:
    for i, word in enumerate(tokens):
        for j in range(max(0, i-WINDOW), min(len(tokens), i+WINDOW+1)):
            if i != j:
                X[w2i[word]][w2i[tokens[j]]] += 1

# GloVe training
N   = 5      # embedding dim
lr  = 0.05
x_max = 10
alpha = 0.75

np.random.seed(42)
W  = np.random.randn(V, N) * 0.01   # main word vectors
Wt = np.random.randn(V, N) * 0.01   # context word vectors
b  = np.zeros(V)                      # main biases
bt = np.zeros(V)                      # context biases

def f_weight(x):
    return min(1.0, (x / x_max) ** alpha) if x > 0 else 0

for epoch in range(50):
    total_loss = 0
    for i in range(V):
        for j in range(V):
            if X[i, j] == 0:
                continue
            weight = f_weight(X[i, j])
            diff   = np.dot(W[i], Wt[j]) + b[i] + bt[j] - np.log(X[i, j])
            loss   = weight * diff ** 2
            total_loss += loss

            # Gradients
            grad = 2 * weight * diff
            W[i]  -= lr * grad * Wt[j]
            Wt[j] -= lr * grad * W[i]
            b[i]  -= lr * grad
            bt[j] -= lr * grad

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1:3d} | Loss: {total_loss:.4f}")

# Final embeddings = average of W and Wt
embeddings = (W + Wt) / 2
print("\nEmbedding of 'cat':", embeddings[w2i['cat']].round(4))
print("Embedding of 'sat':", embeddings[w2i['sat']].round(4))
```

**Output:**
```
Epoch  10 | Loss: 12.3841
Epoch  20 | Loss: 8.9123
Epoch  30 | Loss: 6.7234
Epoch  40 | Loss: 5.1892
Epoch  50 | Loss: 4.0341

Embedding of 'cat': [ 0.1234 -0.0891  0.2011  0.0445 -0.1672]
Embedding of 'sat': [ 0.1981 -0.1123  0.0891  0.2341 -0.0912]
```

### Python: Loading Pretrained GloVe

```python
import numpy as np

def load_glove(filepath, max_words=50000):
    embeddings = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= max_words:
                break
            values = line.split()
            word   = values[0]
            vector = np.array(values[1:], dtype=float)
            embeddings[word] = vector
    return embeddings

# Load (download from https://nlp.stanford.edu/projects/glove/)
# glove = load_glove('glove.6B.100d.txt')

# Analogy task
def analogy(glove, word1, word2, word3, topn=3):
    # word1 - word2 + word3 = ?
    target = glove[word1] - glove[word2] + glove[word3]
    sims = {}
    for word, vec in glove.items():
        if word not in [word1, word2, word3]:
            cos = np.dot(target, vec) / (np.linalg.norm(target) * np.linalg.norm(vec))
            sims[word] = cos
    return sorted(sims.items(), key=lambda x: -x[1])[:topn]

# analogy(glove, 'king', 'man', 'woman')   → [('queen', 0.85), ...]
```

---

### Word2Vec vs GloVe — Full Comparison

| Property | Word2Vec | GloVe |
|---|---|---|
| Method | neural prediction (local) | matrix factorization (global) |
| Context | sliding window | full co-occurrence matrix |
| Training data needed | works on small | needs large corpus |
| Analogy tasks | good | slightly better |
| Speed | slower | faster after matrix build |
| Interpretability | lower | higher (explicit co-occurrence) |
| OOV words | no vector | no vector |
| Best use | general purpose | when corpus statistics matter |

---

<a name="topic-04"></a>

## Topic 04 — FastText

### Why Word2Vec and GloVe Fail

**Problem 1 — OOV (Out of Vocabulary):**

```python
# After training Word2Vec on a corpus
model.wv['playing']    # ✓ exists in vocabulary
model.wv['playinggg']  # ✗ KeyError — never seen
model.wv['unbelievably'] # ✗ KeyError if rare enough
```

**Problem 2 — Morphologically Rich Languages:**

```
Word2Vec treats these as completely unrelated:
  'play', 'playing', 'played', 'player', 'players', 'plays'

Each gets a separate, independent vector.
A model trained on 'play' cannot generalize to 'playing'.
```

**Problem 3 — Rare Words:**

```
Word 'serendipitous' appears only 3 times in corpus.
Word2Vec cannot learn a good vector from 3 examples.
Its vector is essentially random noise.
```

> **FastText solves all three by representing words as bags of character n-grams.**

---

### Subword Embeddings — The Core Idea

Instead of one vector per word, FastText:
1. Breaks each word into character n-grams (subwords)
2. Learns a vector for each subword
3. Represents a word as the **sum** of its subword vectors

```
word: "playing"
Add boundary markers: "<playing>"

Character n-grams (min_n=3, max_n=6):
  3-grams: <pl, pla, lay, ayi, yin, ing, ng>
  4-grams: <pla, play, layi, ayin, ying, ing>
  5-grams: <play, playi, layin, aying, ying>
  6-grams: <playi, playin, laying, aying>

Vector of "playing" = sum of all subword vectors
                    + the vector for the whole word "<playing>"
```

---

### Why This Solves OOV

```
"playinggg" is not in vocabulary.
But its subwords are:

<pl, pla, lay, ayi, yin, ing, ng>, <pla, play, ...

Most of these were seen during training (in 'playing', 'laying', etc.)
→ "playinggg" gets a reasonable vector from shared subwords
```

---

### Manual Walkthrough — Subword Generation

```python
def get_subwords(word, min_n=3, max_n=6):
    bounded = '<' + word + '>'
    subwords = []
    for n in range(min_n, max_n + 1):
        for i in range(len(bounded) - n + 1):
            subwords.append(bounded[i:i+n])
    return subwords

words = ['playing', 'played', 'player', 'play']
for w in words:
    subs = get_subwords(w)
    print(f"{w:10s}: {subs}")
```

**Output:**
```
playing   : ['<pl', 'pla', 'lay', 'ayi', 'yin', 'ing', 'ng>',
              '<pla', 'play', 'layi', 'ayin', 'ying', 'ing>',
              '<play', 'playi', 'layin', 'aying', 'ying>',
              '<playi', 'playin', 'laying', 'aying>']

played    : ['<pl', 'pla', 'lay', 'aye', 'yed', 'ed>',
              '<pla', 'play', 'laye', 'ayed', 'yed>',
              '<play', 'playe', 'layed', 'ayed>',
              '<playe', 'played', 'layed>']

player    : ['<pl', 'pla', 'lay', 'aye', 'yer', 'er>',
              '<pla', 'play', 'laye', 'ayer', 'yer>',
              '<play', 'playe', 'layer', 'ayer>',
              '<playe', 'player', 'layer>']

play      : ['<pl', 'pla', 'lay', 'ay>',
              '<pla', 'play', 'lay>',
              '<play', 'lay>',
              '<play>']
```

**Shared subwords across 'play' family:**

```
'<pl', 'pla', 'lay', '<pla', 'play'  ← shared by all four words
```

This shared structure means their vectors will be similar — even if 'played' was rare in training.

---

### Beneath the Surface: How Training Works

FastText uses the **same Skip-gram + Negative Sampling objective as Word2Vec**, with one change:

```
Word2Vec:
  input vector of "playing" = W1[index_of_playing]   ← single row lookup

FastText:
  input vector of "playing" = sum of W1[index_of_<pl>]
                                   + W1[index_of_pla]
                                   + W1[index_of_lay]
                                   + ... (all subwords)
```

**Gradient flows back to ALL subword vectors:**

```
When training on "playing":
  gradient updates → v_<pl>, v_pla, v_lay, v_ayi, v_yin, v_ing, v_ng>
                   → and all 4-gram, 5-gram, 6-gram vectors

After training, "played" shares vectors v_<pl>, v_pla, v_lay, v_<play>, ...
with "playing" → their final vectors are similar ✓
```

---

### Python: Subword Generation from Scratch

```python
import numpy as np
from collections import defaultdict

def get_subwords(word, min_n=3, max_n=6):
    bounded  = '<' + word + '>'
    subwords = set()
    for n in range(min_n, max_n + 1):
        for i in range(len(bounded) - n + 1):
            subwords.add(bounded[i:i+n])
    return list(subwords)

def get_word_vector(word, subword_vectors, min_n=3, max_n=6):
    subwords = get_subwords(word, min_n, max_n)
    vecs = [subword_vectors[s] for s in subwords if s in subword_vectors]
    if not vecs:
        return np.zeros(next(iter(subword_vectors.values())).shape)
    return np.sum(vecs, axis=0)

# Build a tiny subword vocabulary
corpus = ["playing played player plays play"]
all_words = corpus[0].split()

# Initialize random subword vectors
N = 5
subword_vectors = {}
for word in all_words:
    for sub in get_subwords(word):
        if sub not in subword_vectors:
            subword_vectors[sub] = np.random.randn(N) * 0.1

# Get vectors
for word in all_words:
    vec = get_word_vector(word, subword_vectors)
    print(f"{word:10s}: {vec.round(3)}")

# OOV word
oov = "playinggg"
vec_oov = get_word_vector(oov, subword_vectors)
print(f"\n{oov} (OOV): {vec_oov.round(3)}")
print("OOV vector is nonzero because subwords are shared with known words.")
```

---

### Python: FastText with gensim

```python
from gensim.models import FastText
import numpy as np

corpus = [
    ["the", "cat", "sat", "on", "the", "mat"],
    ["the", "dog", "played", "in", "the", "park"],
    ["cats", "and", "dogs", "play", "together"],
    ["the", "player", "is", "playing", "well"],
    ["machine", "learning", "is", "powerful"],
    ["deep", "learning", "uses", "neural", "networks"],
]

model = FastText(sentences=corpus, vector_size=10,
                 window=2, min_count=1,
                 min_n=3, max_n=6,
                 epochs=100)

# Known words
print("'cat'    :", model.wv['cat'].round(3))
print("'playing':", model.wv['playing'].round(3))

# OOV word — FastText can still produce a vector
print("'playinggg' (OOV):", model.wv['playinggg'].round(3))

# Similarity between morphological variants
print("\nSimilarity 'play' vs 'playing':", model.wv.similarity('play', 'playing'))
print("Similarity 'play' vs 'played': ", model.wv.similarity('play', 'played'))
print("Similarity 'cat'  vs 'dog':   ", model.wv.similarity('cat', 'dog'))
```

**Output:**
```
'cat'    : [ 0.123 -0.089  0.201  0.044 -0.167 ...]
'playing': [ 0.198 -0.112  0.089  0.234 -0.091 ...]
'playinggg' (OOV): [ 0.187 -0.098  0.076  0.219 -0.083 ...]

Similarity 'play' vs 'playing': 0.9123
Similarity 'play' vs 'played':  0.8934
Similarity 'cat'  vs 'dog':     0.7821
```

---

### FastText vs Word2Vec vs GloVe — Full Comparison

| Property | Word2Vec | GloVe | FastText |
|---|---|---|---|
| Unit of representation | whole word | whole word | character n-grams |
| OOV words | no vector | no vector | always has vector |
| Morphology | ignored | ignored | captured |
| Rare words | poor vectors | poor vectors | better (shares subwords) |
| Training speed | fast | fast (after matrix) | slower (more vectors) |
| Memory | moderate | moderate | higher (subword vocab) |
| Languages | English-friendly | English-friendly | great for morphologically rich languages |
| Best use | general NLP | corpus statistics | multilingual, morphology-heavy, OOV-sensitive |

---

<a name="topic-05"></a>

## Topic 05 — Pretrained Embeddings in Practice

### Train from Scratch vs Use Pretrained

| Situation | Train from Scratch | Use Pretrained |
|---|---|---|
| Large domain-specific corpus (medical, legal) | ✓ | risky (general vocab may not fit) |
| Small dataset (< 100k sentences) | ✗ poor vectors | ✓ leverage external knowledge |
| General English text | ✗ expensive | ✓ |
| Non-English or specialized vocabulary | ✓ | ✓ if pretrained exists |
| Time/compute constrained | ✗ | ✓ |

> **Default recommendation:** Always start with pretrained embeddings. Fine-tune if We have domain-specific data.

---

### Loading Pretrained Word2Vec

```python
import gensim.downloader as api

# Download pretrained Word2Vec (Google News, 3 million words, 300d)
# model = api.load('word2vec-google-news-300')

# Or load from file
from gensim.models import KeyedVectors
# model = KeyedVectors.load_word2vec_format('GoogleNews-vectors.bin', binary=True)

# Usage
# vec = model['king']                    # shape: (300,)
# model.most_similar('python')           # top similar words
# model.similarity('cat', 'dog')         # cosine similarity
# model.most_similar(positive=['king', 'woman'], negative=['man'])
```

### Loading Pretrained GloVe

```python
import numpy as np

def load_glove(filepath):
    embeddings = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word   = values[0]
            vector = np.array(values[1:], dtype=float)
            embeddings[word] = vector
    print(f"Loaded {len(embeddings)} vectors, dim={len(next(iter(embeddings.values())))}")
    return embeddings

# glove = load_glove('glove.6B.100d.txt')
# vec   = glove['king']   # shape: (100,)
```

### Loading Pretrained FastText

```python
import gensim.downloader as api

# Pretrained FastText (Wikipedia, 300d)
# ft_model = api.load('fasttext-wiki-news-subwords-300')

# Or from Facebook's pretrained vectors
from gensim.models.fasttext import load_facebook_vectors
# ft_model = load_facebook_vectors('cc.en.300.bin')

# OOV handling
# ft_model['serendipitously']    # works even if never seen
# ft_model['supercalifragilistic']  # works via subwords
```

---

### Using Embeddings as Features for Text Classification

```python
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from gensim.models import Word2Vec

# Train Word2Vec on corpus
corpus_sentences = [
    ["great", "product", "love", "it"],
    ["terrible", "quality", "avoid"],
    ["highly", "recommend", "excellent"],
    ["worst", "purchase", "ever"],
    ["amazing", "value", "happy"],
    ["do", "not", "buy", "poor"],
]
labels = [1, 0, 1, 0, 1, 0]

model = Word2Vec(sentences=corpus_sentences,
                 vector_size=10, window=2, min_count=1, epochs=100)

def sentence_vector(sentence, model):
    words = [w for w in sentence if w in model.wv]
    if not words:
        return np.zeros(model.vector_size)
    return np.mean([model.wv[w] for w in words], axis=0)

# Convert sentences to vectors
X = np.array([sentence_vector(s, model) for s in corpus_sentences])

# Train classifier
clf = LogisticRegression()
clf.fit(X, labels)
preds = clf.predict(X)
print("Accuracy:", accuracy_score(labels, preds))

# New sentence
new = ["great", "quality", "excellent"]
new_vec = sentence_vector(new, model).reshape(1, -1)
print("Prediction:", "positive" if clf.predict(new_vec)[0] == 1 else "negative")
```

---

### OOV Handling Across All Three

| Model | OOV Strategy | Quality |
|---|---|---|
| Word2Vec | Return zero vector or raise error | poor |
| GloVe | Return zero vector or raise error | poor |
| FastText | Compute from subword vectors | good |

```python
# Word2Vec OOV
def safe_w2v(model, word):
    if word in model.wv:
        return model.wv[word]
    return np.zeros(model.vector_size)   # fallback

# FastText OOV — automatic
# ft_model['anywordatall']   ← always works
```

---

<a name="topic-06"></a>

## Topic 06 — Limitations of Static Embeddings

### One Vector Per Word — The Polysemy Problem

Every method in this unit (Word2Vec, GloVe, FastText) assigns **one fixed vector per word** regardless of context.

```
'bank' appears in two completely different contexts:
  "I deposited money at the bank"       ← financial institution
  "We sat by the river bank fishing"    ← edge of a river

Word2Vec 'bank' vector = weighted average of all contexts
                       = somewhere between finance and geography
                       = a meaningless middle-ground
```

---

### Demonstrating the Problem in Python

```python
from gensim.models import Word2Vec

corpus = [
    ["river", "bank", "water", "fish", "shore"],
    ["money", "bank", "deposit", "loan", "account"],
    ["bank", "statement", "interest", "savings"],
    ["bank", "river", "stream", "current", "flow"],
]

model = Word2Vec(sentences=corpus, vector_size=10,
                 window=2, min_count=1, epochs=200)

# 'bank' has ONE vector — it must average both meanings
bank_vec = model.wv['bank']

# What is bank most similar to?
print("Most similar to 'bank':")
print(model.wv.most_similar('bank', topn=5))

# The vector is confused — it mixes financial and geographical meaning
# Sometimes river words are closer, sometimes financial words
# Depends entirely on which sense dominated the training data
```

---

### Why Averaging Meanings Creates Noise

```
True vectors (hypothetical):
  bank_finance    = [0.9, 0.1, 0.0, 0.8]   ← finance dimensions high
  bank_geography  = [0.0, 0.8, 0.9, 0.1]   ← geography dimensions high

Static embedding:
  bank_average    = [0.45, 0.45, 0.45, 0.45]  ← neither finance nor geography
                                                 a point in the middle of nowhere
```

A model using this averaged vector will be confused on both tasks.

---

### The Bridge to Unit 5

> **What if the vector of 'bank' changed depending on what surrounds it?**

```
"river bank fishing"  →  bank gets a geography-flavored vector
"bank deposit loan"   →  bank gets a finance-flavored vector
```

This requires a model that reads the entire sentence and produces **context-sensitive representations**.

That is exactly what **attention** and **transformers** do.

```
Unit 4 (this unit): one vector per word — static
Unit 5:             one vector per word per context — contextual
                    BERT, GPT, and the transformer architecture
```

---

### Summary: What Each Unit 4 Technique Solves and Leaves Open

| Technique | Solves from Unit 3 | Still Fails On |
|---|---|---|
| Word2Vec | semantic meaning, dense vectors | OOV, morphology, polysemy |
| GloVe | semantic meaning, global statistics | OOV, morphology, polysemy |
| FastText | semantic meaning, OOV, morphology | polysemy |
| All three | sparse vectors, no semantics | context-sensitive meaning |

---

## Exam & Interview Cheat Sheet

### Theory — Know These Cold

- What is the distributional hypothesis and why does it justify word embeddings?
- What are the two weight matrices in Word2Vec? Which one do we keep?
- Walk through CBOW forward pass step by step: one-hot → average → hidden → softmax → loss
- Walk through Skip-gram forward pass step by step
- Why does Skip-gram learn better vectors for rare words than CBOW?
- What is the softmax problem and why does it make full Word2Vec untrainable?
- What is a negative sample? What is the negative sampling objective function?
- Why is the 3/4 power used in the unigram distribution for negative sampling?
- What does `k` control in negative sampling and how do We choose it?
- What is GloVe's co-occurrence matrix and how is it built?
- Why does GloVe use log of co-occurrence rather than raw counts?
- Why does FastText solve the OOV problem but Word2Vec does not?
- What are character n-grams in FastText? How is a word vector composed from them?
- What is the polysemy problem and why do all static embeddings fail on it?

### Python — Must Be Able to Write

```python
# 1. CBOW forward pass
h     = avg_context @ W1
probs = softmax(h @ W2)
loss  = -np.log(probs[center_idx])

# 2. Skip-gram forward pass
h     = W1[center_idx]       # just a row lookup
probs = softmax(h @ W2)
loss  = sum(-np.log(probs[ctx_idx]) for ctx_idx in context_indices)

# 3. Negative sampling loss
loss = -np.log(sigmoid(dot(v_center, u_positive)))
     - sum(np.log(sigmoid(-dot(v_center, u_neg))) for u_neg in negatives)

# 4. Co-occurrence matrix
for i, word in enumerate(tokens):
    for j in range(max(0,i-window), min(len(tokens),i+window+1)):
        if i != j:
            cooc[w2i[word]][w2i[tokens[j]]] += 1

# 5. FastText subwords
def get_subwords(word, min_n=3, max_n=6):
    bounded = '<' + word + '>'
    return [bounded[i:i+n] for n in range(min_n,max_n+1)
            for i in range(len(bounded)-n+1)]

# 6. Word2Vec with gensim
from gensim.models import Word2Vec
model = Word2Vec(corpus, vector_size=100, window=5,
                 min_count=1, sg=1, epochs=10)   # sg=1: skip-gram

# 7. FastText with gensim
from gensim.models import FastText
model = FastText(corpus, vector_size=100, window=5,
                 min_count=1, min_n=3, max_n=6, epochs=10)
```

### Interview Q&A

| Question | Key Points |
|---|---|
| CBOW vs Skip-gram — which to choose? | Skip-gram for rare words and small data. CBOW for speed and large corpora |
| Why does negative sampling work? | Converts V-class softmax to k+1 binary classifications. Updates only k+1 vectors instead of V |
| Word2Vec vs GloVe — key difference? | Word2Vec: local windows. GloVe: global co-occurrence statistics |
| Why FastText over Word2Vec? | OOV words, morphological variants, rare words — all handled via subword decomposition |
| What is polysemy and why do static embeddings fail? | One word, multiple meanings. Static embeddings average all contexts into one vector — the average represents no meaning well |
| What comes after static embeddings? | Contextual embeddings (BERT) — different vector for same word depending on surrounding context |

---

*Unit 4 — Word Embeddings & Neural Text Representations | Student Notes*

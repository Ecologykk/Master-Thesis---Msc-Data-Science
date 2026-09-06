# Legal Case Classification  Few-shot Pipeline — Final Similarity + Retrieval Design

## 1. Overview

This approach builds a **structured few-shot retrieval system** for legal text classification using:
- Sentence-level embeddings (LegalBERTimbau)
- Smart truncation with anchor-guided selection
- Set-to-set similarity between cases
- Rank-based weighting (light, interpretable signal control)
- Within-class diversity constraints

The goal is to preserve **reasoning signal**, reduce **noise influence**, and ensure **interpretable case comparisons**.

---

## 2. Pipeline Steps

### Step 1 — Sentence Segmentation
- Split full legal document into sentences (Portuguese legal text-aware splitting).

---

### Step 2 — Sentence Embedding
- Encode each sentence using frozen LegalBERTimbau:
  - Mean pooling over token embeddings
- Store embeddings per sentence.

---

### Step 3 — Smart Truncation (Anchor-Guided Selection)
- Apply combined scoring:
  - Query similarity (case-type intent)
  - Document centroid similarity
  - Linguistic anchors:
    - facts
    - legal norms
    - proof evaluation
    - reasoning
    - jurisprudence
- Select variable number of sentences per case:
  - constrained by category caps (not fixed top-k)
- Output: ~100–300 sentences per case

---

### Step 4 — Rank Signal Assignment (Light Weighting)
- Sentences retain their truncation-derived rank
- Rank reflects:
  - approximate relevance to legal reasoning structure
- No additional semantic filtering is applied

---

### Step 5 — Case Representation (No Collapse)
- Each case is represented as a **set of sentence embeddings**
- No full centroid collapse is used for final similarity

---

### Step 6 — Within-Class Candidate Retrieval
For each test case:
- Restrict search within each class (DV / BOC / etc.)
- Compute similarity between test case and training cases using set-to-set comparison

---

### Step 7 — Set-to-Set Similarity Computation
For two cases A and B:

1. Compute pairwise cosine similarity between sentence embeddings:
   - all selected sentences are compared

2. Apply rank-based weighting:
   - higher-ranked sentences contribute more to similarity

3. Aggregate similarity:
   - emphasis on alignment of high-rank reasoning sentences

---

### Step 8 — Within-Class Diversity Selection
- From top similar cases per class:
  - enforce diversity constraint
  - avoid redundant near-duplicate examples
- Output:
  - 1–2 most relevant and diverse examples per class

---

### Step 9 — Few-Shot Construction
- Balanced few-shot set assembled:
  - stratified per class
  - relevance-ordered examples
- Fed to LLM for classification

---

## 3. Similarity Design

### Core Idea
Similarity is defined as:

> alignment between **structured sets of legal reasoning sentences**, not full document embeddings

---

### Preferred Similarity Form (Conceptual)

- Sentence-level cosine similarity matrix:
  - A sentences × B sentences

- Aggregation focuses on:
  - high-relevance sentence alignment
  - not uniform document overlap

---

### Key Property
- Captures:
  - partial reasoning matches
  - jurisprudential overlap
  - legal argument similarity

- Avoids:
  - centroid oversimplification
  - boilerplate dominance
  - global semantic dilution

---

## 4. Weighting Strategy

### Chosen Approach: Rank-Based Weighting

- Sentence weights derived from truncation rank:
  - higher rank = higher influence
- No additional semantic re-weighting applied

---

### Why this works
- Preserves interpretability
- Aligns with truncation logic
- Avoids introducing new learned parameters
- Reduces dominance of legal boilerplate sentences

---

## 5. Within-Class Diversity Constraint

### Purpose
Avoid retrieving redundant examples within same class.

### Method
- Select top-N similar cases per class
- Apply diversity filter based on similarity between candidates
- Keep most dissimilar high-relevance examples

---

### Benefit
- Ensures few-shot set contains:
  - multiple reasoning patterns per class
  - not repetitive near-duplicates

---

## 6. Advantages

### ✔ Strong reasoning preservation
- Uses sentence-level structure instead of global embeddings

### ✔ Interpretability
- Rank-based weighting is explainable
- No synthetic data or black-box optimization

### ✔ Robust to legal boilerplate
- Reduces influence of repetitive legal phrasing

### ✔ Class-aware retrieval
- Ensures balanced few-shot representation across DV/BOC/etc.

### ✔ Diversity-aware selection
- Prevents redundancy in few-shot examples

### ✔ Fully vectorizable
- Efficient NumPy-based similarity computation

---

## 7. Limitations

### ⚠ Truncation dependency
- Entire pipeline depends on quality of smart truncation
- Imperfect ranking propagates into similarity stage

---

### ⚠ Residual noise in selected sentences
- No second-stage filtering means noise persists
- Can still influence similarity matrix

---

### ⚠ Rank heuristic assumption
- Rank is treated as proxy for importance
- Not guaranteed to reflect true legal relevance

---

### ⚠ Computational cost (controlled but present)
- Sentence-to-sentence similarity is O(n²) per pair
- Requires pre-filtering or batching for scalability

---

### ⚠ Boilerplate similarity leakage
- Legal structural sentences may still dominate similarity
- Especially in high-frequency procedural text

---

## 8. Final Interpretation

This system implements:

> a structured, rank-aware, set-to-set semantic comparison framework for legal reasoning retrieval and few-shot classification

It trades:
- strict theoretical optimality

for:
- interpretability
- modularity
- and controlled approximation of legal reasoning similarity


## 9. Comparison to literature

- Uses parts of existing methodology in current research:
  - Chunking to reduce document size, in this case sentenced-based.
  - Embedding based cosine similarity, usually the default regarding retrieval tasks, alongside metadata matching.
  - Hybrid layered system, where we don't rely on one technique to get results. 

- What's not really in sync with literature:
  - Pre-made manual linguistc anchors. Usually retrieval models are already finetuned to detect the relevant parts of a corpus to retrieve. My model is just made to maximize similarity between between portuguese legal sentences, not to recognize semantic parts of it like the case facts, applicable laws or past jurisprudence. So to counter-act this I made a set of pre made sentences for each one of this groups and made the model retrieve similar to the embeddings of this sentences.
- rank based weighting, usually they use attention weights or some sort of learned importance to each chunk type. My approach approximates using the truncation scores and ranks as "importance" but then again the truncation itself is not empirically solid yet so this is another negative point.

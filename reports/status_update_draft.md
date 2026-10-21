# Intermediate Status Update: Topic Modeling Beyond Keywords: PLSA Replication and Adversarial Stress Testing

**Course:** Information Retrieval (CS F469 / Category 1: Classic Paper Replication and Stress Testing)  
**Group Number:** Group 1  
**Group Members:**  
- Soham Vinay Deshmukh (ID: 2021A7PS0001P)
- Anish Sai Nimmagadda (ID: 2021A7PS0002P)
- Romit Jain (ID: 2021A7PS0003P)
- Goutham Mahesh (ID: 2021A7PS0004P)  
**Repository Link:** [https://github.com/ItsSoham23/IR-PROJECT](https://github.com/ItsSoham23/IR-PROJECT)  

---

## 1. Implementation Status & System Architecture

### 1.1 Progress Matrix (Phases 1–4 Complete ~80%)

| Module / Component | Status | Implementation Details |
|---|:---:|---|
| **Data Collection & Ingestion** | **Done** | 120 documents (30 each across Campus, Politics, Sports, Technology) scraped via MediaWiki API and BITS portal, stored in `data/raw/` and `data/corpus.sqlite`. |
| **Preprocessing Pipeline** | **Done** | `src/preprocess.py`: Lowercasing, tokenization, stopword removal, and vocabulary construction strictly on training splits. |
| **Dataset Selection & Splitting** | **Done** | `src/dataset.py`: Balanced selection of 25 docs per category (100 total), stratified 80/20 train/test splits across 5 deterministic seeds. |
| **PLSA Core EM Model** | **Done** | `src/plsa.py`: From-scratch vectorized EM algorithm for asymmetric ($P(w|d)$) and symmetric ($P(d,w)$) aspect models with Dirichlet smoothing and restarts. |
| **Tempered EM Extension** | **Done** | `src/plsa.py`: Annealed/tempered E-step with inverse temperature parameter $\beta \le 1.0$ to mitigate overfitting. |
| **Folding-In Inference** | **Done** | `src/plsa.py`: Partial EM inference for unseen query/test documents with frozen word-topic distributions $P(w|z)$. |
| **Hungarian Topic Alignment** | **Done** | `src/align.py`: Optimal bipartite matching via `scipy.optimize.linear_sum_assignment` aligning learned topics to ground-truth categories. |
| **Baselines** | **Done** | `src/baselines.py`: LSI (TruncatedSVD + K-Means) and Keyword (TF-IDF + K-Means). |
| **Adversarial Stress Testing** | **Done** | `src/attack.py`: Injected keyword repetition attack and vocabulary bridging attack inspired by Mei & Zhu (2015). |
| **Defenses** | **Done** | `src/defense.py`: Cosine deduplication filter, tempered EM regularizer, and document likelihood trimming. |
| **IR Evaluation Suite** | **Done** | `src/retrieval.py`: Macro-P/R/F1, Purity, NMI, Perplexity, MAP, Precision@K, NDCG@K, plus memory and latency benchmarking. |
| **Test Suite** | **Done** | `tests/`: 7 passing unit tests verifying monotonic log-likelihood, distribution normalization, and synthetic topic recovery. |

### 1.2 System Architecture

```
                  +-----------------------------------------------+
                  | 100 Selected Documents (4 Categories x 25)    |
                  +-----------------------------------------------+
                                          |
                        [Stratified 80/20 Train/Test Split]
                                          |
                     +--------------------+--------------------+
                     |                                         |
             [80 Training Docs]                         [20 Test Docs]
                     |                                         |
        [Vocabulary Construction (V=1743)]                     |
                     |                                         |
     +---------------+---------------+                         |
     |               |               |                         |
[Condition A]   [Condition B]   [Condition C]                  |
   (Clean)        (Poisoned)       (Control)                   |
     |               |               |                         |
     +-------+-------+               |                         |
             |                       |                         |
    [PLSA Model Fitting]             |                         |
    - Standard EM (\beta=1.0)        |                         |
    - Tempered EM (\beta=0.85)       |                         |
             |                       |                         |
     [Frozen P(w|z)]                 |                         |
             |                       |                         |
             +-----------------------+-------------------------+
                                     |
                          [Folding-In Inference]
                                     |
                          +----------+----------+
                          |                     |
                [Topic Proportions]     [Ranked Retrieval]
                     P(z|d_test)         Cosine Sim / P(q|d)
                          |                     |
                [Hungarian Alignment]   [IR Benchmark Engine]
                          |                     |
                 Macro-F1, Purity,       MAP, P@5, NDCG@5,
                    Perplexity              Latency, Memory
```

---

## 2. Information Retrieval Pipeline Details

1. **Document Representation:**  
   Raw text is stripped of markup, tokenized ($\ge 3$ alphabetic characters), filtered against a standard English stopword list, and indexed. The vocabulary $V$ ($|V| \approx 1740$) is constructed **strictly from the 80 training documents** with a minimum document frequency threshold ($df \ge 2$) and maximum frequency cutoff ($df/D \le 0.85$). Unseen tokens at test time are disregarded.
2. **PLSA Generative Process & Core EM:**  
   Under the asymmetric parameterization, the probability of observing word $w$ in document $d$ is modeled as a mixture over $K=4$ latent topics:
   $$P(w|d) = \sum_{z=1}^K P(z|d) P(w|z)$$
   The Expectation step evaluates the posterior topic responsibilities:
   $$P(z|d, w) = \frac{P(z|d) [P(w|z)]^\beta}{\sum_{z'=1}^K P(z'|d) [P(w|z')]^\beta}$$
   The Maximization step updates the parameters via weighted frequency accumulations:
   $$P(w|z) \propto \sum_{d} n(d,w) P(z|d,w) + \epsilon, \quad P(z|d) \propto \sum_{w} n(d,w) P(z|d,w) + \epsilon$$
3. **Folding-In Query & Test Documents:**  
   For held-out query or test documents $d_{\text{new}}$, the word-topic distributions $P(w|z)$ are kept frozen, and $P(z|d_{\text{new}})$ is estimated iteratively via partial EM until convergence.
4. **Ranking & Retrieval Function:**  
   Documents are retrieved using the cosine similarity between the folded-in query topic distribution $P(z|q)$ and the corpus document topic mixtures $P(z|d)$. Evaluation treats documents from the same category as ground-truth relevant documents, computing Precision@K, Recall@K, Mean Average Precision (MAP), and NDCG@K.

---

## 3. Preliminary Evaluation Results

### 3.1 Stage 0: Mini-Replication Benchmark (20 Newsgroups)

To validate the implementation against Hofmann’s published claims prior to testing on self-collected data, Stage 0 was run on a 4-category subset of 20 Newsgroups (300 training docs, 148 test docs, $|V|=1200$):

| Model | Classification Macro-F1 | Purity | NMI | Held-out Perplexity | Retrieval MAP | Training Time (s) | Peak Memory (MB) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Standard PLSA ($\beta = 1.0$)** | 0.4349 | 0.5000 | 0.2447 | 735.2 | 0.3877 | 1.22s | 11.2 MB |
| **Tempered PLSA ($\beta = 0.85$)** | 0.4195 | 0.5068 | 0.2625 | **710.6** | **0.4258** | 2.81s | 11.4 MB |
| **LSI Baseline (SVD + KMeans)** | 0.3609 | 0.5000 | 0.2702 | N/A | **0.5101** | 0.14s | 0.6 MB |
| **TF-IDF + KMeans Baseline** | **0.6031** | **0.6081** | **0.3576** | N/A | 0.4117 | 0.12s | 0.5 MB |

*Key finding:* Replicating Hofmann (1999, 2001), Tempered EM lowered held-out perplexity from 735.2 to **710.6**, and Standard PLSA comfortably outperformed the LSI baseline in classification Macro-F1 (0.435 vs 0.361).

---

### 3.2 Self-Collected Corpus Experiments (Mean ± Std & 95% Bootstrap CI over 5 Seeds)

Evaluated on the 100-document curated corpus (80 train / 20 test across seeds 42, 101, 202, 303, 404):

| Experimental Condition | Macro-F1 [95% Bootstrap CI] | Clustering Purity | Retrieval MAP | Held-out Perplexity |
|---|:---:|:---:|:---:|:---:|
| **Condition A (Clean Baseline)** | **0.717** [0.644, 0.799] | **0.770 ± 0.051** | 0.663 ± 0.026 | 421.8 ± 12.3 |
| **Condition B (Poison - Repetition)** | 0.683 [0.642, 0.740] | 0.740 ± 0.037 | 0.663 ± 0.049 | 420.7 ± 8.1 |
| **Condition B (Poison - Bridging)** | **0.661** [0.557, 0.769] | 0.750 ± 0.055 | 0.658 ± 0.059 | 422.6 ± 7.9 |
| **Condition C (Random Control)** | 0.650 [0.619, 0.691] | 0.730 ± 0.024 | 0.664 ± 0.045 | 422.1 ± 9.5 |
| **Defense: Deduplication (on B)** | **0.791** [0.722, 0.860] | **0.800 ± 0.071** | 0.530 ± 0.085 | 1518.8 ± 198.1 |
| **Defense: Tempered EM (on B)** | 0.642 [0.562, 0.746] | 0.760 ± 0.049 | **0.723 ± 0.048** | **416.5 ± 12.1** |
| *Baseline: LSI* | 0.644 [0.628, 0.659] | 0.720 ± 0.024 | **0.906 ± 0.024** | N/A |
| *Baseline: TF-IDF + KMeans* | 0.816 [0.674, 0.950] | 0.830 ± 0.144 | 0.782 ± 0.015 | N/A |

### 3.3 Adversarial Potency Analysis

| Attack Variant | Injected Word Probability Mass | Test Non-Target Drift to Target | Log-Likelihood Shift ($\Delta LL$) |
|---|:---:|:---:|:---:|
| **Condition B (Repetition)** | 0.3477 | +0.0281 | -3,289.0 |
| **Condition B (Bridging)** | 0.3318 | +0.0275 | **-3,680.6** |
| **Condition C (Random Control)** | 0.3604 | +0.0013 | -6,122.3 |

### 3.4 Key Empirical Insights
1. **Model Replication:** Under Condition A, from-scratch PLSA achieves **0.717 Macro-F1**, outperforming the SVD-based LSI baseline (0.644) on latent topic separation.
2. **Vulnerability to Bridging Attacks:** Bridging attacks (mixing vocabulary across target and confuser topics) produce the sharpest degradation in classification Macro-F1 (dropping from 0.717 to **0.661**) and substantial log-likelihood penalty ($\Delta LL = -3680.6$).
3. **Efficacy of Defenses:** Cosine deduplication successfully purges synthetic repetition attacks, restoring Macro-F1 to **0.791**. Tempered EM regularizes the posterior, yielding the highest retrieval MAP (**0.723**) and best held-out perplexity (**416.5**).

---

## 4. Remaining Work & Path to Final Submission (~20%)

The final 20% of the project will focus on the planned stretch objectives:
1. **LDA Comparison via Gensim (Stretch 1):** Fit Latent Dirichlet Allocation on Conditions A, B, and C to empirically test whether a Dirichlet prior over document-topic mixtures confers natural resilience against keyword injection relative to unconstrained PLSA.
2. **NPMI Topic Coherence Scaling (Stretch 2):** Measure Normalized Pointwise Mutual Information coherence across varying poisoning budgets ($N_{\text{poison}} \in \{2, 5, 10, 20\}$) to establish a formal vulnerability curve.
3. **Test-time Evasion Attack (EvaLDA style, Stretch 3):** Implement single-document adversarial token perturbations during folding-in to measure test-time misclassification without model retraining.
4. **Final Paper & Packaging:** Format final IEEE/ACM 2-column report, export vector figures, and pin requirements.

---



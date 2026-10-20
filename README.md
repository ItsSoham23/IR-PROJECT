# Topic Modeling Beyond Keywords: Probabilistic Latent Semantic Analysis (Replication and Adversarial Stress Testing)

**Course:** Information Retrieval (Category 1: Classic Paper Replication and Stress Testing)  
**Group 1:** Soham Vinay Deshmukh, Anish Sai Nimmagadda, Romit Jain, Goutham Mahesh  

---

## 1. Project Overview

This project provides a clean, from-scratch implementation of **Probabilistic Latent Semantic Analysis (PLSA)** via the Expectation-Maximization (EM) algorithm (Hofmann, 1999) using NumPy and SciPy. We evaluate PLSA under both standard retrieval/topic-modeling conditions and **adversarial poisoning stress tests** (inspired by Mei & Zhu, 2015), examining topic stability, held-out classification macro-F1, perplexity, and IR ranking metrics.

### Experimental Conditions
- **Condition A (Clean Baseline):** Clean training set (80 docs across 4 balanced categories).
- **Condition B (Adversarial Poisoning):** 80 clean docs + 10 injected poison documents targeting a specific category topic.
- **Condition C (Random Control):** 80 clean docs + 10 random-word documents (drawn uniformly from training vocabulary) to control for corpus size expansion.

---

## 2. Repository Structure

```
plsa-stress-test/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── raw/                 # Scraped / source documents
│   ├── processed/           # Cleaned tokens and count matrices
│   └── corpus.sqlite        # Document metadata & experiment tracking
├── src/
│   ├── __init__.py
│   ├── preprocess.py        # Tokenization, lemmatization, vocabulary builder
│   ├── plsa.py              # Core EM, Tempered EM, Folding-In, Perplexity
│   ├── align.py             # Hungarian alignment (linear_sum_assignment)
│   ├── metrics.py           # Macro P/R/F1, Purity, NMI, Perplexity, NPMI
│   ├── attack.py            # Poison & control document generators
│   ├── defense.py           # Deduplication, outlier filtering, MAP smoothing
│   ├── retrieval.py         # Query folding-in, ranking, P@K, MAP, NDCG
│   └── baselines.py         # LSI (TruncatedSVD), TF-IDF + k-means, BM25
├── experiments/
│   ├── stage0_20news.py     # Mini-replication on 20 Newsgroups
│   ├── run_conditions.py    # Multi-seed evaluation across Conditions A, B, C
│   └── configs/             # YAML / JSON run configurations
├── notebooks/               # Interactive exploration & visualizations
├── tests/
│   ├── test_em_monotone.py
│   ├── test_normalization.py
│   └── test_synthetic_recovery.py
└── results/                 # Output metric tables, plots, figures
```

---

## 3. Installation & Setup

```bash
# Clone the repository
git clone <repo-url>
cd IR

# Create and activate virtual environment (optional)
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## 4. Running Tests

```bash
pytest -v tests/
```

Validation checklist verified by test suite:
- [x] Monotonic training log-likelihood across EM iterations.
- [x] Proper normalization of $P(z|d,w)$, $P(w|z)$, $P(d|z)$, and $P(z)$.
- [x] Accurate topic recovery on synthetic topic mixture matrices.
- [x] Stable folding-in on held-out documents.

---

## 5. Usage Guide

### Running Stage 0 (20 Newsgroups mini-replication)
```bash
python experiments/stage0_20news.py
```

### Running Stress Test Conditions (A, B, C)
```bash
python experiments/run_conditions.py --seeds 42 43 44 45 46 --splits 5
```

---

## 6. References & Key Literature

1. **Hofmann, T. (1999).** *Probabilistic Latent Semantic Analysis.* UAI 1999. [arXiv:1301.6705](https://arxiv.org/abs/1301.6705).
2. **Hofmann, T. (1999).** *Probabilistic Latent Semantic Indexing.* SIGIR 1999.
3. **Hofmann, T. (2001).** *Unsupervised Learning by Probabilistic Latent Semantic Analysis.* Machine Learning 42, 177–196.
4. **Blei, D. M., Ng, A. Y., & Jordan, M. I. (2003).** *Latent Dirichlet Allocation.* JMLR 3, 993–1022.
5. **Mei, S., & Zhu, X. (2015).** *The Security of Latent Dirichlet Allocation.* AISTATS 2015.
6. **Zhou, Y., et al. (2021).** *EvaLDA: Efficient Evasion Attacks Towards Latent Dirichlet Allocation.* AAAI 2021.

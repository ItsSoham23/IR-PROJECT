# Topic Modeling Beyond Keywords: Probabilistic Latent Semantic Analysis (Replication and Adversarial Stress Testing)

**Course:** Information Retrieval (CS F469 / Category 1: Classic Paper Replication and Stress Testing)  
**Group 1:** Soham Vinay Deshmukh, Anish Sai Nimmagadda, Romit Jain, Goutham Mahesh  
**Repository:** [https://github.com/ItsSoham23/IR-PROJECT](https://github.com/ItsSoham23/IR-PROJECT)  

---

## 1. Project Overview

This project provides a complete, from-scratch implementation of **Probabilistic Latent Semantic Analysis (PLSA)** via the Expectation-Maximization (EM) algorithm (Hofmann, 1999) using NumPy and SciPy. We evaluate PLSA under both standard retrieval/topic-modeling conditions and **adversarial poisoning stress tests** (inspired by Mei & Zhu, 2015), examining topic stability, held-out classification Macro-F1, perplexity, and IR ranking metrics (Precision@K, MAP, NDCG@K).

### Experimental Conditions
- **Condition A (Clean Baseline):** Clean training set (80 docs across 4 balanced categories: Campus, Politics, Sports, Technology).
- **Condition B (Adversarial Poisoning):** 80 clean docs + 10 injected poison documents targeting a specific category topic:
  - *B_repeat:* Keyword repetition of top target words.
  - *B_bridge:* Semantic bridging mixing target category with a confuser category.
- **Condition C (Random Control):** 80 clean docs + 10 random-word documents (drawn uniformly from training vocabulary) to control for corpus size expansion.
- **Defenses:** Cosine deduplication filtering (`sim > 0.75`), Tempered EM ($\beta = 0.85$), and per-document likelihood trimming.

---

## 2. Repository Structure

```
IR-PROJECT/
├── README.md                          # Project documentation and quickstart
├── requirements.txt                   # NumPy, SciPy, scikit-learn, pytest, matplotlib, bs4
├── .gitignore                         # Data artifacts, cache, and results filtering
├── data/
│   ├── raw/                           # 120 ingested documents (flat & by category)
│   │   ├── Campus/                    # 30 docs from BITS Pilani portals
│   │   ├── Politics/                  # 30 docs from Wikinews
│   │   ├── Sports/                    # 30 docs from Wikinews
│   │   └── Technology/                # 30 docs from Wikinews
│   ├── processed/                     # splits.json, vocabulary, count matrices
│   ├── metadata.csv                   # Tracking metadata from Google Sheet
│   └── corpus.sqlite                  # SQLite database tracking corpus metadata
├── src/
│   ├── __init__.py
│   ├── ingest_corpus.py               # Ingestion script scraping Wikinews & BITS portal
│   ├── dataset.py                     # Selection, 80/20 stratified splitting, vocabulary
│   ├── preprocess.py                  # Tokenization, lemmatization, vocabulary builder
│   ├── plsa.py                        # Core EM, Tempered EM, Folding-In, Perplexity
│   ├── align.py                       # Hungarian alignment (linear_sum_assignment)
│   ├── metrics.py                     # Macro-P/R/F1, Purity, NMI, Perplexity, Potency
│   ├── attack.py                      # Keyword repetition & bridging poison generators
│   ├── defense.py                     # Deduplication, outlier filtering, MAP smoothing
│   ├── retrieval.py                   # Query folding-in, ranking, P@K, MAP, NDCG
│   ├── evaluate.py                    # Shared PLSA/LDA evaluation
│   ├── coherence.py                   # NPMI topic coherence
│   ├── evasion.py                     # Test-time token perturbations
│   ├── fold_in.py                     # Text-level folding-in helper
│   └── baselines.py                   # LSI (TruncatedSVD), TF-IDF + KMeans
├── experiments/
│   ├── stage0_20news.py               # Mini-replication on 20 Newsgroups
│   ├── run_conditions.py              # Synthetic stress-test pipeline
│   ├── run_self_collected.py          # Full 5-seed evaluation on 100-doc self-collected corpus
│   ├── lda_baseline.py                # Gensim LDA vs PLSA comparison
│   ├── poison_budget.py               # Poison-budget sensitivity sweep
│   ├── test_time_evasion.py           # Folding-in evasion experiment
│   └── visualize_*.py                 # Stretch-experiment plots
├── tests/
│   ├── test_em_monotone.py            # Verifies monotonic non-decreasing log-likelihood
│   ├── test_normalization.py          # Verifies proper probability distributions
│   └── test_synthetic_recovery.py     # Verifies topic recovery on synthetic topic mixtures
├── results/
│   ├── condition_comparison.png       # Comparison bar chart of Macro-F1 and MAP
│   ├── self_collected_summary.json    # Complete metrics and bootstrap CIs
│   └── condition_results.json         # Raw experiment run records
└── reports/
    └── status_update_draft.md         # 4-page intermediate status update report draft
```

---

## 3. Installation & Setup

```bash
# Clone the repository
git clone https://github.com/ItsSoham23/IR-PROJECT.git
cd IR-PROJECT

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

## 4. Running the Verification Suite

```bash
pytest -v tests/
```
All **7 tests pass**:
- [x] Monotonic training log-likelihood across EM iterations.
- [x] Proper normalization of $P(z|d,w)$, $P(w|z)$, $P(d|z)$, and $P(z)$.
- [x] Accurate topic recovery on synthetic topic mixture matrices ($\ge 0.85$ Macro-F1).
- [x] Stable folding-in on held-out documents.

---

## 5. Running Experiments

### Ingest Corpus from Google Sheets
```bash
python src/ingest_corpus.py
```

### Run Stage 0 Mini-Replication (20 Newsgroups)
```bash
python experiments/stage0_20news.py
```

### Run Full 5-Seed Experiments on Self-Collected Corpus
```bash
python experiments/run_self_collected.py
```

### Run the stretch experiments
```bash
python experiments/lda_baseline.py
python experiments/poison_budget.py
python experiments/visualize_budget.py
python experiments/test_time_evasion.py
python experiments/visualize_evasion.py
python experiments/visualize_coherence.py
```

The stretch scripts write JSON summaries and PNG figures under `results/`.
They use the four existing corpus categories, so the default topic count is
four. Main parameters are exposed as Python function arguments for
reproducible reruns.

---

## 6. Experimental Results Summary

### Reproducibility revision

The current protocol removes BITS navigation boilerplate before tokenization,
uses deterministic five-restart PLSA with a 1000-iteration ceiling and
`tol=1e-6`, protects all real training documents during poison deduplication,
and measures every model's fit time and peak memory at runtime. The attack
comparison must be interpreted against the random-control condition and the
five-seed confidence intervals; a poisoned score is not evidence of attack
potency unless it separates from that control.

### Self-Collected Corpus (Mean ± Std & 95% Bootstrap CI over 5 Seeds)

| Method / Condition | Classification Macro-F1 [95% CI] | Purity | Retrieval MAP | Held-out Perplexity |
|---|:---:|:---:|:---:|:---:|
| **Condition A (Clean Baseline)** | **0.849** [0.681, 0.949] | 0.890 ± 0.097 | 0.782 ± 0.074 | 743.5 ± 21.5 |
| **Condition B (Poison - Repetition)** | 0.927 [0.894, 0.969] | 0.930 ± 0.040 | 0.835 ± 0.060 | 734.7 ± 38.0 |
| **Condition B (Poison - Bridging)** | 0.839 [0.746, 0.909] | 0.860 ± 0.066 | 0.765 ± 0.043 | 759.8 ± 16.0 |
| **Condition C (Random Control)** | 0.939 [0.908, 0.970] | 0.940 ± 0.037 | 0.838 ± 0.051 | 737.2 ± 32.0 |
| **Defense: Deduplication (on B)** | 0.856 [0.780, 0.931] | 0.860 ± 0.086 | 0.800 ± 0.049 | 735.9 ± 37.7 |
| **Defense: Tempered EM (on B)** | **0.949** [0.917, 0.980] | **0.950 ± 0.032** | **0.919 ± 0.019** | **714.7 ± 27.6** |
| *Baseline: LSI* | 0.644 [0.628, 0.659] | 0.720 ± 0.024 | **0.906 ± 0.024** | N/A |
| *Baseline: TF-IDF + KMeans* | 0.816 [0.674, 0.950] | 0.830 ± 0.144 | 0.782 ± 0.015 | N/A |

![Condition Comparison Plot](results/condition_comparison.png)

---

## 7. Key References

1. **Hofmann, T. (1999).** *Probabilistic Latent Semantic Analysis.* UAI 1999. [arXiv:1301.6705](https://arxiv.org/abs/1301.6705).
2. **Hofmann, T. (1999).** *Probabilistic Latent Semantic Indexing.* SIGIR 1999.
3. **Hofmann, T. (2001).** *Unsupervised Learning by Probabilistic Latent Semantic Analysis.* Machine Learning 42, 177–196.
4. **Mei, S., & Zhu, X. (2015).** *The Security of Latent Dirichlet Allocation.* AISTATS 2015.
5. **Zhou, Y., et al. (2021).** *EvaLDA: Efficient Evasion Attacks Towards Latent Dirichlet Allocation.* AAAI 2021.

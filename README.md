# Topic Modeling Beyond Keywords: Probabilistic Latent Semantic Analysis

**Replication and Adversarial Stress Testing**

**Course:** Information Retrieval (CS F469 / Category 1: Classic Paper Replication and Stress Testing)  
**Group 1:** Soham Vinay Deshmukh, Anish Sai Nimmagadda, Romit Jain, Goutham Mahesh  
**Repository:** https://github.com/ItsSoham23/IR-PROJECT

---

## 1. Project Overview

This project implements **Probabilistic Latent Semantic Analysis (PLSA)** from scratch using the Expectation-Maximization (EM) algorithm, following Hofmann's formulation. We evaluate PLSA as both:

- a latent topic model for document classification, and
- a document representation for information retrieval.

We then stress-test the learned latent space using controlled **document-poisoning attacks**, compare the effect against a matched random-control condition, and evaluate simple defenses.

The final study uses a frozen, balanced self-collected corpus with four categories:

- **Campus** — BITS Pilani BITScan
- **Politics** — English Wikinews
- **Sports** — English Wikinews
- **Technology** — English Wikinews

### Main research question

> Can a classic latent semantic model remain useful for retrieval while resisting simple corpus-level manipulation?

---

## 2. Final Experimental Design

### Corpus

The source registry contains **320 candidate documents**: 80 candidates per category.

After automated source and extraction validation:

- **304 candidates are valid**
- **16 Campus candidates are rejected**
  - 14 require OCR / have unusable text layers
  - 2 return HTTP 404
- **256 documents are selected for modeling**
- **64 documents per category**
- **204 training documents**
- **52 held-out test documents**
- per category: **51 train / 13 test**

For categories with more than 64 valid candidates, selection is deterministic:

1. sort by extracted word count,
2. break ties by document ID,
3. select the centered 64-document subset.

This selection happens **before modeling** and does not depend on experiment seed or model performance.

### Five-seed evaluation

Main results are averaged over:

`42, 101, 202, 303, 404`

The selected 64 documents per category remain fixed across seeds; only the 51/13 train-test partition changes.

### PLSA configuration

- Topics: `K = 4`
- Maximum EM iterations: `1000`
- Convergence tolerance: `1e-6`
- Random restarts: `5`
- Best-likelihood restart retained
- Standard PLSA: `beta = 1.0`
- Tempered EM defense: `beta = 0.85`

The 1000-iteration value is a **ceiling**, not a forced iteration count. Training stops early when the relative log-likelihood change falls below the tolerance.

---

## 3. End-to-End IR Pipeline

```text
Candidate sources
      |
      v
Source validation
(URL / PDF / text quality)
      |
      v
Source-aware cleaning
(BITScan / Wikinews)
      |
      v
Deterministic balanced selection
64 documents/category
      |
      v
Seeded train/test split
51 train + 13 test/category
      |
      v
Train-only vocabulary + count matrix
      |
      v
PLSA / baselines / adversarial conditions
      |
      v
Fold-in held-out documents
      |
      v
Topic-to-class alignment
(Hungarian matching)
      |
      v
Classification + retrieval + robustness evaluation
```

Rejected sources never enter the model. Test documents do not contribute to vocabulary construction.

---

## 4. Source-Aware Cleaning and Representation

### BITScan / Campus

BITScan magazines contain repeated publication furniture that can falsely dominate topic models. The pipeline therefore:

- removes repeated page-edge headers and footers,
- normalizes changing page numbers before repeated-layout detection,
- removes page numbers, TOC dot-leader lines and explicit contact labels,
- preserves substantive prose even when it mentions BITS/Pilani/Campus,
- rejects PDFs with unusable text layers instead of silently accepting them,
- keeps the full cleaned extraction in SQLite for auditability,
- uses the first substantial non-front-matter page as the model representation.

### Wikinews

The Wikinews cleaner removes observed non-article material such as:

- leading publication-date lines,
- Sources / References,
- Related news / Related stories,
- External links / Sister links / See also,
- the standard opinion invitation,
- HTML related-story widgets.

Ordinary article prose, lists and interview content are preserved.

### Shared document length

Campus magazines are much longer than Wikinews articles. To avoid document-length domination, every model document is normalized to a shared training-derived budget.

For seed 42:

- **word cap: 432**
- derived from the 90th percentile of cleaned **non-Campus training documents**

### Vocabulary

For seed 42:

- **3,457 terms**
- `min_df = 2`
- `max_df_ratio = 0.85`
- fit exclusively on training text

---

## 5. Models, Attacks and Defenses

### Clean baseline

**Condition A — Standard PLSA**

PLSA models:

`P(w|d) = sum_z P(z|d) P(w|z)`

Training uses EM. Held-out documents are evaluated by **folding in** test documents while keeping `P(w|z)` fixed.

### Poisoning conditions

**Condition B1 — Keyword repetition**

Adds 10 synthetic poison documents dominated by high-probability words from the target topic.

**Condition B2 — Bridging attack**

Adds 10 synthetic poison documents mixing words from:

- target category: **Campus**
- confuser category: **Politics**

### Random control

**Condition C**

Adds 10 random-word documents sampled from the training vocabulary. This controls for the effect of simply enlarging the corpus.

### Defenses

**Cosine deduplication**

Filters suspicious near-duplicate poison documents while protecting all real training documents.

**Tempered EM**

Uses `beta = 0.85` in the E-step to soften posterior concentration.

### Additional baselines / comparisons

- **LSI** — TF-IDF + TruncatedSVD + KMeans
- **TF-IDF + KMeans**
- **Gensim LDA**
- poison-budget sensitivity utilities
- test-time evasion utilities
- NPMI coherence analysis
- likelihood-trimming utility

---

## 6. Evaluation Metrics

### Classification / clustering

- Macro-F1
- Accuracy
- Purity
- NMI

### Information retrieval

Held-out test documents are queries. Training documents with the same category are treated as relevant.

- MAP
- Precision@5
- Recall@K / NDCG utilities

Ranking uses cosine similarity between model representations.

### Topic quality

- NPMI coherence

### Generative quality

- held-out perplexity

### Attack-potency diagnostics

The project additionally measures:

- probability mass assigned to target words in the learned target topic,
- change in non-target test-document probability toward the target category,
- common-corpus log-likelihood change per token.

Topic alignment is **condition-specific**, so semantic categories are compared rather than raw topic numbers.

For likelihood comparison, both models fold in the same clean evaluation count matrix:

`L(X; M) = sum_dw X_dw log(sum_k theta_M(d,k) phi_M(k,w)) / sum_dw X_dw`

`delta_eval_ll_per_token = L(X; condition) - L(X; baseline)`

Negative values indicate worse fit to the shared evaluation corpus.

---

## 7. Final Five-Seed Results

### Main PLSA and baseline comparison

Mean ± standard deviation over seeds 42, 101, 202, 303 and 404:

| Method / Condition | Macro-F1 | Accuracy | Purity | MAP | Perplexity |
|---|---:|---:|---:|---:|---:|
| **PLSA: Clean (A)** | **0.870 ± 0.041** | **0.877 ± 0.036** | **0.877 ± 0.036** | **0.839 ± 0.039** | 1081.7 ± 38.2 |
| PLSA: Poison Repetition | 0.860 ± 0.049 | 0.865 ± 0.046 | 0.865 ± 0.046 | 0.799 ± 0.055 | 1075.5 ± 37.7 |
| PLSA: Poison Bridging | 0.870 ± 0.048 | 0.877 ± 0.045 | 0.877 ± 0.045 | 0.817 ± 0.051 | 1079.3 ± 36.6 |
| PLSA: Random Control | 0.876 ± 0.055 | 0.881 ± 0.052 | 0.881 ± 0.052 | 0.814 ± 0.067 | 1081.6 ± 38.8 |
| Defense: Deduplication | 0.871 ± 0.050 | 0.877 ± 0.046 | 0.877 ± 0.046 | 0.832 ± 0.034 | 1076.2 ± 42.8 |
| **Defense: Tempered EM** | **0.891 ± 0.049** | **0.896 ± 0.045** | **0.896 ± 0.045** | **0.854 ± 0.030** | **1073.9 ± 40.1** |
| Baseline: LSI | 0.675 ± 0.037 | 0.688 ± 0.031 | 0.688 ± 0.031 | 0.781 ± 0.049 | N/A |
| Baseline: TF-IDF + KMeans | 0.838 ± 0.051 | 0.842 ± 0.048 | 0.842 ± 0.048 | 0.736 ± 0.016 | N/A |

### Main findings

- Clean PLSA outperformed LSI and TF-IDF + KMeans on average for classification.
- Clean PLSA also achieved higher MAP than the two main retrieval baselines.
- At the tested poison budget, **classification degradation was limited**.
- Keyword repetition reduced MAP from **0.839 to 0.799**, a larger effect than its change in Macro-F1.
- Bridging poisoning caused only modest average degradation.
- The random control did not produce a clear targeted attack effect.
- Deduplication restored performance close to the clean baseline.
- Tempered EM achieved the strongest average Macro-F1 and MAP among the PLSA conditions.

These results do **not** support a claim that the tested poisoning attacks catastrophically break PLSA. Instead, the final experiment suggests that classification is fairly robust at this poison budget, while retrieval is somewhat more sensitive.

---

## 8. Attack-Potency Diagnostics

Five-seed means:

| Condition | Target-word mass in learned target topic | Test drift to target category | Δ log-likelihood / token |
|---|---:|---:|---:|
| Repetition attack | 0.2653 ± 0.0043 | -0.0075 ± 0.0014 | -0.010766 ± 0.007415 |
| Bridging attack | 0.2289 ± 0.0051 | -0.0033 ± 0.0027 | -0.009390 ± 0.010186 |
| Random control | 0.1869 ± 0.0121 | +0.0050 ± 0.0160 | -0.011609 ± 0.009212 |

The attacks increase target-word probability mass, but they do **not** produce consistent positive movement of non-target held-out documents toward the Campus target category. This is why the final conclusions describe the attack effect as modest rather than catastrophic.

---

## 9. PLSA vs Gensim LDA

Five-seed head-to-head comparison:

| Condition | PLSA F1 | LDA F1 | PLSA MAP | LDA MAP |
|---|---:|---:|---:|---:|
| Clean | **0.870 ± 0.041** | 0.611 ± 0.073 | **0.839 ± 0.039** | 0.593 ± 0.028 |
| Repetition poison | **0.860 ± 0.049** | 0.655 ± 0.100 | **0.799 ± 0.055** | 0.631 ± 0.060 |
| Bridging poison | **0.870 ± 0.048** | 0.614 ± 0.072 | **0.817 ± 0.051** | 0.605 ± 0.037 |
| Random control | **0.876 ± 0.055** | 0.605 ± 0.104 | **0.814 ± 0.067** | 0.599 ± 0.036 |

PLSA produced higher Macro-F1 and MAP than Gensim LDA under all four conditions in this experiment.

**Perplexity caveat:** PLSA and Gensim LDA use different perplexity / likelihood estimation protocols, so their raw perplexity values should not be treated as a direct head-to-head measure.

---

## 10. Recovered Topics

Seed-42 clean PLSA recovered interpretable category-aligned topics.

| Aligned category | Example high-probability words |
|---|---|
| Sports | match, wickets, australia, south, world, england, africa |
| Politics | said, people, new, government, billion, house, year, states |
| Campus | bits, prof, pilani, students, campus, india, research, faculty |
| Technology | apple, new, software, taipei, taiwan, iphone, show, wimax |

Campus retains institution-specific vocabulary because words such as BITS, Pilani and campus also occur naturally in substantive BITScan content. Source-specific boilerplate is removed, but meaningful institution references are intentionally preserved.

---

## 11. Computational Cost and Convergence

| Method / Condition | Fit Time (s) | Peak Memory (MB) | Mean EM Iterations | Status |
|---|---:|---:|---:|---|
| PLSA Clean | 51.71 ± 11.07 | 71.04 ± 0.75 | 454.4 | 100% converged |
| PLSA Poison (Repetition) | 57.47 ± 15.53 | 74.49 ± 0.79 | 406.2 | 100% converged |
| PLSA Poison (Bridging) | 57.48 ± 12.18 | 74.52 ± 0.76 | 288.4 | 100% converged |
| PLSA Random Control | 56.32 ± 10.45 | 74.52 ± 0.76 | 368.8 | 100% converged |
| Deduplication Defense | 46.13 ± 4.01 | 71.63 ± 0.80 | 259.4 | 100% converged |
| Tempered EM (`beta = 0.85`) | 53.86 ± 3.13 | 97.21 ± 1.00 | 164.8 | 100% converged |
| LSI Baseline | 0.07 ± 0.03 | 6.04 ± 0.06 | N/A | non-EM baseline |
| TF-IDF + KMeans | 0.10 ± 0.01 | 6.04 ± 0.07 | N/A | non-EM baseline |

The final 1000-iteration ceiling is justified by the observed convergence behavior: the selected clean PLSA runs required roughly 454 EM iterations on average, while all final PLSA conditions converged before the ceiling.

---

## 12. Repository Structure

```text
IR-PROJECT/
├── README.md
├── requirements.txt
├── requirements-lock.txt
├── demo.py
├── data/
│   ├── metadata.csv
│   ├── processed/
│   │   └── splits.json
│   └── raw/                    # generated locally / ignored
├── src/
│   ├── ingest_corpus.py
│   ├── source_extract.py
│   ├── corpus_policy.py
│   ├── dataset.py
│   ├── preprocess.py
│   ├── plsa.py
│   ├── fold_in.py
│   ├── align.py
│   ├── attack.py
│   ├── defense.py
│   ├── evasion.py
│   ├── baselines.py
│   ├── retrieval.py
│   ├── metrics.py
│   └── coherence.py
├── experiments/
│   ├── stage0_20news.py
│   ├── run_self_collected.py
│   ├── lda_baseline.py
│   ├── poison_budget.py
│   └── test_time_evasion.py
├── tests/
├── reports/
│   ├── dataset_source_audit.csv
│   ├── dataset_update_audit.md
│   └── verification_environment.json
└── results/
    └── ...                     # experiment outputs
```

Generated raw text and SQLite files are intentionally ignored. They are rebuilt from the reviewed metadata registry.

---

## 13. Installation

Clone the repository:

```bash
git clone https://github.com/ItsSoham23/IR-PROJECT.git
cd IR-PROJECT
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.\.venv\Scripts\Activate.ps1
```

Activate it on Linux/macOS:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

For the exact tested environment:

```bash
pip install -r requirements-lock.txt
```

The verification environment is recorded in:

`reports/verification_environment.json`

---

## 14. Reproducing the Final Corpus

Using the reviewed metadata snapshot:

```bash
python -m src.ingest_corpus --metadata data/metadata.csv --audit-path reports/dataset_source_audit.csv
python -m src.dataset
```

The dataset preparation step creates the balanced 256-document corpus and writes the split manifest to:

`data/processed/splits.json`

To force a full source refresh:

```bash
python -m src.ingest_corpus --force
```

---

## 15. Running the Final Experiments

### Main five-seed experiment

```bash
python experiments/run_self_collected.py
```

This runs:

- clean PLSA,
- repetition poisoning,
- bridging poisoning,
- random control,
- deduplication defense,
- tempered EM,
- LSI,
- TF-IDF + KMeans,

over the five final seeds.

### PLSA vs LDA

```bash
python experiments/lda_baseline.py
```

### Optional stretch experiments

```bash
python experiments/poison_budget.py
python experiments/test_time_evasion.py
```

Experiment outputs are written under:

`results/dataset_320/`

The directory name is retained for compatibility and refers to the **320-candidate source registry**. The modeling corpus itself contains **256 selected documents**.

### Short offline demo

```bash
python demo.py
```

The demo intentionally uses a much smaller 80-document subset, one seed, one restart and a shorter iteration ceiling for speed. Its numbers are **not** the final five-seed results.

---

## 16. Verification

Run the test suite:

```bash
python -m pytest -v tests/
```

Final local verification:

**89 tests passed**

The suite includes checks for:

- EM log-likelihood behavior,
- probability normalization,
- synthetic topic recovery,
- held-out folding-in,
- source routing and extraction,
- source-specific cleaning,
- stale-cache rejection,
- candidate-pool readiness,
- deterministic 64-document selection,
- seed-independent corpus membership,
- train/test separation,
- attack-potency topic alignment,
- common-corpus likelihood comparison.

---

## 17. Limitations

- The final corpus contains only 256 modeling documents.
- BITScan magazines and Wikinews articles have different source structures.
- Campus page selection remains heuristic.
- Multi-column PDF extraction can still produce imperfect reading order.
- Fourteen Campus PDFs require OCR and two candidate URLs return 404.
- Centered-length selection intentionally excludes length extremes from the oversized Wikinews pools.
- Five seeds improve reliability but still provide limited statistical power.
- The poisoning documents are controlled synthetic attacks, not a complete real-world adversary model.
- PLSA/LDA perplexity values are not directly comparable because the models use different likelihood-estimation procedures.

---

## 18. Future Work

Potential extensions include:

- OCR support for scanned BITScan issues,
- a larger corpus,
- more seeds / repeated resampling,
- broader poison budgets and target categories,
- stronger IR baselines such as BM25 and query likelihood,
- neural embedding retrieval,
- more realistic adversarial document generation,
- comparison against modern neural topic models.

---

## 19. References

1. **Hofmann, T. (1999).** *Probabilistic Latent Semantic Analysis.* UAI 1999.
2. **Hofmann, T. (1999).** *Probabilistic Latent Semantic Indexing.* SIGIR 1999.
3. **Hofmann, T. (2001).** *Unsupervised Learning by Probabilistic Latent Semantic Analysis.* Machine Learning, 42, 177-196.
4. **Mei, S., & Zhu, X. (2015).** *The Security of Latent Dirichlet Allocation.* AISTATS 2015.
5. **Zhou, Y., et al. (2021).** *EvaLDA: Efficient Evasion Attacks Towards Latent Dirichlet Allocation.* AAAI 2021.

### Dataset sources

- **BITS Pilani BITScan** — Campus corpus candidates
- **English Wikinews** — Politics, Sports and Technology corpus candidates

---

## Final Summary

The project successfully reproduces PLSA from scratch and evaluates it as both a topic model and an IR representation on a cleaned, auditable, balanced corpus.

The final five-seed study shows that:

- PLSA learns interpretable category-aligned topics,
- clean PLSA outperforms LSI and TF-IDF + KMeans in the final comparison,
- PLSA also outperforms the tested Gensim LDA configuration on Macro-F1 and MAP,
- the tested poisoning attacks cause only modest average degradation at the chosen budget,
- retrieval is somewhat more sensitive than classification,
- deduplication largely restores clean performance,
- tempered EM produces the strongest average PLSA classification and retrieval scores,
- and the complete pipeline is versioned, auditable and reproducible from source ingestion through evaluation.

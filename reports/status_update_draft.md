# Intermediate Status Update: PLSA Replication and Adversarial Stress Testing

**Course:** Information Retrieval (CS F469), Category 1: Classic Paper Replication and Stress Testing
**Group No.:** 1
**Project title:** Topic Modeling Beyond Keywords: Probabilistic Latent Semantic Analysis. Replication and Adversarial Stress Testing
**Members:**
- Soham Vinay Deshmukh (2023A7PS0025P)
- Anish Sai Nimmagadda (2023A7PS0027P)
- Romit Jain (2023A7PS0021)
- Goutham Mahesh (2023A7PS0674P)

**GitHub repository:** https://github.com/ItsSoham23/IR-PROJECT (README with setup and run instructions included). This report describes commit `8c42d72` ("Complete PLSA stretch experiments", 23 Oct 2026).

---

## 1. Implementation Status and Architecture

Overall: the full pipeline and most stretch experiments are built (about 80% of planned work). The remaining work is mainly **correctness fixes and statistical strength**, because a code review found issues that make several current numbers provisional (Section 5). Status labels: **Done**, **Done, needs fix**, **Partial**, **Pending**.

| Module | Status | Details |
|---|:---:|---|
| Data collection (`src/ingest_corpus.py`) | Done, needs fix | 120 candidate documents, 30 per category (Campus from the BITS Pilani portal; Politics, Sports, Technology from Wikinews). Duplicate and cleaning status in `metadata.csv` are still "pending" for all 120. Campus pages still contain site navigation text. |
| Preprocessing (`src/preprocess.py`) | Partial | Lowercasing, tokenization, stopwords, min-df 2, max-df 0.85, vocabulary from the training split only. **Lemmatization/stemming and the OOV token from the proposal are not implemented.** |
| Dataset and split (`src/dataset.py`) | Done | 25 documents per category chosen around median length, stratified 80/20 split. The seed changes both the split and the vocabulary. Now also exports `count_matrix` and `cleaned_corpus.json`. |
| PLSA core EM (`src/plsa.py`) | Done, needs fix | From-scratch NumPy EM (asymmetric and symmetric, dense and sparse, restarts, perplexity). Smoothing is a 1e-12 additive constant. **All experiments use `max_iter=50`, which never converges.** |
| Tempered EM | Partial | Fixed beta = 0.85 from the first iteration; not annealed, and beta is not tuned on validation data. |
| Folding-in (`PLSA.transform`, `src/fold_in.py`) | Done | Freezes P(w\|z), re-estimates P(z\|d_new) only. Text-level helper added for evasion tests. |
| Hungarian alignment (`src/align.py`) | Done | Fit on real training documents only (poison/control excluded). |
| Baselines (`src/baselines.py`, `experiments/lda_baseline.py`) | Done | LSI (TF-IDF, SVD, k-means), TF-IDF + k-means, and now **Gensim LDA**. BM25 is mentioned in a docstring but not implemented. |
| Attacks (`src/attack.py`, `src/evasion.py`) | Done | Keyword-repetition and bridging poisoning, random-word control, and a test-time token-swap evasion. These are heuristics inspired by Mei and Zhu (2015) and EvaLDA, not their optimization attacks. |
| Defenses (`src/defense.py`) | Partial | Deduplication and tempered EM are run. Likelihood trimming is implemented but never used; MAP/Dirichlet smoothing is not implemented. |
| Retrieval and coherence (`src/retrieval.py`, `src/coherence.py`) | Done, needs fix | P@K, R@K, MAP, NDCG@K, NPMI coherence. Ranking is cosine over 4-dimensional topic mixtures. Time and memory columns are partly invalid (Section 5). |
| Stretch experiments (`lda_baseline.py`, `poison_budget.py`, `test_time_evasion.py`) | Done, needs more seeds | LDA comparison uses 5 seeds; the budget sweep and evasion test use **one seed (42)** only. |
| Tests (`tests/`) | Done | 7 pytest tests pass (monotone log-likelihood at beta = 1, normalization, synthetic recovery, folding-in). |

### 1.2 Architecture (changes from proposal marked *new*)

```
120 collected docs -> select 25/category -> stratified split: 80 train / 20 test
        |
 vocabulary (train only) -> count matrices
        |
 Cond A: 80 clean | Cond B: +10 poison (repetition | bridging) | Cond C: +10 random-word docs
 *new* poison-budget sweep: +2, 5, 10, 15, 20 poison documents
        |
 PLSA EM (beta 1.0) | tempered EM (beta 0.85) | dedup filter | *new* Gensim LDA
        |
 frozen P(w|z) -> folding-in of the 20 test docs  (*new* test-time token-swap evasion)
        |                                |
 dominant topic -> Hungarian alignment   cosine ranking of topic mixtures
 -> Macro-P/R/F1, Purity, NMI,           -> MAP, P@K, NDCG   *new*: NPMI coherence
    perplexity
```

---

## 2. IR Pipeline Details

1. **Representation.** Alphabetic tokens of 3 or more characters, lowercased, minus a built-in stopword list. Vocabulary from the 80 training documents only (about 1,700 terms, varying by seed), min-df 2, max-df 0.85. Unseen words are dropped. No stemming or lemmatization yet.
2. **Model.** Asymmetric aspect model with K = 4: `P(w|d) = sum_z P(z|d) P(w|z)`.
   - E-step: `P(z|d,w)` is proportional to `[P(z|d) P(w|z)]^beta`, normalized over z (beta = 1 is standard EM).
   - M-step: `P(w|z)` and `P(z|d)` proportional to the expected counts `n(d,w) P(z|d,w)`, plus a small constant.
   - Uniform(0.1, 1) random initialization; stop at relative log-likelihood change below 1e-5 or `max_iter`.
3. **Folding-in.** P(w|z) frozen; only P(z|d_new) is iterated (up to 50 iterations).
4. **Classification.** A test document's dominant topic is mapped to a category by the Hungarian alignment; macro-P/R/F1 are computed against the true label.
5. **Retrieval.** Each test document queries the training documents; cosine over topic mixtures ranks them. Same-category documents count as relevant (a coarse proxy, not human judgments).
6. **Seeds.** Five seeds (42, 101, 202, 303, 404). Each seed fixes the split, vocabulary and EM initialization, shared across conditions, so comparisons are paired within a seed.

---

## 3. Preliminary Evaluation Results

### 3.1 Self-collected corpus (5 seeds; 95% bootstrap CI for F1)

Numbers logged by our committed run (`results/self_collected_summary.json`). See Section 5, issue 6: we could not reproduce them exactly on a fresh clone.

| Condition | Macro-F1 [95% CI] | Purity | MAP | Perplexity |
|---|:---:|:---:|:---:|:---:|
| A: clean PLSA | 0.717 [0.644, 0.799] | 0.770 | 0.663 | 421.8 |
| B: poison, repetition | 0.683 [0.642, 0.740] | 0.740 | 0.663 | 420.7 |
| B: poison, bridging | 0.661 [0.557, 0.769] | 0.750 | 0.658 | 422.6 |
| C: random-word control | 0.650 [0.619, 0.691] | 0.730 | 0.664 | 422.1 |
| Defense: dedup (on B) | 0.791 [0.722, 0.860] | 0.800 | 0.530 | 1518.8 |
| Defense: tempered EM (on B) | 0.642 [0.562, 0.746] | 0.760 | 0.723 | 416.5 |
| Baseline: LSI | 0.644 [0.628, 0.659] | 0.720 | 0.906 | n/a |
| Baseline: TF-IDF + k-means | 0.816 [0.674, 0.950] | 0.830 | 0.782 | n/a |

- PLSA beats LSI on F1 but loses to TF-IDF + k-means, and LSI wins on MAP. The proposal's criterion (PLSA improving on a keyword or LSI baseline) is only partly met.
- Poisoning lowers F1 by 0.03 to 0.06, but the random-word control (0.650) scores below both attacks, so the attack does not clearly beat plain noise.
- Tempered EM improves MAP and perplexity but not F1.
- The deduplication row is **not a valid defense result** (issue 3).

### 3.2 New: PLSA vs Gensim LDA (same splits and poison documents, 5 seeds)

| Condition | PLSA F1 | LDA F1 | PLSA MAP | LDA MAP | PLSA NPMI | LDA NPMI |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| A: clean | 0.717 | 0.670 | 0.663 | 0.643 | 0.484 | 0.420 |
| B: repetition | 0.683 | 0.584 | 0.663 | 0.637 | 0.413 | 0.437 |
| B: bridging | 0.661 | 0.555 | 0.658 | 0.632 | 0.442 | 0.423 |
| C: random control | 0.650 | 0.603 | 0.664 | 0.633 | 0.444 | 0.431 |

LDA F1 falls more under poisoning than PLSA F1 (about 0.09 to 0.12 vs 0.03 to 0.06), so there is no evidence that the Dirichlet prior protects against this attack. The standard deviations are 0.04 to 0.15, so the gaps are within noise. LDA perplexity (about 191 to 196) is **not comparable** with PLSA perplexity (about 421 to 423): LDA uses Gensim's variational bound and PLSA uses folding-in on the same words, so we do not compare them.

### 3.3 New: poison-budget sweep (seed 42 only)

| Poison documents | 2 | 5 | 10 | 15 | 20 |
|---|:---:|:---:|:---:|:---:|:---:|
| PLSA Macro-F1 | 0.847 | 0.848 | 0.651 | 0.800 | 0.539 |
| Tempered PLSA Macro-F1 | 0.848 | 0.848 | 0.623 | 0.848 | 0.623 |

F1 falls at 10 and 20 documents but recovers at 15, so there is no reliable dose-response curve from one seed. Tempered EM NPMI rises to 0.58 to 0.63 at budgets 10 and 20; a possible reason is that poison documents make one topic very coherent, which means coherence can be inflated by the attack. This needs checking.

### 3.4 New: test-time evasion (seed 42 only)

Replacing 0, 1, 2 or 3 random tokens per test document with high-probability target-topic words left Macro-F1 unchanged at 0.848. This is expected, since 3 random swaps in documents of hundreds of tokens barely move the topic mixture. The test is therefore uninformative, and it needs larger swap budgets and word selection that targets the most influential tokens, as in EvaLDA.

### 3.5 Stage 0: 20 Newsgroups mini-replication (logged earlier, not re-verified)

| Model | Macro-F1 | NMI | Perplexity | MAP |
|---|:---:|:---:|:---:|:---:|
| PLSA, beta 1.0 | 0.435 | 0.245 | 735.2 | 0.388 |
| Tempered PLSA, beta 0.85 | 0.420 | 0.263 | 710.6 | 0.426 |
| LSI | 0.361 | 0.270 | n/a | 0.510 |
| TF-IDF + k-means | 0.603 | 0.358 | n/a | 0.412 |

Consistent with Hofmann, tempering lowers held-out perplexity and PLSA beats LSI on F1; TF-IDF + k-means beats PLSA on F1 and LSI beats PLSA on MAP.

### 3.6 Memory, time and potency

Only the fit time and memory of Condition A (about 0.44 s, 13.9 MB) and B-repetition (about 0.48 s, 15.6 MB) were measured correctly; all other rows are copies or hard-coded constants and are excluded until re-measured. The three attack-potency quantities are implemented, but the first numbers are **withdrawn** (issue 4) and will be re-reported.

---

## 4. Findings so far

1. From-scratch PLSA trains and passes its tests, but it does not yet beat the simple baselines.
2. **Initialization variance is much larger than any attack effect.** On a fixed split, fully converged EM with different random initializations gave Macro-F1 anywhere from about 0.64 to 0.95 (for example 0.639 to 0.949 on split 101, and 0.667 to 1.000 on split 404). The poisoning effects we measure (0.03 to 0.06) are far smaller, and our setup runs only one initialization per split, so these effects cannot be separated from noise yet.
3. The Campus topic is partly driven by website boilerplate; the Politics and Sports topics include generic Wikinews words such as "sources" and "said".
4. Tempered EM helps perplexity and retrieval, matching Hofmann qualitatively, but not classification.

---

## 5. Known Issues Found in Code Review (being fixed before final results)

| # | Issue | Evidence | Fix |
|---|---|---|---|
| 1 | EM not converged | `converged_` was False for all 5 seeds at `max_iter=50`; convergence needed 127 to 323 iterations; F1 on split 101 moved from 0.854 to 0.949. The new stretch scripts also use 50. | `max_iter` of 500 or more; report iterations. |
| 2 | Invalid time/memory | Times and memory after B-repetition are copies of that row; LSI and TF-IDF figures (0.05 s, 0.04 s, 0.5 MB) are hard-coded. | Benchmark every model the same way. |
| 3 | Dedup removes real data | At threshold 0.75 on raw counts it dropped about 20 of 80 clean training documents per seed (in the two seeds checked, all or nearly all Campus pages sharing boilerplate), plus 8 to 9 of 10 poison documents. This explains the perplexity of about 1,500 and the MAP drop. | Clean boilerplate first, tune the threshold on training data, report clean documents dropped. |
| 4 | Potency topic index | The clean model's target-topic index is reused for poisoned models; in 2 of 5 seeds the poisoned model's target topic has a different index. | Align each poisoned model with its own alignment. |
| 5 | Boilerplate in corpus | Campus navigation text, Wikinews date lines and "Sources" sections inflate separability. | Add cleaning rules; re-run duplicate and quality checks. |
| 6 | Not reproducible | On a fresh clone (same seeds, same code), the seed-42 split and vocabulary (1,743 terms) and the LSI numbers matched ours exactly, but PLSA results differed (Condition A F1 0.825 vs 0.717) and B vs C reversed (B 0.692, C 0.754). The cause is not yet identified; adding 1e-9 noise to the counts did not change results. | Pin library versions, fix issue 1, and report the spread across initializations. |
| 7 | Single-seed stretch results | Budget sweep and evasion use seed 42 only; the main table has one initialization per split. | Repeat across all splits with several initializations per split; use best-likelihood restarts. |
| 8 | Weak evasion test and poison heuristic | 0 to 3 random swaps cannot move a document; poison documents of repeated top words are easy to detect. | Larger budgets, targeted word selection, adaptive poison. |
| 9 | Ranking and perplexity protocol | Cosine over 4 topic proportions is a weak ranking function; PLSA perplexity folds in and scores the same words; LDA perplexity uses a different protocol. | Query-likelihood or Hellinger ranking, BM25 baseline, fold-in on half of each document. |

---

## 6. Path to Final Submission (remaining about 20%)

**Phase 1: correctness.** Fix issues 1 to 5, clean the corpus (boilerplate, lemmatization, OOV token), freeze a corpus snapshot, and re-run everything.

**Phase 2: statistical strength.**
- Repeated splits (20 or more) with several EM initializations per split, taking the best-likelihood restart.
- Paired bootstrap tests for B vs C and B vs A.
- Tune tempered-EM beta on a validation split taken from training data.

**Phase 3: stretch objectives.**

| Stretch objective | Status |
|---|---|
| LDA comparison with Gensim | Done on 5 seeds; needs a comparable perplexity protocol |
| Poison-budget sweep | Done on seed 42; needs more seeds and a matched control at each budget |
| NPMI coherence | Done for LDA and budget sweep; not yet in the main condition summary |
| Test-time evasion | Done, but uninformative; redo with larger budgets and targeted swaps |
| Defense prototypes | Dedup and tempered EM run; trimming not run; MAP smoothing missing |
| Scaling beyond 100 documents | Not started |

**Phase 4.** Final report, figures, and a README with exact reproduction commands and pinned requirements.

---


## 7. References

1. Hofmann, T. (1999). Probabilistic Latent Semantic Analysis. UAI 1999. https://arxiv.org/abs/1301.6705
2. Hofmann, T. (1999). Probabilistic Latent Semantic Indexing. SIGIR 1999. https://dl.acm.org/doi/10.5555/2073796.2073829
3. Hofmann, T. (2001). Unsupervised Learning by Probabilistic Latent Semantic Analysis. Machine Learning 42, 177-196. https://doi.org/10.1023/A:1007617005950
4. Mei, S., and Zhu, X. (2015). The Security of Latent Dirichlet Allocation. AISTATS 2015. https://proceedings.mlr.press/v38/mei15.html
5. Zhou, Y., et al. (2021). EvaLDA: Efficient Evasion Attacks Towards Latent Dirichlet Allocation. AAAI 2021. https://arxiv.org/abs/2012.04864
6. Blei, D., Ng, A., and Jordan, M. (2003). Latent Dirichlet Allocation. JMLR 3, 993-1022.

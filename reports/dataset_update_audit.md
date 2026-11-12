# Frozen 256-document corpus audit - 12 November 2026

Branch: **dataset-320-ingestion-update**. Extraction version: **corpus-v2.1**. Ingestion-state schema: **2**. Dataset-manifest schema: **3**.

**320 candidates; 304 valid; 16 rejected; 256 selected; 48 valid but unused.** The completed audit is model-ready. The real seed-42 corpus is prepared with **204 train / 52 test**, **432 words** as the training-derived normalization cap, and **3,457 vocabulary terms**. No full five-seed experiments were run.

## Design rationale and readiness

The Campus candidate pool yielded 64 documents that passed predefined source and extraction-quality checks. To preserve class balance without introducing OCR-derived documents or weakening validation, the final corpus was fixed at 64 documents per category. Categories with more than 64 valid candidates use a predefined deterministic selection rule independent of model results and experimental split seeds.

The candidate registry remains data/metadata.csv. An audit attempts every candidate, revalidating current-version cached text and retrying rejected candidates. This run reused 304 hash-verified corpus-v2.1 extractions and retried all 16 failures against their current URLs. No older extraction version, stale raw file, automatic OCR, or weaker quality rule was used.

The state records audit_finished separately from model_ready, the candidate metadata hashes, accepted IDs, failed IDs, valid counts, minimum per category, required categories, and detailed audit evidence. Readiness requires a finished audit and at least 64 valid sources in each required category. Legacy complete-flag states require explicit ingestion rebuilding. The loader verifies the snapshot, accepted/rejected partition, usable-row membership, URL/metadata/text hashes, extraction version and counts. A pool with 63 valid Campus sources is rejected; 64 or more can proceed. Previously accepted sources that fail a refresh are removed from usable SQLite rows. Failed sources that are repaired must be fetched and validated.

## Candidate and split counts

| Category | Candidates | Valid | Rejected | Selected | Valid unused | Train | Test |
|---|---:|---:|---:|---:|---:|---:|---:|
| Campus | 80 | 64 | 16 | 64 | 0 | 51 | 13 |
| Politics | 80 | 80 | 0 | 64 | 16 | 51 | 13 |
| Sports | 80 | 80 | 0 | 64 | 16 | 51 | 13 |
| Technology | 80 | 80 | 0 | 64 | 16 | 51 | 13 |
| **Total** | **320** | **304** | **16** | **256** | **48** | **204** | **52** |

The [source audit CSV](dataset_source_audit.csv) retains each candidate, URL, extraction outcome, word count, error/warning, hashes and selection_status. Valid-but-unused sources are explicitly distinct from rejected sources. No article text is included.

## Frozen selection and split

Policy **centered_extracted_length_v1** sorts valid candidates within each category by (extracted word count, document ID), then takes the contiguous 64-document subset starting at floor((valid_count - 64) / 2). All 64 valid Campus sources are selected; each 80-source Wikinews pool contributes its centered 64. Categories only define balance; model outputs, test scores and experimental seeds never affect membership. Documents are never swapped based on performance. The split seed only controls the 51/13 partition.

Real preparations with **seeds 42 and 17** produced identical selected IDs, unused IDs and selected-corpus fingerprints, with different train/test membership. Seed 17 used a separate ignored database/output copy; the default database and manifest retain seed 42.

The [seed-42 manifest](../data/processed/splits.json) includes all selected, unused and failed IDs, per-category counts, train/test IDs, selection policy, extraction version, normalization policy, vocabulary size and candidate/source-pool/selected-corpus fingerprints.

Selected-corpus SHA-256: `9ab17d51dc6bcd6d3e814c41a988db373f868c2e7a992611da1b72f010401bb3`.

## Post-cleaning word-count distribution

| Category | Valid sources | Min | Median | p90 | Max |
|---|---:|---:|---:|---:|---:|
| Campus | 64 | 3853 | 8926.5 | 12899.6 | 17862 |
| Politics | 80 | 66 | 310.5 | 683.0 | 1184 |
| Sports | 80 | 51 | 264.0 | 435.4 | 634 |
| Technology | 80 | 65 | 256.5 | 565.1 | 2598 |

These counts cover all valid candidates before selection and model normalization.

## BITScan evidence and changes

Real page-block extractions were inspected from CAMPUS_001, 021, 039 (Pilani), 045 (Hyderabad), and 066, 079 (Goa). Observed furniture now covered by offline regressions:

- Pilani: BITS Pilani, Pilani campus, followed by a changing page number. Combined-line variants such as BITS Pilani, Pilani campus 1 are also covered.
- Pilani: Semester I, 2023-24, BITScan Issue-43; Semester I 2013-14, bitscan, Pilani Campus; Birla Institute of Technology and Science; and - 3 - Semester I, 2004-05.
- Hyderabad: Issue 22 (Jan-June 2025), BITScan, Hyderabad Campus, with page numbers before/after or on a separate line. The combined review example is tested too.
- Goa: semester I notes 2021-22 bitscan, BITS Pilani K K Birla Goa Campus, and Semester I 2015-16 bitscan, K.K. Birla Goa Campus, with changing page numbers.

Only short strings composed of restricted publication-layout vocabulary qualify for repeated-edge matching. Boundary page numbers and punctuation are normalized; issues and years remain in the signature. A signature must recur on at least two pages and occur within the first/last three nonempty lines. Interior one-off labels and substantive sentences survive. In CAMPUS_074, two interior speaker affiliations saying BITS Pilani, Pilani Campus were inspected and deliberately preserved. Paragraph/page boundaries are retained; numbers, dot-leader TOC entries and explicit contact labels are cleaned conservatively.

## Wikinews evidence and changes

Live API and HTML samples were inspected across Sports (SPORTS_001, 039), Technology (TECH_001, 025), and Politics (POL_001, 029), supplemented by heading counts from the corpus. The cleaner removes:

- Leading weekday/month publication-date lines.
- Source/Sources, References, Related news, Related stories, Related, Related wikinews, External link/External links, Sister links, and See also sections, including subsections. It resumes at a peer article heading rather than deleting all later article content.
- The exact opinion invitation: Have an opinion on this story? Share it!
- HTML .infobox.noprint related-story widgets, observed in TECH_001 and POL_001. Article bullet lists and interview answers remain.

Singular Source and External link headings were observed in TECH_008 and TECH_052; Related wikinews appeared in TECH_065, 066 and 069. Both extraction paths use the same text cleaner. No blanket removal of said, sources, Wikinews, licensing-related words or journalistic prose was introduced. Archive/licensing templates were not recurring material in these extracted samples; no speculative text rules were added.

## Normalization after selection

The cap is recomputed from the 90th percentile of cleaned non-Campus **training** text only (153 articles under the new split). Seed 42 yields **432 words**. The earlier 509-word diagnostic belonged to the superseded 80-per-category design and is not used. The vocabulary is fitted exclusively on the 204 normalized training documents, with min_df=2 and max_df=0.85.

Campus retains first substantial non-front-matter page selection, including observed split Goa editorial headings and Pilani editorial continuations. Sentence-bounded prefixes preserve order; full cleaned source text remains in SQLite. Page selection and the actual cap are recorded in the manifest and per-document outputs. No random word sampling or test-derived length statistics are used. Multi-column continuation fragments remain a source-quality limitation.

## Corrected attack potency

Each condition fits its own TopicAligner on clean training labels. Word mass uses that condition's topic for the target class. Test probabilities are independently reordered with align_proportions before target-class drift is computed. Clean-model topic numbers are used only for its own attack seed words.

Both frozen models fold in the identical clean-training matrix X with the same vocabulary. With theta from this common inference protocol and phi the frozen topic-word matrix:

L(X; M) = sum_dw X_dw log(max(sum_k theta_M(d,k) phi_M(k,w), 1e-12)) / sum_dw X_dw

delta_eval_ll_per_token = L(X; condition) - L(X; baseline).

Units are nats/token. Negative values mean worse fit to the common corpus; positive values mean better fit. This is an in-sample diagnostic, not a held-out generalization estimate. Augmented training objectives are never subtracted. Both normalized scores and the difference are saved to results/dataset_320/attack_potency.json. Tests deliberately permute target topics and supply incomparable training objectives to verify the comparison.

## Exact corpus membership

### Campus

**Selected (64):** CAMPUS_001, CAMPUS_002, CAMPUS_003, CAMPUS_004, CAMPUS_005, CAMPUS_006, CAMPUS_007, CAMPUS_008, CAMPUS_010, CAMPUS_011, CAMPUS_013, CAMPUS_014, CAMPUS_015, CAMPUS_016, CAMPUS_017, CAMPUS_018, CAMPUS_019, CAMPUS_020, CAMPUS_021, CAMPUS_022, CAMPUS_023, CAMPUS_024, CAMPUS_025, CAMPUS_027, CAMPUS_028, CAMPUS_033, CAMPUS_039, CAMPUS_041, CAMPUS_043, CAMPUS_044, CAMPUS_045, CAMPUS_046, CAMPUS_047, CAMPUS_048, CAMPUS_049, CAMPUS_050, CAMPUS_051, CAMPUS_052, CAMPUS_053, CAMPUS_054, CAMPUS_055, CAMPUS_056, CAMPUS_057, CAMPUS_058, CAMPUS_059, CAMPUS_060, CAMPUS_061, CAMPUS_063, CAMPUS_065, CAMPUS_066, CAMPUS_067, CAMPUS_068, CAMPUS_069, CAMPUS_070, CAMPUS_071, CAMPUS_072, CAMPUS_073, CAMPUS_074, CAMPUS_075, CAMPUS_076, CAMPUS_077, CAMPUS_078, CAMPUS_079, CAMPUS_080.

**Valid unused (0):** None.

### Politics

**Selected (64):** POL_002, POL_003, POL_004, POL_005, POL_006, POL_007, POL_008, POL_009, POL_011, POL_012, POL_013, POL_014, POL_015, POL_016, POL_017, POL_018, POL_019, POL_020, POL_021, POL_022, POL_023, POL_024, POL_025, POL_026, POL_027, POL_028, POL_030, POL_031, POL_033, POL_034, POL_035, POL_036, POL_038, POL_039, POL_040, POL_041, POL_043, POL_044, POL_045, POL_046, POL_047, POL_048, POL_049, POL_050, POL_051, POL_052, POL_055, POL_057, POL_058, POL_059, POL_060, POL_062, POL_063, POL_064, POL_066, POL_067, POL_069, POL_071, POL_072, POL_074, POL_075, POL_076, POL_079, POL_080.

**Valid unused (16):** POL_001, POL_010, POL_029, POL_032, POL_037, POL_042, POL_053, POL_054, POL_056, POL_061, POL_065, POL_068, POL_070, POL_073, POL_077, POL_078.

### Sports

**Selected (64):** SPORTS_001, SPORTS_002, SPORTS_003, SPORTS_004, SPORTS_005, SPORTS_006, SPORTS_007, SPORTS_008, SPORTS_009, SPORTS_010, SPORTS_011, SPORTS_012, SPORTS_013, SPORTS_014, SPORTS_015, SPORTS_016, SPORTS_018, SPORTS_019, SPORTS_022, SPORTS_023, SPORTS_024, SPORTS_025, SPORTS_027, SPORTS_028, SPORTS_029, SPORTS_030, SPORTS_031, SPORTS_032, SPORTS_033, SPORTS_035, SPORTS_036, SPORTS_037, SPORTS_038, SPORTS_039, SPORTS_040, SPORTS_041, SPORTS_042, SPORTS_043, SPORTS_044, SPORTS_045, SPORTS_046, SPORTS_047, SPORTS_048, SPORTS_049, SPORTS_050, SPORTS_055, SPORTS_056, SPORTS_057, SPORTS_059, SPORTS_061, SPORTS_062, SPORTS_063, SPORTS_064, SPORTS_065, SPORTS_066, SPORTS_067, SPORTS_068, SPORTS_069, SPORTS_071, SPORTS_072, SPORTS_073, SPORTS_074, SPORTS_079, SPORTS_080.

**Valid unused (16):** SPORTS_017, SPORTS_020, SPORTS_021, SPORTS_026, SPORTS_034, SPORTS_051, SPORTS_052, SPORTS_053, SPORTS_054, SPORTS_058, SPORTS_060, SPORTS_070, SPORTS_075, SPORTS_076, SPORTS_077, SPORTS_078.

### Technology

**Selected (64):** TECH_001, TECH_002, TECH_003, TECH_004, TECH_005, TECH_008, TECH_009, TECH_010, TECH_011, TECH_012, TECH_013, TECH_014, TECH_015, TECH_016, TECH_017, TECH_018, TECH_019, TECH_020, TECH_021, TECH_024, TECH_026, TECH_028, TECH_029, TECH_030, TECH_031, TECH_032, TECH_033, TECH_034, TECH_035, TECH_036, TECH_037, TECH_038, TECH_040, TECH_041, TECH_044, TECH_045, TECH_046, TECH_047, TECH_049, TECH_050, TECH_051, TECH_052, TECH_053, TECH_054, TECH_056, TECH_059, TECH_062, TECH_063, TECH_064, TECH_065, TECH_066, TECH_067, TECH_068, TECH_069, TECH_070, TECH_071, TECH_072, TECH_074, TECH_075, TECH_076, TECH_077, TECH_078, TECH_079, TECH_080.

**Valid unused (16):** TECH_006, TECH_007, TECH_022, TECH_023, TECH_025, TECH_027, TECH_039, TECH_042, TECH_043, TECH_048, TECH_055, TECH_057, TECH_058, TECH_060, TECH_061, TECH_073.

## Rejected candidates retained in the audit

All 16 are Campus PDFs. Fourteen have unusable text layers; two return HTTP 404. They are absent from usable SQLite rows, selected IDs and both splits. They do not block the 64-per-category design.

| Document ID | Source type | Reason | URL |
|---|---|---|---|
| CAMPUS_009 | pdf | Probable failed PDF extraction: most pages have no usable text; run OCR | [source](https://www.bits-pilani.ac.in/wp-content/uploads/I-Semester-2019-2020.pdf) |
| CAMPUS_012 | pdf | Probable failed PDF extraction: most pages have no usable text; run OCR | [source](https://www.bits-pilani.ac.in/wp-content/uploads/II-Semester-2018-2019.pdf) |
| CAMPUS_026 | pdf | Probable failed PDF extraction: most pages have no usable text; run OCR | [source](https://www.bits-pilani.ac.in/wp-content/uploads/II-Semester-2011-2012.pdf) |
| CAMPUS_029 | pdf | Probable failed PDF extraction: most pages have no usable text; run OCR | [source](https://www.bits-pilani.ac.in/wp-content/uploads/I-Semester-2009-2010.pdf) |
| CAMPUS_030 | pdf | Probable failed PDF extraction: most pages have no usable text; run OCR | [source](https://www.bits-pilani.ac.in/wp-content/uploads/II-Semester-2009-2010.pdf) |
| CAMPUS_031 | pdf | Probable failed PDF extraction: most pages have no usable text; run OCR | [source](https://www.bits-pilani.ac.in/wp-content/uploads/I-Semester-2008-2009.pdf) |
| CAMPUS_032 | pdf | Probable failed PDF extraction: most pages have no usable text; run OCR | [source](https://www.bits-pilani.ac.in/wp-content/uploads/II-Semester-2008-2009.pdf) |
| CAMPUS_034 | pdf | Probable failed PDF extraction: most pages have no usable text; run OCR | [source](https://www.bits-pilani.ac.in/wp-content/uploads/II-Semester-2007-2008.pdf) |
| CAMPUS_035 | pdf | Probable failed PDF extraction: most pages have no usable text; run OCR | [source](https://www.bits-pilani.ac.in/wp-content/uploads/I-Semester-2006-2007.pdf) |
| CAMPUS_036 | pdf | Probable failed PDF extraction: most pages have no usable text; run OCR | [source](https://www.bits-pilani.ac.in/wp-content/uploads/II-Semester-2006-2007.pdf) |
| CAMPUS_037 | pdf | Probable failed PDF extraction: most pages have no usable text; run OCR | [source](https://www.bits-pilani.ac.in/wp-content/uploads/I-Semester-2005-2006.pdf) |
| CAMPUS_038 | pdf | Probable failed PDF extraction: most pages have no usable text; run OCR | [source](https://www.bits-pilani.ac.in/wp-content/uploads/II-Semester-2005-2006.pdf) |
| CAMPUS_040 | pdf | Probable failed PDF extraction: most pages have no usable text; run OCR | [source](https://www.bits-pilani.ac.in/wp-content/uploads/II-Semester-2004-2005.pdf) |
| CAMPUS_042 | pdf | Probable failed PDF extraction: most pages have no usable text; run OCR | [source](https://www.bits-pilani.ac.in/wp-content/uploads/II-Semester-2003-2004.pdf) |
| CAMPUS_062 | pdf | HTTP Error 404: Not Found | [source](https://www.bits-pilani.ac.in/wp-content/uploads/BITSCAN-Jan15-toDec15.pdf) |
| CAMPUS_064 | pdf | HTTP Error 404: Not Found | [source](https://www.bits-pilani.ac.in/wp-content/uploads/BITSCAN-Jan-13-toDec14.pdf) |

## Verification and files changed

**89 tests passed**, including all previous 78 and 11 new candidate-pool cases. Tests cover 64/63 readiness, exact balanced splits, length/ID selection, seed independence, rejected exclusion, recovered URLs, removed accepted rows, invalid states and legacy migration. The previous source-cleaning, potency, train-only vocabulary and short experiment integrations remain passing. Single-source extraction fixtures now explicitly request a one-source test policy; production readiness is separately tested at 64.

Python compilation and git diff --check passed. Real ingestion and python -m src.dataset completed successfully. No core PLSA, tempered-EM, folding-in, alignment, attack, retrieval or source-extraction algorithms changed in this task.

- Code: new src/corpus_policy.py; src/ingest_corpus.py; src/dataset.py; production descriptions in experiments/run_self_collected.py and experiments/lda_baseline.py.
- Tests: new tests/test_candidate_pool.py; tests/test_dataset.py; tests/test_ingestion.py.
- Documentation/evidence: README.md; this report; reports/dataset_source_audit.csv; data/processed/splits.json.

The existing results/dataset_320 directory name is retained for compatibility and denotes the candidate registry; new result manifests identify the actual 256-document corpus. Historical results stay historical. The 20 Newsgroups pipelines and explicit small demo configuration are unchanged.

## Reproduction and remaining risks

~~~bash
python -m src.ingest_corpus --metadata data/metadata.csv --audit-path reports/dataset_source_audit.csv
python -m src.dataset
~~~

Local raw text and databases remain ignored. The old default database was backed up to data/corpus.pre256.legacy.bak before rebuilding from the verified cache. The registry content has not been changed. This task created no branch, commit, push or merge.

The tested environment remains recorded in [verification_environment.json](verification_environment.json) and [requirements-lock.txt](../requirements-lock.txt): Python 3.13.7, NumPy 2.5.4, SciPy 1.18.1, scikit-learn 1.9.1, Gensim 4.4.0, PyMuPDF 1.28.2. Cross-environment numerical equality is not established.

No remaining code/readiness blocker was found. The rejected candidates remain unavailable; multi-column PDF reading order and coherent-page heuristics still warrant manual corpus review. Centered-length selection intentionally excludes length extremes and should be considered when interpreting generalization. The unchanged ten-document poison budget is now 10/204 (about 4.9%) of clean training size. Candidate or content changes can alter deterministic membership and require re-audit/review; they must not be made based on model performance. Full final experiments remain intentionally unrun pending inspection of this frozen corpus.

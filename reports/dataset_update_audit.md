# Dataset review audit - 12 November 2026

Branch: **dataset-320-ingestion-update**. Extraction version: **corpus-v2.1**.

Live audit: **304 accepted / 16 failed**, from 320 metadata rows. The target remains 80 documents/category, 64/16 train/test per category, and 256/64 overall. The real full corpus is not prepared; final five-seed experiments have not been run.

## Live source audit

The authoritative registry is data/metadata.csv. Current URLs were downloaded using the new cleaner; same-version resumes are marked cached. No bitscan-v1 or intermediate corpus-v2 rows are accepted. [The per-source CSV](dataset_source_audit.csv) contains IDs, URLs, source types, statuses, word counts, warnings, errors, version and metadata/text hashes, with no article text. The detailed local report and raw extractions are under ignored data/rebuild/.

| Category | Accepted | Failed | Min words | Median | p90 | Max |
|---|---:|---:|---:|---:|---:|---:|
| Campus | 64 | 16 | 3853 | 8926.5 | 12899.6 | 17862 |
| Politics | 80 | 0 | 66 | 310.5 | 683.0 | 1184 |
| Sports | 80 | 0 | 51 | 264.0 | 435.4 | 634 |
| Technology | 80 | 0 | 65 | 256.5 | 565.1 | 2598 |

Counts are after source-specific cleaning and before model normalization. Quality limits were not weakened. Google Drive source CAMPUS_048 succeeds.

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

## Normalization after cleaning

The six inspected magazines showed four clear editorial openings (Pilani 001/021 and Goa 066/079), an opening semester overview (Pilani 039), and a cover followed by events (Hyderabad 045). A whole-document prefix was therefore not retained for Campus PDFs.

The deterministic policy selects the first page with at least 50 words that is not recognized front matter: Editorial, from the editor (including adjacent split heading lines), Dear Readers, contents, In Pursuit of Excellence, or the observed Pilani alumni editorial continuation. It then takes a sentence-bounded prefix of that page. Other sources keep the prefix policy. Full editorial/source text remains in SQLite. The rule and cap are recorded in split manifests; model_page is saved per document. CAMPUS_067 and CAMPUS_072 confirmed that Goa column continuations can precede the split editorial heading; regression tests cover this layout. No random word sampling or test-derived length statistic is used.

The training-article p90 remains the budget. The **diagnostic seed-42 value is 509 words**, derived from the 192 non-Campus training IDs that would occur in a complete 80-per-category split. This is not a completed 320-document split or production manifest: preparation correctly remains blocked.

| Reviewed source | Selected PDF page (1-based) | Model words at diagnostic cap |
|---|---:|---:|
| CAMPUS_001 | 3 | 501 |
| CAMPUS_021 | 4 | 384 |
| CAMPUS_039 | 3 | 290 |
| CAMPUS_045 | 2 | 268 |
| CAMPUS_066 | 3 | 500 |
| CAMPUS_079 | 3 | 328 |

Selected samples concern visits, academic events, a symposium, Foundation Day, convocation and technical sessions instead of editorial introductions. All accepted Campus sources were checked for a selectable page and bounded model length. Multi-column PDF reading order can still contain continuation fragments; selection does not guarantee perfect reconstruction of the source layout.

## Corrected attack potency

Each condition fits its own TopicAligner on clean training labels. Word mass uses that condition's topic for the target class. Test probabilities are independently reordered with align_proportions before target-class drift is computed. Clean-model topic numbers are used only for its own attack seed words.

Both frozen models fold in the identical clean-training matrix X with the same vocabulary. With theta from this common inference protocol and phi the frozen topic-word matrix:

L(X; M) = sum_dw X_dw log(max(sum_k theta_M(d,k) phi_M(k,w), 1e-12)) / sum_dw X_dw

delta_eval_ll_per_token = L(X; condition) - L(X; baseline).

Units are nats/token. Negative values mean worse fit to the common corpus; positive values mean better fit. This is an in-sample diagnostic, not a held-out generalization estimate. Augmented training objectives are never subtracted. Both normalized scores and the difference are saved to results/dataset_320/attack_potency.json. Tests deliberately permute target topics and supply incomparable training objectives to verify the comparison.

## Remaining source failures

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
| CAMPUS_062 | pdf | HTTP 404: repair/replace the official URL | [source](https://www.bits-pilani.ac.in/wp-content/uploads/BITSCAN-Jan15-toDec15.pdf) |
| CAMPUS_064 | pdf | HTTP 404: repair/replace the official URL | [source](https://www.bits-pilani.ac.in/wp-content/uploads/BITSCAN-Jan-13-toDec14.pdf) |

## Artifact cleanup and reproducibility

The legacy SQLite database and all 240 generated raw text files were untracked with git rm --cached; local files remain. data/raw/.gitkeep and data/metadata.csv remain tracked. Raw text, SQLite databases and their journal/sidecar files are ignored. Existing legacy local files are not the new corpus. This review-fix task did not create additional commits, push or merge; Git history records earlier work.

The broad requirements.txt was retained. [requirements-lock.txt](../requirements-lock.txt) captures the tested packages; [verification_environment.json](verification_environment.json) records Python/OS and versions. Tested: Python 3.13.7, NumPy 2.5.4, SciPy 1.18.1, scikit-learn 1.9.1, Gensim 4.4.0 and PyMuPDF 1.28.2 on Windows 11. Cross-environment numerical equality is not established.

## Verification and changed files

**78 tests passed**, including all original 43 and short 320-document fixture integration runs through self-collected, LDA, poison-budget and evasion pipelines. Compilation and working-tree/staged diff checks passed. The live incomplete corpus was confirmed to reject preparation.

- Code: src/preprocess.py, src/source_extract.py, src/ingest_corpus.py, src/dataset.py, src/metrics.py, experiments/run_self_collected.py.
- Tests: strengthened tests/test_ingestion.py and tests/test_dataset.py; new tests/test_source_cleaning.py and tests/test_attack_potency.py.
- Other: .gitignore, README.md, requirements-lock.txt, this report, reports/dataset_source_audit.csv, reports/verification_environment.json; untracking data/corpus.sqlite and 240 data/raw/**/*.txt files.

README now describes actual preprocessing (no lemmatization), the two main defenses (trimming is a utility), incomplete local data, the new potency metric, normalization and rebuild path. Old results remain explicitly historical. Searches found no production use of prepare_100_document_corpus. Old counts remain only in historical records, the explicit rebuild-required manifest, and the separate 20 Newsgroups pipeline. Core PLSA EM, tempered EM, folding-in, alignment, retrieval and poisoning generators were not modified.

## Rebuild and remaining risks

After fixing source URLs or providing OCR-backed PDFs, run:

~~~bash
python -m src.ingest_corpus --metadata data/metadata.csv --audit-path reports/dataset_source_audit.csv
python -m src.dataset
~~~

Use --force for changed content at an unchanged URL. To reproduce the isolated audit, add --db-path data/rebuild/corpus.sqlite --raw-dir data/rebuild/raw. Keep full experiments blocked until all categories have 80 valid documents. Source repair and visual checking of OCR/multi-column text remain prerequisites for final scientific results. The code does not accept missing or unusable sources. The ten-document poison budget is unchanged and represents a smaller training fraction.

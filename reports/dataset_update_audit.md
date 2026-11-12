# Dataset update audit ? 12 November 2026

Branch: `dataset-320-ingestion-update`. No commits or pushes were made.

The code targets **320 documents, 80 per category, with 256 train / 64 test** (64/16 per category). The live extraction audit found **304 usable documents and 16 rejected Campus sources**. The production corpus is therefore **not ready**; no 320-document model results are claimed.

## Source audit and observed lengths

The current [metadata spreadsheet](https://docs.google.com/spreadsheets/d/1D2YwcC_XE9SZRnGOFqxx1-l5SPYDg5UfdqJ2rno8eeU/edit) was read without edits and saved to `data/metadata.csv`. All 320 IDs and URLs were inspected; each category has 80 entries and there are no duplicate source URLs.

Counts below are whitespace-separated words after extraction and source-specific cleaning, before model normalization. Rejected sources are excluded.

| Category | Accepted | Min | Median | 90th percentile | Max |
|---|---:|---:|---:|---:|---:|---:|
| Sports | 80 | 85 | 276.5 | 441.8 | 638 |
| Technology | 80 | 65 | 264.5 | 535.8 | 2,519 |
| Politics | 80 | 70 | 316.0 | 736.1 | 1,314 |
| Campus | 64 | 3,838 | 8,953.5 | 12,902.6 | 17,872 |

## Normalization decision

The magazines are tens of times longer than typical articles. Use the 90th percentile of cleaned non-Campus **training** document lengths as the prefix budget: this keeps approximately 90% of training articles within budget without letting magazine lengths determine the limit. At seed 42, the 192 article training documents yield a **534-word cap**. This is a calculated reference value, not a hardcoded constant. Other seeds may produce different caps.

Apply the same cap to every category. Retain the original word order, stop at the last complete sentence if it fills at least half the budget, and otherwise use a word-boundary prefix for poorly punctuated PDF text. Never randomly sample words. Override `max_document_words` or `length_percentile` through `prepare_document_corpus`. Full extraction remains available in SQLite; `model_text` is used consistently for feature matrices and test-time evasion. Vocabulary remains train-only with `min_df=2`, `max_df_ratio=0.85`.

## Extraction and stale-data protection

- Central routing: Wikinews uses MediaWiki with an HTML fallback; ordinary HTML uses BeautifulSoup; PDF URL hints, response MIME type and byte signatures route to PDF extraction.
- Added `PyMuPDF>=1.24.0,<2`; tested with 1.28.2. PDF bytes are signature-checked, opened from memory, and extracted page by page as ordered text blocks. See the [PyMuPDF text extraction documentation](https://pymupdf.readthedocs.io/en/latest/recipes-text.html). PDF bytes never enter UTF-8 HTML decoding.
- Drive viewer URLs are converted to download requests. `CAMPUS_048` succeeded (14,457 words). Login/confirmation responses produce actionable failures.
- BITScan cleaning removes page numbers, recognizable issue/semester headers, short repeated page-edge furniture, dot-leader TOC entries and explicit editorial/contact labels. Campus terms and substantive editorial prose remain.
- Empty, fewer-than-50-word, garbled and mostly empty PDF text layers are rejected. Over 5,000 words is flagged as unusually long, then handled by model normalization. The PDF coverage check requires at least half the pages to have 20 words of text.
- SQLite cache reuse requires matching source URL, metadata hash, extraction version and text hash. Raw filenames alone are never trusted; verified database content repairs missing or altered raw files.
- Changed/removed sources are invalidated before downloads. Failed refreshes leave no obsolete SQLite row. An incomplete run cannot be prepared for modeling. First migration of a legacy database creates a `.legacy.bak` copy; `--force` refreshes even same-URL sources.
- Processed matrices are always rebuilt. Manifests record readiness, source fingerprint, seed, split counts and normalization. New experiment outputs go to `results/dataset_320/`, with per-experiment dataset manifests. Historical results and source files were preserved.

## Sources requiring intervention

The following 14 PDFs require OCR or a replacement official source with a usable text layer; the remaining two URLs return HTTP 404. They were rejected rather than replaced with old Campus text.

| ID | Source URL | Required action |
|---|---|---|
| CAMPUS_009 | [PDF](https://www.bits-pilani.ac.in/wp-content/uploads/I-Semester-2019-2020.pdf) | OCR or supply an official PDF with a usable text layer |
| CAMPUS_012 | [PDF](https://www.bits-pilani.ac.in/wp-content/uploads/II-Semester-2018-2019.pdf) | OCR or supply an official PDF with a usable text layer |
| CAMPUS_026 | [PDF](https://www.bits-pilani.ac.in/wp-content/uploads/II-Semester-2011-2012.pdf) | OCR or supply an official PDF with a usable text layer |
| CAMPUS_029 | [PDF](https://www.bits-pilani.ac.in/wp-content/uploads/I-Semester-2009-2010.pdf) | OCR or supply an official PDF with a usable text layer |
| CAMPUS_030 | [PDF](https://www.bits-pilani.ac.in/wp-content/uploads/II-Semester-2009-2010.pdf) | OCR or supply an official PDF with a usable text layer |
| CAMPUS_031 | [PDF](https://www.bits-pilani.ac.in/wp-content/uploads/I-Semester-2008-2009.pdf) | OCR or supply an official PDF with a usable text layer |
| CAMPUS_032 | [PDF](https://www.bits-pilani.ac.in/wp-content/uploads/II-Semester-2008-2009.pdf) | OCR or supply an official PDF with a usable text layer |
| CAMPUS_034 | [PDF](https://www.bits-pilani.ac.in/wp-content/uploads/II-Semester-2007-2008.pdf) | OCR or supply an official PDF with a usable text layer |
| CAMPUS_035 | [PDF](https://www.bits-pilani.ac.in/wp-content/uploads/I-Semester-2006-2007.pdf) | OCR or supply an official PDF with a usable text layer |
| CAMPUS_036 | [PDF](https://www.bits-pilani.ac.in/wp-content/uploads/II-Semester-2006-2007.pdf) | OCR or supply an official PDF with a usable text layer |
| CAMPUS_037 | [PDF](https://www.bits-pilani.ac.in/wp-content/uploads/I-Semester-2005-2006.pdf) | OCR or supply an official PDF with a usable text layer |
| CAMPUS_038 | [PDF](https://www.bits-pilani.ac.in/wp-content/uploads/II-Semester-2005-2006.pdf) | OCR or supply an official PDF with a usable text layer |
| CAMPUS_040 | [PDF](https://www.bits-pilani.ac.in/wp-content/uploads/II-Semester-2004-2005.pdf) | OCR or supply an official PDF with a usable text layer |
| CAMPUS_042 | [PDF](https://www.bits-pilani.ac.in/wp-content/uploads/II-Semester-2003-2004.pdf) | OCR or supply an official PDF with a usable text layer |
| CAMPUS_062 | [PDF](https://www.bits-pilani.ac.in/wp-content/uploads/BITSCAN-Jan15-toDec15.pdf) | Repair/replace the official URL (HTTP 404) |
| CAMPUS_064 | [PDF](https://www.bits-pilani.ac.in/wp-content/uploads/BITSCAN-Jan-13-toDec14.pdf) | Repair/replace the official URL (HTTP 404) |

## Local artifacts and rebuild

The isolated audit is in `data/rebuild/`: `corpus.sqlite`, `raw/`, `metadata.csv`, `ingestion_report.json` and `ingestion.log`. It contains the current accepted sources and an incomplete-run marker. The old production `data/corpus.sqlite` and `data/raw/` were preserved; the loader rejects that legacy database. The old processed split was replaced with `rebuild_required`. No model matrices or full experiment results were produced from the incomplete live corpus.

After repairing the metadata URLs or providing OCR-backed PDFs, resume the isolated audit:

```bash
python -m src.ingest_corpus --metadata data/metadata.csv --db-path data/rebuild/corpus.sqlite --raw-dir data/rebuild/raw
```

For the production rebuild, using the reviewed CSV (or omit `--metadata` to fetch the latest sheet):

```bash
python -m src.ingest_corpus --metadata data/metadata.csv
python -m src.dataset
python experiments/run_self_collected.py
python experiments/lda_baseline.py
python experiments/poison_budget.py
python experiments/test_time_evasion.py
```

Use `--force` if the contents at an unchanged URL have been repaired. Keep OCR text inside a PDF text layer and point the metadata at its accessible PDF URL; no viewer HTML or manual SQLite injection is required.

## Verification and changed files

`python -m pytest -q`: **43 passed**, including the original nine. Tests cover routing, PDF extraction/rejection, conservative cleaning, stale URLs, cache integrity, legacy backups, deterministic normalization, all 320 IDs, 64/16 splits, held-out vocabulary exclusion, and short integration runs through all four self-collected experiment entry points. Reduced iteration counts apply only within the integration test. Python compilation and `git diff --check` passed. The real legacy and incomplete databases were both verified to reject preparation.

- Ingestion/preprocessing: `src/ingest_corpus.py`, new `src/source_extract.py`, `src/preprocess.py`, `requirements.txt`.
- Corpus construction: `src/dataset.py`, `data/metadata.csv`, `data/processed/splits.json`.
- Consumers: `demo.py`, `experiments/run_self_collected.py`, `experiments/lda_baseline.py`, `experiments/poison_budget.py`, `experiments/test_time_evasion.py`, and the three `experiments/visualize_*.py` scripts.
- Verification/docs: new `tests/test_ingestion.py`, new `tests/test_dataset.py`, `README.md`, `.gitignore`, the historical-report banner in `reports/status_update_draft.md`, and this audit.

## Remaining limitations

The live 320-document rebuild is blocked by the 16 sources listed above. Review multi-column reading order and OCR quality before accepting replacements. Prefix normalization favors the opening sections of a magazine; a future section-based policy could reduce that bias but would need a separate evaluation. Same-URL remote changes require `--force`; the offline loader checks the saved metadata snapshot rather than contacting the Sheet. The unchanged ten-document poison budget represents a smaller fraction of 256 clean training documents. PLSA EM, tempered EM, folding-in, alignment, attacks, defenses and retrieval algorithms were not changed. The separate 20 Newsgroups experiments keep their own configuration.

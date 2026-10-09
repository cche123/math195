# MATH 195: emotional mood in EDM

Research question: Which musical characteristics help explain the emotional mood of EDM?


## Report sections

The report asks which musical characteristics help explain the emotional mood of EDM.

- [Section 2: Background and Literature Review](Report/02_background_and_literature_review.tex)
- [Section 3.1: Mathematical Model](Report/03_1_mathematical_model.tex)
- [Section 3.2: Preliminary Data Selection Audit](Report/03_2_data_selection_audit.tex)
- [Bibliography](Report/references.bib)

The LaTeX sections contain academic prose and section headings, with no paste instructions. The literature review is a developed draft for continued revision. The mathematical framework specifies acoustic measurements, separate valence and arousal ridge models, two propositions with proofs, grouped evaluation, feature-family comparisons, and case-study prediction contributions. Model fitting and the final EDM sample review remain upcoming work.

The sections were checked together using the instructor template's class, font, margins and theorem definitions. The local check used BibTeX because biblatex is unavailable in the runtime; Overleaf can retain the original biblatex configuration. All nine citation keys resolved without layout warnings.


## Correcting the earlier count

The recovered MERGE Audio Complete metadata (Zenodo record 13939205, version 1.1.0) contains **3,554 excerpt records**. An exact `Electronic` token in the comma-separated `Genres` field selects **543**, including tracks with several genres. This is not 543 verified EDM songs. The earlier 542 count is not supported by the reproduced filter and should be corrected; there is no documented one-record exclusion to justify it.

Requiring `Genres` to contain only Electronic selects 61 records. The earlier exploratory script required Electronic plus at least one genre/style from an explicit dance-style allowlist; that rule selects **92 provisional candidates**, with quadrant counts 34, 35, 18, 5. This filter is documented for reproducibility, not presented as the definitive meaning of EDM. It includes borderline styles and misses tracks outside the allowlist. A final sample requires a stated genre policy, manual review and logged inclusion/exclusion decisions.

We have not reproduced a count around 300. A different MERGE release, audio versus bimodal subset, or genre rule could yield a different count. Compare the actual input file and rule before reconciling it. Multiple emotion or genre tags do not create extra songs.

## Reproduce the audit

Python 3.11 or later; standard library only. Download `MERGE_Audio_Complete.zip` from [the official record](https://zenodo.org/records/13939205). It is roughly 1.16 GB. The archive is not stored in this repository.

```sh
python scripts/audit_merge.py --archive /path/to/MERGE_Audio_Complete.zip --out audit_output
```

Or use the metadata CSV after extracting the official archive:

```sh
python scripts/audit_merge.py --metadata /path/to/merge_audio_complete_metadata.csv --out audit_output
python -m unittest discover -s tests -v
```

The archive route checks the official MD5. The CSV route records a SHA-256 fingerprint but does not claim to verify the archive. `audit/counts.json` records the reproduced CSV audit; its fingerprint identifies the recovered CSV bytes. The earlier extraction used pandas to rewrite the CSV, so this fingerprint need not match the byte formatting of the original archive member.

Outputs include counts and per-quadrant counts, the full Electronic-tagged selection, and the earlier provisional selection with the matching tags. Counting uses unique `Song` IDs and fails on missing or repeated IDs. Artist/title duplicates are reported separately and are not silently dropped. They should be reviewed before modeling.

## Modeling scope

The proposed primary models are two ridge regressions for supplied valence and arousal scores, using seven interpretable acoustic features. Genre/style metadata select the sample; mood tags, emotion-derived metadata and response labels must never become predictors. No fitted results or causal conclusions are claimed here. Models, preprocessing and tuning will use song/recording groups and preferably artist-held-out evaluation; repeated versions require a reviewed grouping key.

GitHub plugin was used to publish this reproducibility code. Source recordings are omitted; consult the dataset terms before redistribution.

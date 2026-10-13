# MATH 195: Acoustic Characteristics and Emotional Mood in Electronic Dance Music

Caden Che and Kevin Zuniga

Research question: **Which musical characteristics help explain the emotional mood of EDM?**

This repository contains an executed exploratory study of **92 provisional dance-style candidates** from MERGE Audio Complete (version 1.1.0), with 63 normalized full artist-credit groups. The documented metadata filter selects 543 broad Electronic-tagged excerpts from 3,554 complete audio records; an additional genre/style allowlist selects the 92 candidates. These candidates include dance-pop and experimental/crossover recordings and have not been manually verified as a final EDM population. The earlier 542 count is not reproduced. No additional emotion dataset has been pooled with MERGE.

## Evaluated findings

The primary seven-feature ridge regressions use five outer artist-credit-group folds and three inner folds. The primary split uses seed 195. The ridge penalty minimizes pooled inner-validation MSE on a fixed grid. All scaling is fitted within training partitions. Each test prediction excludes its complete normalized artist-credit group, although aliases and shared personnel across different credits are not resolved.

| Response | Training-mean RMSE | Ridge RMSE | Ridge out-of-fold R² |
| --- | ---: | ---: | ---: |
| Valence | 0.1828 | 0.1873 | -0.0569 |
| Arousal | 0.0992 | 0.0931 | 0.1180 |

Arousal RMSE improves by approximately 6.2% relative to the baseline in the primary split. Valence does not improve. Across ten group-split seeds, arousal improves over its corresponding baseline in all ten evaluations; valence is worse in nine. Repeated partitions overlap and are not independent datasets or confidence intervals.

Removing spectral brightness and spectral change worsens arousal prediction in all five primary outer folds. Removing amplitude or rhythm slightly improves aggregate error in that split. These are conditional predictive comparisons, not causal effects.

An exploratory twelve-feature extension adds time slopes and adjacent-bin acoustic changes. Its main-split improvements are small and inconsistent across split seeds. The strongest adjacent-bin amplitude rise occurred after the first bin in 69 of 92 clips, indicating sensitivity to excerpt-edge behavior. A post-hoc first/last-bin exclusion check did not improve on the primary model in the primary split. The ten-bin profiles measure changes in sound; they do not provide verified buildup/drop/breakdown labels or dynamic emotion annotations. The two illustrated dataset recordings were selected by amplitude change, not prediction accuracy, and are not Caden and Kevin's personal case studies.

## Reproduce results without downloading recordings

Python 3.12 was used. From the repository root:

```sh
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python scripts/run_all.py
```

The included immutable original feature table is `data/provisional_92_features.csv`. The included recomputed feature table and `data/temporal_profiles.csv` supply the temporal extension. No audio is needed to reproduce the numerical model results and figures from these tables.

Outputs include held-out predictions, fold details and selected penalties, standardized coefficients, feature-family comparisons, split-sensitivity results, input fingerprints, and JSON primary-model refits. `scripts/run_all.py` regenerates the outputs, figures, and result prose. It does not perform a new feature extraction or compile LaTeX. Figures and figures' source code are included.

## Recompute features from the official audio

Install the audio dependencies, download the official archive (approximately 1.16 GB), verify its checksum, and extract the selected IDs:

```sh
python -m pip install -r requirements-audio.txt
python scripts/download_audio.py
python scripts/extract_features.py --audio-dir data/audio --metadata data/provisional_92_features.csv --out data/recomputed_features.csv
python scripts/run_all.py
```

The archive URL and MD5 refer to [the official MERGE release](https://zenodo.org/records/13939205). Source recordings are not redistributed here. This extraction uses librosa 0.11.0, mono 22,050 Hz audio, a 2,048-sample Hann STFT, and a 512-sample hop. It reproduced all seven original stored features exactly for the 92 excerpts in this run. Consult the dataset terms before using or redistributing its materials.

The audit can also be reproduced from the official metadata or archive:

```sh
python scripts/audit_merge.py --archive data/raw/MERGE_Audio_Complete.zip --out audit_output
```

## Report and Overleaf

`Report/main.tex` is a complete exploratory report using the instructor's 11-point AMS article class and one-inch margins. It includes introduction, literature review, mathematical model and proofs, sample audit, implementation, measured results, temporal/structure discussion, conclusion, bibliography, and supplementary figures. Upload the contents of `Report` into an Overleaf project and compile `main.tex`; keep `references.bib`, `preamble.tex` and `figures/` alongside it. No instruction placeholders are included in the report body.

The local PDF preview uses a temporary BibTeX wrapper because biblatex is unavailable in the runtime. The supplied Overleaf main file retains biblatex. The wrapper and build intermediates are omitted from the repository.

## Personal case studies and annotated sections

The primary full-sample refits are saved in `results/fitted_models.json`. They are not evaluated using their training predictions. To predict measured features for external songs:

```sh
python scripts/predict_case_study.py --features data/external_song_features.csv
```

The input must contain the seven named acoustic features and may include Song, Artist, and Title. The script rejects known training song IDs and flags measurements outside the training ranges and artist credits seen in training. It outputs individual feature contributions rather than invented mood ground truth.

For independently annotated full-song sections, create a CSV with Song, start_seconds, end_seconds, and section. Valid section labels are intro, buildup, drop, breakdown, core, and outro. Intervals must be non-overlapping, lie inside the recording, and span at least six seconds:

```sh
python scripts/analyze_sections.py --annotations data/section_annotations.csv --audio-dir data/full_songs
```

Actual annotated section results and the two chosen favorite-song case studies remain pending. The trained models use static excerpt labels. Applying them to short sections does not establish how emotion changes over time, and it introduces a duration/context difference from the training data.

## Analysis boundaries

The primary model and features were specified before this fitting task. The temporal extension was requested after the primary results were examined and is explicitly exploratory. No test results were used to select a favorable split or suppress a negative finding. There are no reported significance tests, causal effects, or manually verified section labels. Only the primary model is saved for later external-song predictions.

GitHub plugin was used to publish the code and report artifacts.

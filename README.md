# How Do Acoustic Features Relate to the Emotions We Hear in Music?

**Caden Che and Kevin Zuniga | MATH 195 | Stanford University**

We study how measurements of recorded sound relate to emotional positivity and activation. A recording can sound energetic and sad at the same time, so we model **valence** and **arousal** separately. We evaluate acoustic relationships on held-out recordings and examine how they apply to individual songs.

## Data and scope

Our primary source is [MERGE Audio Complete, version 1.1.0](https://zenodo.org/records/13939205), with 3,554 excerpts, descriptive metadata, and supplied emotion labels. Predictors are extracted from audio; supplied valence and arousal scores are the responses. Mood tags and emotion-derived metadata are excluded from predictors. [DEAM](https://cvml.unige.ch/databases/DEAM/) was considered for its time-varying annotations but has not been pooled with MERGE.

We initially focused on EDM. Exact-token metadata filtering identified 543 Electronic-tagged excerpts and 92 provisional dance-style candidates. These tags included music outside our intended focus. Our review identified approximately seven recordings matching that narrower definition, motivating a broader study across genres. This is not a count of every electronic recording in MERGE.

## Current written report

The current report starts at [`Report/main.tex`](Report/main.tex). It follows the course's AMS article template with 11-point text, one-inch margins, one-and-a-half spacing, and a separate bibliography. Upload the contents of `Report/` to Overleaf and select `main.tex` as the main document. The figures in the current Results section are drawn directly in LaTeX.

The report describes the broader follow-up on 3,542 eligible recordings in 1,755 normalized complete artist-credit groups. It compares seven initial measurements with 82 features covering amplitude, rhythm, timbre, and pitch-class content.

| Model | Valence RMSE | Valence R² | Arousal RMSE | Arousal R² |
| --- | ---: | ---: | ---: | ---: |
| Training mean | 0.2233 | -0.000 | 0.1347 | -0.005 |
| Seven-feature OLS | 0.2129 | 0.091 | 0.1121 | 0.305 |
| Seven-feature ridge | 0.2129 | 0.090 | 0.1121 | 0.304 |
| Expanded ridge | 0.1980 | 0.213 | 0.1028 | 0.415 |
| Expanded kernel approximation | 0.1957 | 0.232 | 0.1022 | 0.422 |
| Expanded random forest | 0.1877 | 0.293 | 0.0956 | 0.494 |
| Small LSTM | 0.2137 | 0.083 | 0.1111 | 0.317 |

These values summarize the earlier executed follow-up documented in the report. Its broader feature tables, fitted models, full prediction outputs, and analysis scripts are not currently included in this repository. The checked-in scripts and data reproduce the earlier 92-recording pilot, not these broader results. The broader analysis is not yet reproducible from this checkout alone. No new broad-sample evaluation was performed in this update.

The evaluated representations predict arousal more strongly than valence. Added timbre and pitch-class measurements improve valence prediction. These are exploratory associations within MERGE. R² measures explained variation, not classification accuracy, and feature contributions do not establish causes of emotion.

## Case studies

| Selected by | Recording | Initial description | Status |
| --- | --- | --- | --- |
| Caden | “Apocalypse” by Cigarettes After Sex | Sad | Excerpt and predictions pending |
| Kevin | “Happy” by Pharrell Williams | Happy | Excerpt and predictions pending |

The case-study text is in [`Report/03_5_case_studies.tex`](Report/03_5_case_studies.tex), included at the end of Results. The selections are recorded in [`data/case_studies.csv`](data/case_studies.csv). Empty timestamp fields are intentional.

We will document the recording versions, approximately 30-second excerpts, and listening descriptions before examining predictions. Training-recording overlap must be checked before treating either case as external. The broader random forest will provide a predictive comparison, while broader ridge refits will provide interpretable contributions once those artifacts are restored. Ridge contributions explain ridge predictions only.

We will also compare manually annotated sections where appropriate. Section predictions remain exploratory because the models were trained on static excerpt labels. Neither selected song has been analyzed yet.

## Readings and extraction

- [MERGE](https://arxiv.org/abs/2407.06060): dataset construction and static emotion labels.
- [Kang and Herremans](https://arxiv.org/abs/2406.08809): perceived versus induced emotion and evaluation challenges.
- [EmoMucs](https://archives.ismir.net/ismir2020/paper/000308.pdf): emotion prediction from separated musical sources.
- [Solberg](https://dj.dancecult.net/index.php/dancecult/article/view/451): expectations and intensity in EDM buildups and drops.
- [Demucs](https://arxiv.org/abs/2211.08553) and [MERT](https://arxiv.org/abs/2306.00107): possible extensions through source separation and learned representations.

Completed extraction used the full mix and librosa. Demucs, MERT, and Essentia were considered but not used. Bass-band power is not an isolated bass instrument, and pitch-class features are not verified chord annotations.

## Reproducing the earlier pilot

The included tables contain seven features for 92 provisional dance-style candidates:

```sh
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python scripts/run_all.py
```

The pipeline updates pilot outputs in `results/`, pilot figures, and `Report/pilot/03_4_results.tex`. It leaves the current report unchanged. The archived pilot document is `Report/pilot/main.tex`; select it explicitly in Overleaf if needed.

To download audio and recompute the seven pilot features:

```sh
python -m pip install -r requirements-audio.txt
python scripts/download_audio.py
python scripts/extract_features.py --audio-dir data/audio --metadata data/provisional_92_features.csv --out data/recomputed_features.csv
```

The downloader selects pilot recordings, not the broader sample. Extraction uses mono audio at 22,050 Hz, a 2,048-sample Hann STFT, and a 512-sample hop. Source audio is not redistributed.

To reproduce the metadata filtering:

```sh
python scripts/audit_merge.py --archive data/raw/MERGE_Audio_Complete.zip --out audit_output
```

The original data-selection account remains unchanged at `Report/03_2_data_selection_audit.tex`. The current report includes `Report/03_2_data_collection.tex`, containing the introductory dataset discussion for Kevin to complete.

## Section measurements and pilot predictions

```sh
python scripts/analyze_sections.py --annotations data/section_annotations.csv --audio-dir data/full_songs
python scripts/predict_case_study.py --features data/external_song_features.csv
```

Section annotations require `Song`, `start_seconds`, `end_seconds`, and `section`. Audio filenames must match `Song`. Supported labels include intro, verse, prechorus, chorus, bridge, interlude, buildup, drop, breakdown, core, and outro. Intervals must lie within the audio, be nonoverlapping, and span at least six seconds. This script extracts the original seven features.

The case predictor defaults to the saved **pilot** ridge models in `results/fitted_models.json`. Its outputs are not the expanded-model case studies described in the current report. The broader refits are required before completing that comparison.

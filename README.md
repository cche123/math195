# MATH 195: Music and Emotion

**Caden Che and Kevin Zuniga | Stanford University**

**Research question: How do acoustic features relate to the emotions we hear in music?**

We both enjoy music, and some of our favorite songs sound energetic and sad at the same time. We want to understand how that combination shows up in the audio. Our project measures properties of a recording, such as amplitude, rhythm, brightness, and low-frequency content, and examines how they relate to its supplied emotion labels.

We study two dimensions of emotion: **valence**, which describes positivity, and **arousal**, which describes activation. Keeping these separate lets us ask how a song can have an active beat and still have a negative emotional character.

## Why we broadened the question

We originally focused on electronic dance music. The metadata audit found 543 Electronic-tagged excerpts in MERGE, and an additional dance-style filter selected 92 candidates. Those categories included experimental music, crossover recordings, and other styles outside the EDM focus we had in mind.

In our subsequent review, we identified about seven recordings that matched that intended focus. This is our assessment of the reviewed candidates, rather than a count of every electronic recording in MERGE. We decided to broaden the question to music across genres.

The existing data tables, models, and figures still come from the **earlier 92-recording pilot**. Updating the question does not turn those results into findings about the full MERGE dataset. Preparing and evaluating the broader sample is the next stage.

## Data and readings

Our primary source is [MERGE Audio Complete, version 1.1.0](https://zenodo.org/records/13939205), which contains 3,554 audio excerpts with metadata and supplied emotion labels. We use the numerical valence and arousal responses. Mood tags and other information derived from those responses are excluded from the predictors.

We have also considered [DEAM](https://cvml.unige.ch/databases/DEAM/), which includes recording-level and time-varying emotion annotations. We have not combined it with MERGE.

Several readings helped shape the project:

- [MERGE](https://arxiv.org/abs/2407.06060) provides the dataset and explains how its labels were constructed.
- [Kang and Herremans' survey](https://arxiv.org/abs/2406.08809) helps us distinguish perceived emotion from induced emotion and overall labels from annotations over time.
- [EmoMucs](https://archives.ismir.net/ismir2020/paper/000308.pdf) examines emotion prediction from separate musical sources.
- [Solberg's study of buildups and drops](https://dj.dancecult.net/index.php/dancecult/article/view/451) motivated our interest in musical changes and contrasts.
- [Demucs](https://arxiv.org/abs/2211.08553) and [MERT](https://arxiv.org/abs/2306.00107) offer possible extensions through source separation and learned audio representations.

The literature review in `Report/` discusses how these studies inform our choices and where their tasks differ from ours.

## What we measure

The pilot uses seven features extracted with librosa:

| Feature | What it describes |
| --- | --- |
| Mean RMS amplitude | Typical signal amplitude |
| RMS standard deviation | Variation in amplitude |
| Spectral centroid | Spectral brightness |
| Bass-band power fraction | Relative power between 20 and 250 Hz |
| Positive spectral flux | Changes in spectral magnitude |
| Onsets per second | Rate of detected sound attacks |
| Estimated tempo | Estimated beat rate in BPM |

These are measurements of sound. The emotion responses come from the dataset's annotations. For example, tempo is a predictor, rather than our definition of arousal.

Our starting models are ordinary least squares and ridge regression, fitted separately for valence and arousal. Ridge limits coefficient size when features contain overlapping information. We compare both models with a baseline that predicts the training mean.

We also plan to examine consecutive windows with an LSTM and compare features from Demucs-estimated vocals, bass, drums, and accompaniment. **LSTM training and source separation have not been completed.** The current extraction uses the full mix and librosa; Essentia and MERT remain possible options.

## What the pilot found

The 92-recording pilot uses five outer folds and three inner folds, grouped by normalized complete artist credits. Feature scaling and ridge tuning use only the relevant training partitions.

| Response | Training-mean RMSE | Ridge RMSE | Ridge out-of-fold R² |
| --- | ---: | ---: | ---: |
| Valence | 0.1828 | 0.1873 | -0.0569 |
| Arousal | 0.0992 | 0.0931 | 0.1180 |

The initial features helped predict arousal modestly, but they did not improve valence prediction over the baseline. Removing brightness and spectral-change features made arousal prediction worse in all five primary folds. These comparisons describe predictive associations in this sample.

An exploratory extension measured changes within ten consecutive bins. Its small improvements were inconsistent across alternative splits. These bins do not identify verified verses, choruses, buildups, or drops, and the recordings do not have separate emotion labels for each bin.

## Reproducing the pilot

Python 3.12 was used. From the repository root:

```sh
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python scripts/run_all.py
```

The included feature tables allow the numerical results and figures to be reproduced without downloading audio. The pipeline saves predictions, evaluation summaries, coefficients, and figures. It refreshes the pilot results section while preserving the manually edited introduction, literature review, mathematical model, conclusion, and main report file.

To download the official audio and recompute the pilot features:

```sh
python -m pip install -r requirements-audio.txt
python scripts/download_audio.py
python scripts/extract_features.py --audio-dir data/audio --metadata data/provisional_92_features.csv --out data/recomputed_features.csv
python scripts/run_all.py
```

The downloader still selects the pilot recordings. It does not prepare the revised sample across genres. The extraction uses mono audio at 22,050 Hz, a 2,048-sample Hann STFT, and a 512-sample hop. The seven recomputed features matched the stored pilot values in the earlier check. Source recordings are not redistributed here.

To reproduce the original metadata selection from the official archive:

```sh
python scripts/audit_merge.py --archive data/raw/MERGE_Audio_Complete.zip --out audit_output
```

## Report and case studies

The LaTeX report is in `Report/`. Upload that folder's contents to Overleaf and compile `main.tex`, keeping the bibliography and figures alongside it. The title, introduction, literature review, and model discussion reflect the revised question. The data, implementation, results, and conclusion sections retain the earlier pilot analysis and still need a coordinated update for the broader study.

We will choose two songs that we perceive as sad, write listening descriptions before inspecting predictions, and examine how their measured features contribute to predicted valence and arousal. The current case-study script uses the saved pilot models:

```sh
python scripts/predict_case_study.py --features data/external_song_features.csv
```

Its input needs the seven named acoustic features. Predictions for the revised study will require refitting on the broader sample first.

The section-analysis script currently accepts intro, buildup, drop, breakdown, core, and outro annotations:

```sh
python scripts/analyze_sections.py --annotations data/section_annotations.csv --audio-dir data/full_songs
```

Supporting other section names, such as verse, chorus, and bridge, requires an update to that script. Actual annotated-section results and our two personal case studies remain pending.

Our aim is to connect model predictions to properties we can hear and measure. A feature contribution explains part of a fitted prediction; it does not establish what caused a song to sound sad to a listener.

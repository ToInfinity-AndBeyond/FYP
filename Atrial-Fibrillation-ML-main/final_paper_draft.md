# PPG-Based Atrial Fibrillation Detection with Signal Quality Assessment and Edge-AI Deployment

**Author:** Minseok Chey  
**Supervisor:** Dr. Shukla Pancham  
**Second Marker:** Dr. Rolandos Potamias  
**Department of Computing, Imperial College London**

> Paper scope: MIMIC PERform AF is used as a controlled baseline, MIMIC-III-Ext-PPG p00-p02 is used as the main large-scale experiment, and Nordic Edge AI integration is used as the embedded deployment component.

## Abstract

Atrial fibrillation (AF) is the most common sustained cardiac arrhythmia and is associated with increased risk of stroke, heart failure, and mortality. Electrocardiography (ECG) remains the clinical diagnostic standard, but intermittent ECG can miss asymptomatic or paroxysmal episodes. Photoplethysmography (PPG) offers a scalable route for continuous rhythm screening, but it is an indirect optical measurement of peripheral blood volume and is sensitive to motion artefacts, low perfusion, poor contact, and non-AF irregular rhythms.

This project develops a PPG-only AF screening pipeline combining preprocessing, signal quality assessment (SQI), hybrid deep learning, record-level aggregation, and embedded deployment feasibility work. PPG is segmented into 30-second windows, band-pass filtered, z-score normalised, and converted into pulse timing, morphology, rhythm, and quality features. The offline classifier, `RhythmMorphologyFusionNet`, combines time-domain waveform morphology, log-magnitude STFT evidence, and a 17-feature rhythm/quality vector using gated fusion. Unreliable windows are rejected using an SQI v2 gate, and segment probabilities are aggregated at record-event level using quality-weighted averaging.

MIMIC PERform AF is used as a small controlled baseline, while MIMIC-III-Ext-PPG v1.1.0 p00-p02 is used as the main large-scale experiment. The p00-p02 AF-versus-sinus-rhythm subset contains 789,864 SQI-accepted segments, including 101,796 AF and 688,068 SR segments. Training uses metadata folds 0-7, validation uses fold 8, and testing uses fold 9.

On MIMIC PERform AF, the model achieved perfect record-level test metrics on a five-record test split, with segment-level F1 of 0.9558. This is treated only as a controlled sanity check. On MIMIC-III-Ext-PPG p00-p02, the final SQI v2 pipeline achieved record-level test accuracy 0.9536, sensitivity 0.8241, specificity 0.9729, precision 0.8194, F1 0.8217, AUROC 0.9272, and AUPRC 0.8365. Segment-level precision was lower at 0.6841, supporting record-level aggregation rather than single-window alerts. A compact 17-feature Nordic Edge AI model was also integrated into Zephyr firmware as a deployment-path demonstration, not as a reproduction of the full offline hybrid model. Overall, the results support the feasibility of SQI-aware record-level PPG AF screening on a large ICU-derived subset, while requiring full-prefix, multi-rhythm, and prospective validation before clinical use.

## Keywords

Atrial fibrillation; photoplethysmography; signal quality index; deep learning; wearable screening; edge AI; MIMIC-III-Ext-PPG; MIMIC PERform AF.

## 1. Introduction

Atrial fibrillation is a supraventricular arrhythmia characterised by disorganised atrial electrical activity and an irregular ventricular response. It is clinically important because undetected AF can lead to preventable thromboembolic stroke and other cardiovascular complications. Detecting AF early is therefore valuable, but the practical detection problem is difficult: many patients experience intermittent or asymptomatic episodes, and short clinical ECG recordings may not capture them.

Wearable and continuous monitoring devices provide an opportunity to detect rhythm abnormalities outside the clinic. PPG is especially attractive because it is already present in many optical heart-rate sensors and can be collected frequently with low user burden. AF can manifest in PPG as irregular pulse intervals, beat-to-beat variability, and changes in pulse morphology. These patterns make PPG useful for screening, but PPG does not directly measure atrial electrical activity. It does not provide P-waves, PR intervals, or QRS morphology. As a result, PPG-based AF detection must infer rhythm from peripheral pulse timing and morphology, which can be confounded by motion, ectopic beats, sensor pressure, vasoconstriction, low perfusion, and device-specific noise.

The central research question of this project is:

> Can a PPG-only AF screening pipeline achieve useful record-level AF detection on a large clinical PPG subset while remaining compatible with eventual embedded deployment?

The project is framed as screening rather than diagnosis. A positive PPG prediction should indicate likely AF and motivate further confirmation, but it should not be presented as a replacement for diagnostic ECG.

The main contributions are:

1. A staged experimental design using MIMIC PERform AF as a controlled baseline and MIMIC-III-Ext-PPG p00-p02 as the main large-scale benchmark.
2. A 30-second PPG preprocessing pipeline with band-pass filtering, peak detection, rhythm feature extraction, and explicit SQI-based segment filtering.
3. A hybrid deep learning model that combines raw waveform morphology, spectral content, and 17 hand-crafted rhythm/quality features.
4. A record-level aggregation strategy that converts segment probabilities into episode-level predictions using quality-weighted averaging.
5. A Nordic Edge AI deployment path using a compact 17-feature model integrated into an nRF9160/Zephyr firmware project with an embedded smoke-test harness, presented as feasibility evidence rather than as a reproduction of the full offline model.

The final paper intentionally reports the datasets separately. MIMIC PERform AF is a controlled baseline, while MIMIC-III-Ext-PPG is the main evaluation. The results are not presented as a single mixed-dataset training result because the datasets differ in scale, label provenance, patient context, and acquisition conditions. Combining them without a careful domain-adaptation study could make the interpretation less clear rather than stronger.

## 2. Background and Motivation

### 2.1 Atrial Fibrillation Detection from PPG

AF detection from PPG relies mainly on pulse irregularity. In sinus rhythm, pulse-to-pulse intervals are relatively regular except for physiologic variability. In AF, atrial activity is disorganised and ventricular response becomes irregularly irregular, which can appear in PPG as irregular inter-beat intervals (IBIs). Classical PPG AF detectors therefore often use features such as SDNN, RMSSD, pNN50, coefficient of IBI variation, sample entropy, pulse-rate variability, and beat-template consistency.

The difficulty is that irregularity is not specific to AF. Premature atrial contractions, premature ventricular contractions, missed peaks, duplicate peaks, motion artefacts, or poor optical contact can all create irregular PPG intervals. A detector trained only to recognise irregularity may therefore produce false positives unless it also understands signal quality and pulse morphology.

### 2.2 Signal Quality Assessment

Signal quality assessment is essential for PPG-based rhythm screening. A segment with severe motion artefact, flatlining, missing peaks, or inconsistent morphology may not contain enough reliable information for AF inference. A low-quality segment should either be rejected or down-weighted. This is especially important for wearable-style use because the sensor is affected by movement, skin contact, and peripheral perfusion.

In this project, quality is not treated as an implicit preprocessing detail. It is exposed in the pipeline through SQI-derived features, acceptance/rejection decisions, quality-weighted training loss, and quality-weighted record aggregation. This makes it possible to interpret whether performance changes are due to better classification or better filtering of unreliable input.

### 2.3 Class Imbalance

AF detection datasets are often imbalanced. In MIMIC-III-Ext-PPG v1.1.0, the full heart-rhythm task contains 597,769 AF segments and 3,950,724 SR segments. In the p00-p02 SQI-accepted binary subset used here, AF accounts for 101,796 of 789,864 accepted SR/AF segments, or 12.9%. A model trained naively on this distribution could achieve high accuracy by favouring the majority SR class.

For this reason, accuracy alone is not sufficient. Sensitivity, specificity, precision, F1, AUROC, and AUPRC are all reported. Training uses class-aware sampling, but the validation and test sets retain their natural fold-level imbalance. This matters because strict 1:1 balancing can reduce majority-class bias during optimisation, but it does not guarantee better validation or test performance. It can also discard useful non-AF diversity if applied too aggressively. The retained final model therefore uses a 1:2 AF:SR sampling target during training rather than strict 1:1 undersampling.

### 2.4 Edge AI Motivation

The offline model is designed to evaluate detection performance from waveform and feature inputs. A wearable or embedded device, however, has memory, power, runtime, and integration constraints. For this reason, the project also includes a compact feature-based Edge AI deployment path. The embedded model is not claimed to be identical to the offline hybrid model. Instead, it demonstrates how the same 17-feature representation can be connected to an on-device inference runtime and exercised inside firmware.

## 3. Datasets

### 3.1 MIMIC PERform AF: Controlled Baseline

MIMIC PERform AF contains 20-minute ECG and PPG recordings from critically ill adults. In this project, 35 records are used: 19 AF and 16 non-AF. Signals are segmented into 30-second windows, producing approximately 40 windows per 20-minute record. The dataset is used as a controlled baseline because it is small and binary. It is suitable for checking whether the preprocessing and classifier pipeline can separate AF from non-AF under relatively curated conditions, but it is not large enough to support strong generalisation claims.

The dataset is part of the MIMIC PERform datasets, which were extracted from the MIMIC-III Waveform Database and released for PPG beat-detection benchmarking and related physiological signal analysis.

The patient-level split used in the reported baseline experiment is:

| Split | Records | AF records | Non-AF records |
|---|---:|---:|---:|
| Train | 25 | 13 | 12 |
| Validation | 5 | 3 | 2 |
| Test | 5 | 3 | 2 |

### 3.2 MIMIC-III-Ext-PPG v1.1.0: Main Dataset

The main experiment uses MIMIC-III-Ext-PPG v1.1.0, published on PhysioNet on 17 March 2026. The resource is a large-scale, annotated, and quality-assessed PPG benchmark derived from the MIMIC-III Waveform Database Matched Subset and MIMIC-III clinical records. The official PhysioNet citation is:

> Moulaeifard, M., Charlton, P. H., & Strodthoff, N. (2026). MIMIC-III-Ext-PPG: A PPG Benchmark Dataset for Cardiorespiratory Analysis (version 1.1.0). PhysioNet. RRID:SCR_007345. https://doi.org/10.13026/r6k1-xt76

The associated publication is:

> Moulaeifard, M., Kutscher, M., Aston, P. J. et al. MIMIC-III-Ext-PPG, a PPG-based Benchmark Dataset for Cardiovascular and Respiratory Signal Analysis. Scientific Data 13, 668 (2026). https://doi.org/10.1038/s41597-026-07335-8

The full MIMIC-III-Ext-PPG v1.1.0 heart-rhythm task contains:

| Property | Value |
|---|---:|
| Patients | 6,189 |
| Non-overlapping 30-second PPG segments | 6,399,754 |
| Total duration | approximately 53,331 hours |
| Sampling rate | 125 Hz |
| Mean age | 64.1 +/- 17.0 years |
| Mean weight | 82.2 +/- 22.6 kg |
| Mean height | 169.5 +/- 10.5 cm |
| Female patients | 43.9% |

The dataset contains harmonised rhythm labels including SR, sinus tachycardia, AF, sinus bradycardia, ventricular pacing, first-degree AV block, atrioventricular pacing, atrial flutter, atrial pacing, bundle branch block, and less common rhythms. The full label distribution includes 3,950,724 SR segments and 597,769 AF segments.

Each segment is stored in WFDB format. PPG is present for all segments, and simultaneous ECG lead II, arterial blood pressure, and respiratory signals are included where available. The metadata also include demographics, clinical information system, ICD-derived diagnosis codes, 10-fold stratification labels, derived HR/RR/BP annotations, and SQI values. The released WFDB waveform signals are raw segmented signals; users are responsible for applying filtering, denoising, and normalisation before model training.

### 3.3 Scope Used in This Project

This project uses prefixes p00, p01, and p02 of MIMIC-III-Ext-PPG. This subset was chosen because it is already much larger than MIMIC PERform AF and was computationally feasible within the project time and available GPU resources. The binary task retains:

- AF segments as the positive class.
- SR segments as the negative class.

Other rhythm classes are not included in the final binary results. This makes the task interpretable as AF versus sinus rhythm rather than general multi-class arrhythmia classification.

The metadata folds are used directly:

| Split | Metadata folds | Accepted segments | SR segments | AF segments | AF fraction | Record-event groups | SR groups | AF groups | Subjects |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Train | 0-7 | 622,065 | 537,476 | 84,589 | 0.1360 | 41,376 | 33,943 | 7,433 | 757 |
| Validation | 8 | 93,200 | 82,745 | 10,455 | 0.1122 | 5,742 | 4,848 | 894 | 97 |
| Test | 9 | 74,599 | 67,847 | 6,752 | 0.0905 | 5,301 | 4,613 | 688 | 103 |
| Total | 0-9 | 789,864 | 688,068 | 101,796 | 0.1289 | 52,419 | 43,404 | 9,015 | 957 |

Record-level evaluation uses `record_id+event_id` groups because a single waveform record may contain more than one rhythm chart event. This avoids assigning conflicting labels to the same evaluation group.

## 4. Signal Processing and Feature Extraction

### 4.1 Segmentation

All signals are represented as non-overlapping 30-second windows. At 125 Hz, each PPG segment contains 3750 samples. The 30-second duration is clinically meaningful because AF episodes of at least 30 seconds are commonly considered significant, and it provides enough beats for IBI-derived rhythm features to be informative.

For MIMIC-III-Ext-PPG, the source dataset already provides 30-second WFDB segments. The project still applies its own filtering, normalisation, SQI processing, and feature extraction because the released waveforms are raw segmented signals.

### 4.2 Filtering and Normalisation

Each PPG segment is filtered using a fourth-order band-pass filter:

| Parameter | Value |
|---|---:|
| Low cut-off | 0.5 Hz |
| High cut-off | 8.0 Hz |
| Filter order | 4 |
| Segment length | 30 s |
| Segment stride | 30 s |
| Normalisation | z-score |

The 0.5-8.0 Hz band keeps the frequency range relevant to plausible pulse rates and suppresses slow baseline drift and high-frequency noise. After filtering, each segment is z-score normalised so that the neural model receives amplitude-standardised waveform inputs.

### 4.3 Peak Detection

Pulse peaks are detected using an Elgendi-inspired morphology and derivative pipeline. The detector combines smoothed PPG, first- and second-derivative cues, local maxima, zero-crossing structure, amplitude constraints, prominence thresholds, and a minimum peak distance derived from plausible heart-rate limits.

The peak detector is configured with:

| Parameter | Value |
|---|---:|
| Minimum HR | 35 bpm |
| Maximum HR | 220 bpm |
| Prominence scale | 0.3 |
| Minimum absolute prominence | 0.02 |
| Peak refinement radius | 0.2 s |

An alternate prominence-based detector is also used as a consistency check. Low agreement between detectors suggests that the detected pulse train may not be reliable.

### 4.4 SQI v2 Acceptance Criteria

The SQI v2 gate rejects segments that are unlikely to support reliable rhythm analysis. The criteria include peak-count limits, spectral concentration in the heart-rate band, waveform distribution checks, template similarity, IBI plausibility, zero-crossing rate, SNR-like quality, and detector agreement.

| SQI criterion | Threshold used |
|---|---:|
| Minimum peak count | 15 |
| Maximum peak count | 125 |
| Minimum HR | 35 bpm |
| Maximum HR | 220 bpm |
| Minimum heart-band energy ratio | 0.55 |
| Maximum absolute skewness | 2.0 |
| Maximum kurtosis | 12.0 |
| Minimum template correlation | 0.55 |
| Maximum short IBI fraction | 0.20 |
| Maximum long IBI fraction | 0.20 |
| Maximum IBI outlier fraction | 0.35 |
| Maximum zero-crossing rate | 0.20 |
| Minimum SNR SQI | 1.2 |
| Minimum detector agreement | 0.55 |

Accepted segments are stored in `ppg_accepted_segments.npz` and summarised in `ppg_accepted_segment_summary.csv`. Rejected segments remain available in the full segment summary for auditing.

### 4.5 Hand-Crafted Feature Vector

The offline neural model and the compact Edge AI model use a 17-dimensional feature vector:

| Feature | Description |
|---|---|
| `peak_count` | Number of detected pulse peaks in the 30-second segment |
| `heart_band_energy_ratio` | Spectral energy concentration in the expected heart-rate band |
| `signal_skewness` | Distribution skewness of the filtered PPG segment |
| `template_correlation` | Similarity between individual beats and the average beat template |
| `estimated_hr_bpm` | Estimated heart rate from detected peaks |
| `quality_score` | Composite segment quality score |
| `ibi_count` | Number of inter-beat intervals |
| `mean_ibi_ms` | Mean IBI duration in milliseconds |
| `median_ibi_ms` | Median IBI duration in milliseconds |
| `sdnn_ms` | Standard deviation of IBIs |
| `rmssd_ms` | Root mean square of successive IBI differences |
| `pnn50` | Fraction of successive IBI differences greater than 50 ms |
| `mean_hr_bpm` | Mean beat-to-beat heart rate |
| `std_hr_bpm` | Standard deviation of beat-to-beat heart rate |
| `cv_ibi` | Coefficient of variation of IBIs |
| `sample_entropy` | Entropy measure of rhythm irregularity |
| `signal_spectral_entropy` | Entropy of the PPG spectral distribution |

Missing feature values are filled using medians computed on the training split only. Features are then standardised using training-split means and standard deviations, preventing validation or test information from leaking into preprocessing.

## 5. Deep Learning Model

### 5.1 Problem Formulation

Each input sample consists of:

- a filtered and normalised 30-second PPG waveform `x` with 3750 samples,
- a 17-dimensional feature vector `f`,
- a binary label `y`, where `y = 1` denotes AF and `y = 0` denotes SR.

The model outputs a logit `z`. The predicted AF probability is:

```text
p(AF | x, f) = sigmoid(z)
```

The classification threshold is selected on the validation set and then fixed for the test set.

### 5.2 Model Selection Rationale and Mathematical Background

The model was chosen to match the structure of the PPG AF detection problem rather than to maximise architectural complexity. A 30-second PPG segment contains several complementary signals: local pulse morphology, beat-to-beat timing irregularity, frequency-domain concentration or noise, and explicit signal-quality indicators. A feature-only model can capture rhythm irregularity but may miss waveform shape. A pure waveform CNN can learn morphology but may underuse clinically interpretable pulse-rate variability features. A Transformer-only model is also more expensive than necessary for local pulse-shape extraction. The retained hybrid model therefore uses separate branches for morphology, spectrum, and hand-crafted rhythm/quality features before fusing them.

Formally, each segment is represented as a pair:

```text
x in R^T, where T = 3750
f in R^17
y in {0, 1}
```

The classifier estimates the posterior probability of AF:

```text
p(y = 1 | x, f) = sigmoid(z)
```

where the logit `z` is produced by a learned function:

```text
z = psi(phi_time(x), phi_spec(x), phi_feat(f))
```

Here, `phi_time` is the time-domain waveform encoder, `phi_spec` is the spectral encoder, `phi_feat` is the feature encoder, and `psi` is the gated fusion classifier.

The time-domain branch uses multi-scale 1D convolution because PPG pulse morphology is local but not fixed to one temporal scale. Short kernels capture peak shape and upstroke/downstroke structure, while wider and dilated kernels capture longer morphology and rhythm context. A small Transformer encoder is placed after convolutional downsampling so that the model can compare pulse evidence across the full 30-second window without applying attention directly to all 3750 raw samples.

The spectral branch is included because clean pulse rhythms usually concentrate energy around plausible heart-rate frequencies, while motion, poor contact, and detector failure can produce broader or less structured spectra. The model uses `log1p(abs(STFT))` as a compact time-frequency representation and applies a small 2D CNN to learn dominant pulse bands, harmonic structure, and broadband artefact patterns.

The feature branch is included because several AF-relevant quantities are already well described mathematically by rhythm statistics. For example:

```text
SDNN = std(IBI)
RMSSD = sqrt(mean((IBI_i - IBI_{i-1})^2))
pNN50 = count(|IBI_i - IBI_{i-1}| > 50 ms) / count(IBI differences)
CV_IBI = std(IBI) / mean(IBI)
```

These features directly encode irregularity, while quality features such as template correlation and heart-band energy help the classifier decide whether the irregularity is likely physiologic or artefactual.

Finally, the fusion stage is gated rather than simple concatenation. This is appropriate for PPG because the reliability of each information source varies by segment. A clean segment may benefit strongly from waveform morphology and rhythm features, while a noisier segment may require more reliance on explicit quality and spectral evidence. The architecture is therefore selected to match the physiological and signal-processing structure of the task, not only to increase neural network capacity.

### 5.3 `RhythmMorphologyFusionNet` Overview

The offline classifier is `RhythmMorphologyFusionNet`. It is a hybrid architecture designed to combine complementary information:

1. Time-domain pulse morphology from the raw PPG waveform.
2. Frequency-domain rhythm and noise structure from a log-magnitude spectrogram.
3. Explicit pulse-rate variability and quality features from the 17-dimensional vector.

Each branch produces a 64-dimensional embedding. These embeddings are concatenated, passed through a learned gate, and classified by a multilayer head.

```text
30 s PPG waveform -> time-domain CNN + Transformer -> 64-d embedding
30 s PPG waveform -> STFT + 2D CNN                -> 64-d embedding
17 features       -> MLP feature encoder          -> 64-d embedding

Concatenate -> learned gate -> classifier head -> AF logit
```

This architecture follows the design rationale developed earlier in the project: a CNN first extracts local pulse-shape evidence, a compact sequence model captures longer-range rhythm context, and interpretable rhythm/SQI features remain available to the classifier rather than being replaced by an opaque waveform-only model. The final implementation extends that plan into a three-branch fusion model by adding an explicit spectral branch and a learned gate over waveform, frequency-domain, and hand-crafted feature evidence.

The model can be represented in the paper as a three-branch architecture diagram:

![RhythmMorphologyFusionNet architecture](figures/rhythm_morphology_fusionnet.svg)

*Figure 1. `RhythmMorphologyFusionNet` combines a time-domain morphology branch, a spectral branch, and a 17-feature rhythm/SQI branch. SQI v2 rejects unreliable windows before inference, and accepted segment probabilities are aggregated at record-event level using quality-weighted averaging.*

### 5.4 Time-Domain Branch

The time-domain branch receives the waveform as a single-channel sequence:

```text
Input shape: batch x 3750
After channel expansion: batch x 1 x 3750
```

It begins with a convolutional stem:

```text
Conv1D(1 -> 32, kernel=15, stride=2)
GroupNorm
SiLU
```

The stem is followed by three multi-scale residual blocks:

| Block | Input channels | Output channels | Stride |
|---|---:|---:|---:|
| Multi-scale block 1 | 32 | 64 | 2 |
| Multi-scale block 2 | 64 | 96 | 2 |
| Multi-scale block 3 | 96 | 128 | 2 |

Each multi-scale residual block has three parallel convolutional branches:

| Branch | Kernel size | Dilation | Purpose |
|---|---:|---:|---|
| Branch 1 | 3 | 1 | Local pulse shape |
| Branch 2 | 7 | 2 | Medium-range morphology |
| Branch 3 | 15 | 3 | Longer pulse and rhythm context |

The branch outputs are concatenated and fused with a 1x1 convolution, group normalisation, SiLU activation, and squeeze-excitation. A residual skip connection is used, with projection when stride or channel count changes.

After the convolutional stack, the waveform has been downsampled into approximately 235 temporal tokens, each with 128 channels. These tokens are passed through a two-layer Transformer encoder:

| Transformer parameter | Value |
|---|---:|
| Model dimension | 128 |
| Attention heads | 4 |
| Feed-forward dimension | 512 |
| Layers | 2 |
| Dropout | 0.1 |
| Activation | GELU |
| Normalisation style | pre-norm |

The Transformer allows the model to relate pulse morphology across the 30-second segment rather than treating each pulse independently. Attention pooling is then used to summarise the temporal tokens:

```text
attention score = Linear(Tanh(Linear(token)))
attention weights = softmax(scores over time)
time embedding = weighted sum of tokens
```

The pooled 128-dimensional representation is projected to 64 dimensions using:

```text
Linear(128 -> 64)
GELU
Dropout(0.1)
```

### 5.5 Spectral Branch

The spectral branch computes a log-magnitude short-time Fourier transform from the same waveform. The STFT uses:

| STFT parameter | Value |
|---|---:|
| FFT size | 128 |
| Window length | 128 |
| Hop length | 32 |
| Window | Hann |
| Transform | `log1p(abs(STFT))` |

This branch helps the model capture periodicity, dominant pulse-rate bands, broadband noise, and spectral irregularity. The spectrogram is processed by a compact 2D convolutional encoder:

```text
Conv2D(1 -> 16, kernel=5x5), GroupNorm, GELU, MaxPool2D
Conv2D(16 -> 32, kernel=3x3), GroupNorm, GELU, MaxPool2D
Conv2D(32 -> 64, kernel=3x3), GroupNorm, GELU
AdaptiveAvgPool2D(1x1)
Flatten
Linear(64 -> 64)
GELU
Dropout(0.1)
```

The output is a 64-dimensional spectral embedding.

### 5.6 Feature Branch

The feature branch embeds the 17 hand-crafted rhythm and quality features using a two-layer multilayer perceptron:

```text
Linear(17 -> 64)
LayerNorm(64)
GELU
Dropout(0.1)
Linear(64 -> 64)
GELU
```

This branch gives the model direct access to clinically interpretable rhythm variability information, such as RMSSD, pNN50, and sample entropy, as well as signal-quality cues such as template correlation and heart-band energy.

### 5.7 Gated Fusion and Classifier Head

The three 64-dimensional embeddings are concatenated:

```text
h = [h_time, h_spectral, h_feature]
shape: batch x 192
```

A learned gate is computed from the concatenated representation:

```text
g = sigmoid(Linear(GELU(Linear(h))))
```

The 192-dimensional gate is split into three 64-dimensional gates and applied separately:

```text
h_gated = [h_time * g_time, h_spectral * g_spectral, h_feature * g_feature]
```

The purpose of this gating mechanism is to let the model change the relative contribution of waveform morphology, spectral evidence, and hand-crafted rhythm features on a per-sample basis. For example, if the waveform branch is less reliable but the rhythm features are stable, the model can place more weight on the feature embedding.

The final classifier head is:

```text
Linear(192 -> 128)
LayerNorm(128)
GELU
Dropout(0.2)
Linear(128 -> 64)
GELU
Dropout(0.15)
Linear(64 -> 1)
```

The output is a single AF logit.

### 5.8 Training Objective

The final large-scale model uses `QualityAwareFocalLoss`, which combines binary cross-entropy with positive-class weighting, focal modulation, label smoothing, and signal-quality weighting.

The label smoothing step is:

```text
y_smooth = y * (1 - 0.02) + 0.5 * 0.02
```

The base loss is binary cross-entropy with logits and positive-class weighting. Focal modulation is applied with:

```text
focal_factor = (1 - p_t) ^ gamma
gamma = 1.5
```

The segment quality score is then used as a sample weight:

```text
sample_weight = 0.6 + 0.4 * quality_score
```

This means that all accepted segments can contribute to training, but higher-quality segments contribute slightly more strongly.

### 5.9 Class Imbalance Handling

The final SQI v2 run uses a weighted random sampler. The raw training split contains 84,589 AF segments and 537,476 SR segments after SQI acceptance. Instead of training on this raw class ratio, the sampler targets an AF fraction of 0.333333, corresponding approximately to an AF:SR ratio of 1:2 in sampled mini-batches.

This choice was made after observing that strict balancing did not automatically improve performance. A 1:1 sampler can reduce class bias, but it also reduces exposure to diverse SR examples. In an AF screening task, the negative class contains a wide range of normal and artefact-affected pulse patterns. Retaining more SR diversity can improve specificity and precision. The final training configuration therefore uses class-aware sampling rather than complete 1:1 undersampling.

The final training configuration is:

| Parameter | Value |
|---|---:|
| Device | CUDA |
| Epoch limit | 30 |
| Early stopping patience | 10 |
| Epochs run | 18 |
| Best epoch | 8 |
| Batch size | 32 |
| Learning rate | 3e-4 |
| Weight decay | 5e-5 |
| Optimiser | AdamW |
| Scheduler | CosineAnnealingLR |
| Loss | Quality-aware focal loss |
| Focal gamma | 1.5 |
| Label smoothing | 0.02 |
| Balanced sampler | yes |
| Sampler AF fraction | 0.333333 |
| Mixup | disabled |
| Automatic mixed precision | disabled |
| Threshold objective | validation record-level F1 |

### 5.10 Data Augmentation and Test-Time Augmentation

During training, waveform augmentation is available for the training split. The implemented augmentations include amplitude scaling, Gaussian noise, circular time shift, low-frequency drift, local masking, and mild time warping. These are intended to reduce overfitting to exact waveform shape and improve robustness to realistic PPG variability.

During evaluation, test-time augmentation averages predictions over three temporal shifts:

```text
shifts = (0, -8, 8)
```

This makes the prediction less sensitive to very small alignment changes in the 30-second window.

### 5.11 Record-Level Aggregation

The model produces segment-level probabilities, but the primary evaluation is record-level. Segment probabilities belonging to the same `record_id+event_id` group are aggregated using a quality-weighted average:

```text
record_probability = weighted_mean(segment_probabilities, quality_scores)
```

If all quality weights are zero, the arithmetic mean is used. This aggregation is important because an AF screening alert should be based on an episode or recording rather than a single isolated 30-second prediction. It also reduces the effect of occasional false-positive windows.

### 5.12 Threshold Selection

The final decision threshold is selected using validation record-level F1. For the final SQI v2 model, the selected threshold is:

```text
threshold = 0.713
```

This threshold is fixed before test evaluation. Test-set threshold sweeps are reported only as analysis and are not used as the primary result.

## 6. Evaluation Protocol

### 6.1 Primary and Secondary Evaluation Levels

The primary evaluation is record-level because this better reflects an AF screening workflow. Segment-level evaluation is also reported because it is useful for diagnosing model behaviour on individual 30-second windows.

| Level | Definition | Role in paper |
|---|---|---|
| Segment-level | Each 30-second window is classified independently | Secondary diagnostic analysis |
| Record-level | Segment probabilities are aggregated per `record_id+event_id` | Primary performance result |

### 6.2 Metrics

The reported metrics are:

| Metric | Meaning |
|---|---|
| Accuracy | Fraction of all predictions that are correct |
| Sensitivity / recall | Fraction of AF examples detected |
| Specificity | Fraction of SR examples correctly rejected |
| Precision / PPV | Fraction of positive predictions that are true AF |
| F1 | Harmonic mean of precision and sensitivity |
| AUROC | Ranking performance across thresholds |
| AUPRC | Precision-recall ranking performance, important under imbalance |

Because the task is imbalanced, accuracy is interpreted cautiously. The headline model-selection metric is record-level F1, because it balances AF detection sensitivity against the false-alert burden captured by precision. Sensitivity, specificity, precision, AUROC, and AUPRC are reported alongside F1 to show the operating-point trade-off and threshold-independent ranking performance.

### 6.3 Reproducibility Artifacts

The final SQI v2 run saved artifacts to:

```text
/vol/bitbucket/mc1920/mimic_ext_p00_p01_p02_sqi_v2_ppg_1to2_20260426_194112
```

Key output files include:

```text
metrics.json
best_model.pt
training_history.csv
val_segment_predictions.csv
test_segment_predictions.csv
val_record_predictions.csv
test_record_predictions.csv
val_segment_threshold_sweep.csv
test_segment_threshold_sweep.csv
val_record_threshold_sweep.csv
test_record_threshold_sweep.csv
```

## 7. Results

### 7.1 Controlled Baseline: MIMIC PERform AF

The MIMIC PERform AF baseline achieved perfect record-level results on its five-record test split. Segment-level results were also high, but not perfect because individual 30-second windows are noisier than record-level averages.

| Evaluation level | Accuracy | Sensitivity | Specificity | Precision | F1 | AUROC | AUPRC |
|---|---:|---:|---:|---:|---:|---:|---:|
| Segment-level test | 0.9447 | 1.0000 | 0.8625 | 0.9154 | 0.9558 | 0.9980 | 0.9986 |
| Record-level test | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

Record-level confusion matrix:

| | Predicted SR/non-AF | Predicted AF |
|---|---:|---:|
| True SR/non-AF | 2 | 0 |
| True AF | 0 | 3 |

Segment-level confusion matrix at threshold 0.170:

| | Predicted SR/non-AF | Predicted AF |
|---|---:|---:|
| True SR/non-AF | 69 | 11 |
| True AF | 0 | 119 |

These results confirm that the signal processing and hybrid model are capable of identifying AF when the data are relatively controlled and binary. However, the dataset is too small for the result to be considered the main evidence.

### 7.2 Large-Scale Experiment Comparison

The main model-development comparison is performed on MIMIC-III-Ext-PPG p00-p02. Record-level metrics are used for the headline comparison.

| Experiment | Dataset / Scope | Threshold | Accuracy | Sensitivity | Specificity | Precision | F1 | AUROC | AUPRC |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Controlled baseline | MIMIC PERform AF | 0.170 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Initial large-scale hybrid | MIMIC-III-Ext-PPG p00-p02 | 0.516 | 0.9435 | 0.8454 | 0.9582 | 0.7519 | 0.7959 | 0.9229 | 0.8165 |
| Class-aware 1:2 sampler | MIMIC-III-Ext-PPG p00-p02 | 0.781 | 0.9514 | 0.8410 | 0.9680 | 0.7973 | 0.8186 | 0.9467 | 0.8311 |
| Final SQI v2 pipeline | MIMIC-III-Ext-PPG p00-p02 | 0.713 | 0.9536 | 0.8241 | 0.9729 | 0.8194 | 0.8217 | 0.9272 | 0.8365 |

The final SQI v2 pipeline gives the strongest record-level F1 and AUPRC among the main p00-p02 runs. Compared with the initial large-scale hybrid experiment, it improves accuracy, specificity, precision, F1, and AUPRC, while sensitivity decreases from 0.8454 to 0.8241. This reflects a more conservative operating point that reduces false positives.

Additional exploratory record-level architectures were tested but not retained:

| Exploratory model | Record-level F1 | AUROC | AUPRC | Interpretation |
|---|---:|---:|---:|---|
| Hierarchical record model | 0.7849 | 0.9486 | 0.7459 | Did not improve over SQI-aware hybrid |
| Record MIL | 0.7558 | 0.9471 | 0.7611 | Lower F1 and precision |
| Record MIL v2 | 0.7864 | 0.9545 | 0.7799 | Better AUROC, lower F1 than final SQI v2 |

### 7.3 Final SQI v2 Validation and Test Results

The final SQI v2 model stopped early after 18 epochs, with the best validation epoch at epoch 8. The validation-selected threshold was 0.713.

Record-level results:

| Split | Accuracy | Sensitivity | Specificity | Precision | F1 | AUROC | AUPRC |
|---|---:|---:|---:|---:|---:|---:|---:|
| Validation | 0.9782 | 0.8736 | 0.9975 | 0.9849 | 0.9259 | 0.9823 | 0.9590 |
| Test | 0.9536 | 0.8241 | 0.9729 | 0.8194 | 0.8217 | 0.9272 | 0.8365 |

Segment-level results:

| Split | Accuracy | Sensitivity | Specificity | Precision | F1 | AUROC | AUPRC |
|---|---:|---:|---:|---:|---:|---:|---:|
| Validation | 0.9744 | 0.8329 | 0.9923 | 0.9316 | 0.8795 | 0.9698 | 0.9122 |
| Test | 0.9526 | 0.8843 | 0.9594 | 0.6841 | 0.7714 | 0.9525 | 0.7947 |

There is a visible validation-test gap, especially at record level where F1 decreases from 0.9259 to 0.8217 and AUPRC decreases from 0.9590 to 0.8365. This gap is important to report because the test fold has lower AF prevalence, different record-event composition, and different prefix mix from the validation fold. The final result should therefore be interpreted as promising performance on the held-out p00-p02 test fold, not as proof of uniform generalisation across the full MIMIC-III-Ext-PPG release.

### 7.3 Ablation Study

Three targeted ablation analyses were used to isolate the contributions of SQI, branch inputs, and aggregation.

1. Feature-level SQI ablation with weighted logistic regression showed that SQI filtering modestly improves a rhythm-only baseline, increasing record-level F1 from 0.8137 to 0.8177. Quality-weighted aggregation also gave a small gain compared with simple averaging for the 17-feature logistic baseline (F1 0.7775 vs 0.7764). Adding explicit SQI-related features to a linear model did not automatically improve the operating point; the all-feature logistic variants had similar AUROC/AUPRC but lower precision and F1 than the rhythm-only baseline.

2. Neural branch ablation retrained four variants on the same p00-p02 SQI-accepted split: full fusion, waveform-only, spectral-only, and feature-only neural models. The waveform-only model produced the strongest validation-selected operating point among these variants, with record-level F1 0.8072. The spectral-only and feature-only variants performed worse at the selected thresholds, and the independently retrained full-fusion variant did not exceed the waveform-only model in this run.

3. Aggregation ablation compared record-level pooling rules on the final retained pipeline. Quality-weighted averaging yielded the highest test F1 at 0.8217, slightly above unweighted mean (0.8205), median (0.8112), and maximum probability pooling (0.7803). This supports the design choice of aggregating repeated evidence and using quality as a modest reliability weight.

Taken together, the ablation results support three conclusions: quality-aware record-level aggregation is beneficial, SQI is useful primarily as a coverage and reliability mechanism rather than as an automatic linear feature boost, and learned waveform morphology is the strongest single neural input in the current experimental setup. At the same time, the evidence does not yet prove that gated multi-branch fusion is consistently superior, so future work should include repeated-seed branch ablations and a full neural SQI ablation comparison.

### 7.5 Confusion Matrices

Record-level test confusion matrix at threshold 0.713:

| | Predicted SR | Predicted AF |
|---|---:|---:|
| True SR | 4488 | 125 |
| True AF | 121 | 567 |

Segment-level test confusion matrix at threshold 0.713:

| | Predicted SR | Predicted AF |
|---|---:|---:|
| True SR | 65090 | 2757 |
| True AF | 781 | 5971 |

The record-level confusion matrix explains the final positive-class metrics:

```text
Precision = 567 / (567 + 125) = 0.8194
Sensitivity = 567 / (567 + 121) = 0.8241
Specificity = 4488 / (4488 + 125) = 0.9729
F1 = 0.8217
```

### 7.6 Threshold Analysis

The primary threshold is selected on validation data. A test-set threshold sweep is included only to understand the trade-off between sensitivity and precision.

| Threshold | Source | Accuracy | Sensitivity | Specificity | Precision | F1 | FP | FN | TP |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.713 | Validation-selected | 0.9536 | 0.8241 | 0.9729 | 0.8194 | 0.8217 | 125 | 121 | 567 |
| 0.834 | Test-best analysis only | 0.9587 | 0.7907 | 0.9837 | 0.8788 | 0.8324 | 75 | 144 | 544 |

The higher test-best threshold would reduce false positives from 125 to 75 and increase precision from 0.8194 to 0.8788, but it would also increase false negatives from 121 to 144 and reduce sensitivity from 0.8241 to 0.7907. This confirms that the operating point should be chosen according to intended use.

### 7.7 Per-Prefix Behaviour

Performance differs across p00, p01, and p02:

| Prefix | Records | AF records | Accuracy | Sensitivity | Specificity | Precision | F1 | AUROC | AUPRC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| p00 | 1761 | 180 | 0.9302 | 0.8278 | 0.9418 | 0.6183 | 0.7078 | 0.8982 | 0.7413 |
| p01 | 1555 | 38 | 0.9794 | 0.6053 | 0.9888 | 0.5750 | 0.5897 | 0.9719 | 0.5783 |
| p02 | 1985 | 470 | 0.9542 | 0.8404 | 0.9894 | 0.9611 | 0.8967 | 0.9350 | 0.9134 |

p02 performs best, with F1 0.8967 and precision 0.9611. p01 is the hardest subset: it contains only 38 AF-positive record-event groups in the test split, and sensitivity falls to 0.6053. This prefix-level variability is an important finding rather than only a weakness: it shows that ICU-derived PPG performance depends on data composition and that aggregate p00-p02 performance should not be interpreted as uniform behaviour across all subsets.

## 8. Embedded Edge AI Deployment

After evaluating the offline PPG models, the project also investigates whether the feature-based part of the pipeline can be connected to an embedded inference workflow. This chapter is placed after the performance results because the deployment work is a downstream implementation step rather than the primary source of model-performance evidence.

### 8.1 Firmware Target

The embedded component uses the `E2_V3_NCS2.3.0_60115_Copy` firmware project, a Zephyr/nRF Connect SDK application targeting an nRF9160-class device. The firmware includes modules for display, GPS, UART/BLE, power management, IMU support, and PPG support. The Edge AI integration is included through:

```text
src/test2_92249_v3
src/test2_92249_v3/nrf_edgeai_generated
src/test2_92249_v3/nrf_edgeai/lib/libnrf_edgeai_cortex-m33.a
src/test2_92249_v3/edgeai_smoke/edgeai_smoke.c
```

The top-level firmware CMake file adds the module with:

```cmake
add_subdirectory(src/test2_92249_v3)
```

### 8.2 Nordic Edge AI Lab Model

The generated Nordic Edge AI Lab model has solution ID `92249`. The generated header exposes:

```c
nrf_edgeai_t* nrf_edgeai_user_model_92249(void);
uint32_t nrf_edgeai_user_model_neuton_size_92249(void);
```

The generated code aliases this to the standard `nrf_edgeai_user_model()` API. The model is a Neuton CPU model, not an Axon/NPU model.

The compact deployed model uses:

| Property | Value |
|---|---:|
| Input features | 17 float32 values |
| Input window size | 1 |
| Output classes | 2 |
| Internal model neurons | 4 |
| Model weights | 16 |

This deployment model is deliberately smaller than the offline hybrid model. It demonstrates an embedded inference path using the same feature representation, while the larger hybrid waveform model remains the main offline performance model.

### 8.3 Embedded Smoke Test

The firmware includes an embedded smoke-test thread. At startup, the thread waits two seconds, obtains the Edge AI model pointer, checks the expected input count, calls `nrf_edgeai_init()`, and then feeds holdout samples using:

```c
nrf_edgeai_feed_inputs(p_edgeai, sample_buf, expected_inputs);
nrf_edgeai_run_inference(p_edgeai);
```

The holdout CSV contains 2,249 samples with 17 feature columns and one binary label column. The build system converts the CSV into a C header using `csv_to_edgeai_header.py`, storing the data as const arrays in flash. The smoke test reports matched samples, total samples, accuracy, mismatch count, and the first mismatched rows through Zephyr logging.

This checks that the generated model, feature order, runtime library, CMake integration, and firmware inference calls are consistent.

### 8.4 Deployment Result and Interpretation

The embedded deployment component successfully integrates a compact 17-feature Nordic Edge AI Lab model into the Zephyr firmware project. The current result is a firmware-level smoke-test harness rather than a full hardware validation study. It confirms that:

- the generated Edge AI model is available through the expected C API,
- the model expects 17 float32 input features,
- the holdout dataset can be converted into a flash-resident C header,
- the firmware can initialise the runtime, feed input vectors, execute inference, and compare predicted classes against labels.

This supports the feasibility of on-device inference, but the deployed compact model should be presented separately from the offline hybrid model.

## 9. Discussion

### 9.1 Main Finding

The key finding is that SQI-aware PPG processing can support promising record-level AF detection on a large ICU-derived PPG subset. The final MIMIC-III-Ext-PPG p00-p02 result has high specificity (0.9729), good precision (0.8194), and balanced F1 (0.8217) at the validation-selected threshold. This is encouraging for screening because high specificity reduces unnecessary alerts, while sensitivity above 0.82 indicates that most AF record-event groups are detected.

The result also shows the limits of small controlled baselines. MIMIC PERform AF achieved perfect record-level test performance, but that result is based on only five test records. It is useful as a sanity check, not as the main claim. The larger p00-p02 subset reveals the more realistic challenges: imbalance, heterogeneous ICU physiology, variation between prefixes, and sensitivity-specificity trade-offs.

### 9.2 Why Strict 1:1 Balancing Did Not Necessarily Improve Results

The supervisor feedback to balance AF and non-AF data is still conceptually correct: a heavily imbalanced training set can bias the model toward the majority class. However, "balanced" does not mean that every final experiment must discard all extra non-AF samples or that performance will automatically improve.

There are three reasons:

1. The negative class is diverse. SR segments include many morphologies, heart rates, noise levels, and artefact patterns. Removing too many SR examples can reduce the model's ability to reject false positives.
2. Validation and test sets should reflect realistic imbalance. If validation is forced to 1:1, threshold selection may not match deployment conditions.
3. Metrics respond differently. A stricter balance may improve sensitivity but reduce precision or specificity. In this project, the retained 1:2 sampler preserved more SR diversity while still increasing AF exposure during training.

Therefore, the final paper should describe balancing as an optimisation strategy, not as a guarantee of better F1 or accuracy.

### 9.3 Segment-Level and Record-Level Metrics Tell Different Stories

Segment-level test sensitivity is higher than record-level sensitivity (0.8843 versus 0.8241), but segment-level precision is much lower (0.6841 versus 0.8194). This occurs because individual 30-second windows are noisy and the segment-level AF prevalence in the test split is only 9.05%. Even a modest number of false-positive SR windows can reduce segment-level precision.

Record-level aggregation improves practical interpretability. By averaging probabilities across a record-event group and weighting by quality, isolated false-positive windows have less influence. This is closer to how a screening system would operate, because alerts should be based on repeated or episode-level evidence rather than a single local window.

### 9.4 Interpretation of Precision and Specificity

At the selected threshold, record-level specificity is high. The model correctly rejects 4488 of 4613 SR record-event groups. This is important because false-positive AF alerts can create unnecessary anxiety and clinical workload.

The cost is that 121 of 688 AF record-event groups are missed. Whether this operating point is acceptable depends on deployment context. For user-facing alerts, higher precision and specificity may be preferred. For clinician-facing retrospective review, a lower threshold may be acceptable to increase sensitivity.

### 9.5 Role of SQI

SQI contributes at three stages:

1. It filters unreliable segments before training and evaluation.
2. It contributes explicit quality features to the model.
3. It weights both the loss and record-level aggregation.

This is important because AF-like irregularity can be produced by signal artefact. The model should not be rewarded for detecting irregularity in segments where the pulse train itself is unreliable. SQI helps separate physiologic irregularity from measurement failure.

### 9.6 Edge Deployment Interpretation

The Edge AI integration demonstrates a practical deployment pathway but does not yet reproduce the full offline model on-device. The offline model uses waveform CNNs, a Transformer, STFT processing, and gated fusion. The embedded model uses only the 17-feature vector and a compact Neuton CPU classifier. This is a reasonable engineering split: the offline model explores performance, while the embedded model demonstrates feasibility under device constraints.

The next deployment step is to align the embedded feature extraction code with the offline Python pipeline, then test end-to-end inference on streaming PPG from the target hardware.

### 9.7 Robustness and Error Analysis Gaps

The current results are strongest as an end-to-end feasibility study, but they do not yet isolate every design choice. The most useful next analyses would be:

1. SQI ablation comparing no SQI, SQI as features only, SQI filtering only, and the full SQI v2 pipeline.
2. Branch ablation comparing feature-only, waveform-only, spectral-only, and full gated fusion models.
3. Aggregation ablation comparing unweighted mean, maximum probability, and quality-weighted mean.
4. False-positive and false-negative review by prefix, signal quality, AF prevalence, and segment count.
5. Specific analysis of p01 to determine whether its lower F1 is driven by low AF support, label timing, signal quality, patient mix, or waveform distribution shift.

These analyses would strengthen the claim from feasibility toward robustness. Without them, the safest interpretation is that the method is promising on the evaluated p00-p02 subset but still needs broader validation.

## 10. Limitations

This project has several limitations.

First, the large-scale experiment uses p00-p02 rather than the full p00-p09 MIMIC-III-Ext-PPG release. The selected subset is already large, but full-prefix evaluation would provide stronger evidence and may reveal additional prefix-level variation.

Second, the binary task uses AF versus SR. This is interpretable, but real screening systems must also handle atrial flutter, ectopy, pacing, tachycardia, bradycardia, and other rhythms that may resemble or confound AF.

Third, rhythm labels in MIMIC-III-Ext-PPG are derived from charted clinical rhythm events and associated waveform windows. They are not prospective smartwatch-style annotations. Although the dataset follows a careful annotation strategy, label noise and timing mismatch remain possible.

Fourth, the SQI thresholds are hand-designed. They worked well enough for this project, but optimal thresholds may differ across devices, patient populations, perfusion states, and movement contexts. This limits how strongly the current SQI v2 settings can be claimed to generalise outside the evaluated dataset.

Fifth, the model is evaluated retrospectively. A deployed screening system would require prospective testing, user-state context, confirmatory ECG, and evaluation of alert burden.

Sixth, subgroup performance is not fully analysed. The dataset includes demographic metadata, but this draft does not yet quantify performance across age, sex, ethnicity, diagnosis group, prefix, or signal-quality strata.

Seventh, the embedded model is not the same as the offline hybrid model. The deployment section therefore demonstrates feasibility rather than proving that the full offline performance can be achieved on the target hardware.

## 11. Future Work

Future work should prioritise:

1. Full p00-p09 evaluation to test whether the final p00-p02 result extends across the full MIMIC-III-Ext-PPG release.
2. Ablation studies for SQI, model branches, and record-level aggregation to isolate which components provide the largest gains.
3. Error analysis of false positives and false negatives by prefix, signal quality, AF prevalence, and segment count.
4. Multi-class rhythm classification, including atrial flutter, pacing, AV block, tachycardia, and bradycardia.
5. Soft SQI modelling, where low-quality segments are down-weighted rather than always rejected.
6. Learned SQI models trained to distinguish motion artefact from physiologic arrhythmia.
7. Threshold calibration for different operating modes, such as high-sensitivity clinician review and high-specificity user alerts.
8. Prefix/domain adaptation to improve weaker subsets such as p01.
9. Demographic and clinical subgroup analysis to assess fairness and robustness.
10. Embedded feature extraction in C so that the Nordic firmware can process live PPG rather than precomputed feature vectors.
11. Prospective device-specific validation with synchronised ECG confirmation.

## 12. Conclusion

This project developed a PPG-only AF screening pipeline combining signal processing, SQI filtering, hybrid neural classification, record-level aggregation, and embedded Edge AI deployment feasibility work. MIMIC PERform AF confirmed that the approach works under controlled binary conditions, while MIMIC-III-Ext-PPG p00-p02 provided the main large-scale subset evaluation.

The final SQI v2 model achieved record-level test accuracy 0.9536, sensitivity 0.8241, specificity 0.9729, precision 0.8194, F1 0.8217, AUROC 0.9272, and AUPRC 0.8365. These results support the feasibility of SQI-aware PPG AF screening on the evaluated p00-p02 subset and show that record-level aggregation provides a more useful screening summary than isolated segment decisions.

The project also demonstrates a deployment pathway through a compact Nordic Edge AI model integrated into Zephyr firmware. This embedded component is a compact feature-based feasibility demonstration, not an on-device reproduction of the full offline hybrid model. The final system should be understood as a screening pipeline: promising for AF risk detection from PPG, but requiring confirmatory ECG, full-prefix evaluation, prospective validation, and further robustness analysis before clinical use.

## References

[1] Goldberger, A. L., Amaral, L. A. N., Glass, L., Hausdorff, J. M., Ivanov, P. C., Mark, R. G., et al. (2000). PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals. Circulation, 101(23), e215-e220.

[2] Johnson, A. E. W., Pollard, T. J., Shen, L., Lehman, L. W. H., Feng, M., Ghassemi, M., Moody, B., Szolovits, P., Celi, L. A., & Mark, R. G. (2016). MIMIC-III, a freely accessible critical care database. Scientific Data, 3, 160035.

[3] Moody, B., Moody, G., Villarroel, M., Clifford, G., & Silva, I. (2020). MIMIC-III Waveform Database Matched Subset v1.0. PhysioNet.

[4] Moulaeifard, M., Charlton, P. H., & Strodthoff, N. (2026). MIMIC-III-Ext-PPG: A PPG Benchmark Dataset for Cardiorespiratory Analysis (version 1.1.0). PhysioNet. RRID:SCR_007345. https://doi.org/10.13026/r6k1-xt76

[5] Moulaeifard, M., Kutscher, M., Aston, P. J., et al. (2026). MIMIC-III-Ext-PPG, a PPG-based Benchmark Dataset for Cardiovascular and Respiratory Signal Analysis. Scientific Data, 13, 668. https://doi.org/10.1038/s41597-026-07335-8

[6] Charlton, P. H., Bonnici, T., Tarassenko, L., Clifton, D. A., Beale, R., Watkinson, P. J., & Alastruey, J. (2021). An impedance pneumography signal quality index: Design, assessment and application to respiratory rate monitoring. Biomedical Signal Processing and Control, 65, 102339.

[7] Makowski, D., Pham, T., Lau, Z. J., Brammer, J. C., Lespinasse, F., Pham, H., Scholzel, C., & Chen, S. A. (2021). NeuroKit2: A Python toolbox for neurophysiological signal processing. Behavior Research Methods, 53, 1689-1696.

[8] Orphanidou, C., Bonnici, T., Charlton, P., Clifton, D., Vallance, D., & Tarassenko, L. (2015). Signal-quality indices for the electrocardiogram and photoplethysmogram: Derivation and applications to wireless monitoring. IEEE Journal of Biomedical and Health Informatics, 19(3), 832-838.

[9] Sun, J. X. (2006). Cardiac output estimation using arterial blood pressure waveforms. Doctoral dissertation, Massachusetts Institute of Technology.

[10] World Health Organization. (1992). International Statistical Classification of Diseases and Related Health Problems, 10th Revision (ICD-10).

[11] Elgendi, M. (2012). On the analysis of fingertip photoplethysmogram signals. Current Cardiology Reviews, 8(1), 14-25.

[12] Charlton, P. H. (2022). MIMIC PERform Datasets. https://doi.org/10.5281/zenodo.6950488

[13] Charlton, P. H., et al. (2022). Detecting beats in the photoplethysmogram: benchmarking open-source algorithms. Physiological Measurement. https://doi.org/10.1088/1361-6579/ac826d

[14] Nordic Semiconductor. Nordic Edge AI Lab and nRF Connect SDK documentation.

## Appendix A: Suggested Figures and Tables

Suggested figures:

1. System overview: PPG input -> filtering -> peak detection -> SQI -> hybrid classifier -> segment probability -> record aggregation -> alert.
2. SQI gate diagram showing accepted and rejected segment logic.
3. Offline model architecture showing time-domain branch, spectral branch, feature branch, gated fusion, and classifier head.
4. Record-level threshold sweep showing precision, sensitivity, specificity, and F1 versus threshold.
5. Embedded deployment diagram showing feature vector -> Edge AI Lab model -> Zephyr firmware -> smoke test.

Suggested tables:

1. MIMIC-III-Ext-PPG full dataset summary and p00-p02 subset summary.
2. SQI v2 acceptance criteria.
3. Hybrid model architecture.
4. Controlled baseline results.
5. Large-scale experiment comparison.
6. Final record-level and segment-level metrics.
7. Confusion matrices.
8. Per-prefix test behaviour.

## Appendix B: Ethical Considerations and Data Governance

MIMIC-III-Ext-PPG is derived entirely from MIMIC-III. According to the dataset documentation, MIMIC-III was released under Institutional Review Board oversight at Beth Israel Deaconess Medical Center and the Massachusetts Institute of Technology, with HIPAA-compliant de-identification. Data were included under an IRB-approved waiver of informed consent because the source data are retrospective and de-identified. MIMIC-III-Ext-PPG does not introduce new patient contact or new identifiable data; it reprocesses and curates existing MIMIC-III waveform and clinical records.

The files are credentialed PhysioNet health data. Access requires signing the appropriate Data Use Agreement and completing required training. The project therefore treats the data as sensitive clinical research data even though they are de-identified. Results are reported only in aggregate, and no attempt is made to re-identify patients.

There are also ethical considerations specific to AF screening. False positives could create anxiety or unnecessary clinical follow-up. False negatives could delay detection. Because PPG is a screening signal rather than a diagnostic ECG, the intended use of this work is risk indication and triage, not standalone diagnosis. Any deployed system should communicate uncertainty, require confirmatory ECG, and be evaluated prospectively before clinical use.

Potential bias is another concern. PPG signal quality can vary with skin tone, perfusion, comorbidities, device placement, and motion context. MIMIC-III-Ext-PPG includes ICU patients and demographic metadata, but this project does not yet perform a full subgroup fairness analysis. That is a limitation and an important direction for future work.

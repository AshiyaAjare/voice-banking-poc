# Model Comparison: SpeechBrain vs. WeSpeaker

This document details the differences between the three primary models currently used in the VoicePrint system.

## 1. SpeechBrain Models

SpeechBrain is an open-source, all-in-one speech toolkit based on PyTorch. In VoicePrint, it accounts for two distinct functions using the ECAPA-TDNN architecture.

### A. Speaker Recognition (Primary Model)
- **Model**: `speechbrain/spkrec-ecapa-voxceleb`
- **Architecture**: **ECAPA-TDNN** (Emphasized Channel Attention, Propagation and Aggregation in Time Delay Neural Networks).
- **Core Mechanism**:
  - Uses 1D Convolutions over the time domain.
  - Incorporates **Channel Attention** mechanisms to focus on important features.
  - Aggregates multi-scale features (propagation) to capture both local and global context.
- **Input**: Raw audio waveform.
- **Output**: 192-dimensional speaker embedding.
- **Training Data**: VoxCeleb1 and VoxCeleb2.
- **Use Case**: This is the **primary** model for generating voice prints (embeddings) and verifying identity. It is optimized for channel robustness and performance in varying acoustic environments.

### B. Language Identification (LID)
- **Model**: `speechbrain/lang-id-voxlingua107-ecapa`
- **Architecture**: **ECAPA-TDNN** (Same backbone as the speaker model).
- **Core Mechanism**:
  - While it uses the same neural architecture as the speaker model, the "head" (final layers) is trained to classify languages instead of speakers.
  - It does **not** output a speaker vector; it outputs a probability distribution over languages.
- **Input**: Raw audio waveform ($16\text{kHz}$).
- **Output**: Language label (one of 107 languages) and confidence score.
- **Training Data**: VoxLingua107.
- **Use Case**: Used strictly as a **control-plane** signal in `LanguageDetectionService`. It routes requests or applies policy (e.g., verifying if the spoken language matches the user's profile) but does not influence the voice biometric embedding itself.

---

## 2. WeSpeaker Model (Comparison/Secondary)

WeSpeaker is a production-oriented speaker verification toolkit heavily focused on ResNet-based architectures.

- **Model**: **eres2net34_voxceleb** (configured via `pretrained_models/wespeaker/config.yaml`).
- **Architecture**: **ERes2Net34** (ResNet-based with multi-scale Res2Net feature splitting).
- **Core Mechanism**:
  - Converts raw audio into log **Mel filterbank (Fbank) features**.
  - Uses 2D convolutions over time–frequency representations
  - **TSTP** (Time-Dependent Statistics Pooling): Aggregates frame-level features into a single utterance-level vector.
  - **ArcMargin Loss**: A specialized loss function that maximizes the angular margin between classes, significantly improving discrimination between speakers.
- **Input**: Mel-spectrograms (computed from raw audio).
- **Output**: 256-dimensional speaker embedding.
- **Training Data**: VoxCeleb2.
- **Use Case**: Used for **A/B Testing** and comparison.
  - ResNet architectures often capture different feature sets compared to TDNNs.
  - It serves as a "challenger" model to validate the performance of the primary ECAPA-TDNN model.
  - It runs in parallel (if enabled) to provide a secondary similarity score.

## Summary Table

| Feature | SpeechBrain (Speaker) | SpeechBrain (LID) | WeSpeaker |
| :--- | :--- | :--- | :--- |
| **Task** | Speaker Verification | Language ID | Speaker Verification |
| **Architecture** | **ECAPA-TDNN** | **ECAPA-TDNN** | **ResNet34** |
| **Input** | Raw Waveform | Raw Waveform | Mel-Spectrogram |
| **Processing** | 1D Conv + Attention | 1D Conv + Attention | 2D Conv (Image-like) |
| **Embedding Dim** | 192 | N/A (Classes) | 256 |
| **Loss Function** | AAM-Softmax | N/A | ArcMargin |
| **Role** | **Primary** | **Control / Policy** | **Secondary / Comparison** |

# VoicePrint System Design & Architecture

This document describes the architecture, data flow, and speaker verification strategies used in the VoicePrint backend.

## 1. System Overview
VoicePrint is a robust speaker verification system designed for multilingual environments (specifically Indian accents). It utilizes deep learning models from the SpeechBrain toolkit and implements a "Language Bucket" strategy to ensure high accuracy across multiple languages for a single user.

### High-Level Architecture
1.  **API Layer (FastAPI)**: Handles XOR-encrypted audio streams and manages user enrollment/verification sessions.
2.  **Security Layer (XOR Cipher)**: Symmetric encryption for audio data in transit.
3.  **Identity Plane (ECAPA-TDNN + X-Vector)**: Core biometric engine for extracting and comparing speaker identity.
4.  **Control Plane (Silero VAD + SpeechBrain LID)**: Auxiliary signals for voice quality and language routing.

---

## 2. Core Components

### 🛡️ XOR Cipher (Audio Security)
All raw audio bytes transferred to the API are encrypted at the source (Frontend) and decrypted at the API boundary using a shared XOR key. This ensures that biometric voice data is never sent in plain text.

### 🌬️ Silero VAD (Voice Activity Detection)
Before processing, every audio sample passes through Silero VAD to ensure:
- Minimum speech duration (e.g., 500ms) is present.
- Silence/background noise is filtered out.
- The user is actually speaking.

### 🌍 SpeechBrain LID (Language Identification)
The system uses the `speechbrain/lang-id-voxlingua107-ecapa` model for control-plane logic.
- **Purpose**: Detect the spoken language.
- **Invariant**: LID is **strictly advisory**. It never modifies audio, generates identity embeddings, or lowers similarity thresholds. It is only used to select the appropriate language bucket or order verification checks.

### 👤 Speaker Identity Models
- **ECAPA-TDNN (Primary)**: The state-of-the-art model for speaker recognition. It receives **raw audio** to extract d-vector embeddings.
- **X-Vector (Secondary)**: An optional secondary model used for score fusion to increase confidence in high-security scenarios.

---

## 3. Enrollment Flow (Multi-Language)

The system avoids "phonetic bias" (where different speakers speaking the same language are confused) by using **Language Buckets**.

### Step 1: Initialization
The user declares their languages (Primary, Secondary, Optional). A profile is created in `storage/embeddings/user_id/profile.json`.

### Step 2: Sample Collection
For each language, the user provides multiple samples (default: 3).
- Audio is decrypted.
- VAD checks for quality.
- LID validates the spoken language against the target bucket.
- **ECAPA-TDNN** extracts identity features from raw waveform.

### Step 3: Centroid Calculation
Once enough samples are collected for a bucket, the embeddings are averaged to create a **Centroid Profile** (.pt file). This centroid represents the stable "voice print" for that user in that specific language.

---

## 4. Verification Flow & Strategies

When a user attempts to verify, the system must decide which of their enrolled language buckets to compare against.

### Verification Process
1.  **Audio In**: Raw audio is received and decrypted.
2.  **LID Check**: Spoken language is detected (e.g., "Hindi").
3.  **Strategy Selection**:
    - **`best_of_all`**: Compare input against ALL enrolled buckets.
    - **`accent_matched`**: Priority check against the LID-detected bucket.
    - **`declared_language_fallback`**: Try buckets in priority order (Primary → Secondary → Optional).
4.  **Dual Scoring**: Compute similarity scores for ECAPA (and optionally X-Vector).
5.  **Decision**: Match is granted if score >= `SIMILARITY_THRESHOLD`.

### Key Design Principle: Raw Audio Processing
Unlike previous versions that used learned acoustic frontends (which collapsed speaker info), the current system passes **raw waveform** directly to ECAPA-TDNN. This preserves speaker-specific nuances and prevents false accepts between different speakers of the same language.

---

## 5. Model Summary

| Task | Model | Source |
|------|-------|--------|
| Speaker Identity | ECAPA-TDNN | `speechbrain/spkrec-ecapa-voxceleb` |
| Speaker Identity (Secondary) | X-Vector | `speechbrain/spkrec-xvect-voxceleb` |
| Language Identification | LID-ECAPA | `speechbrain/lang-id-voxlingua107-ecapa` |
| Speech Detection | Silero VAD | `snakers4/silero-vad` |

---

## 📊 Observability (Logs)
The system logs separate signals for transparency:
- `[LID]`: Detected language and confidence.
- `[ECAPA]`: Similarity score for a specific bucket and the final decision.

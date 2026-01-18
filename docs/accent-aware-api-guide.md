# Accent-Aware Multi-Lingual Voice Verification API Guide

## Overview

This document describes the **new accent-aware, multi-lingual voice verification system** that allows users to enroll in multiple languages (Indian multilingual use case) and verify their identity regardless of which enrolled language they speak during verification.

---

## What Changed?

### Before (Single-Language Flow)

```
User → Enroll (single language) → Verify (same language required)
```

- One embedding per user
- Verification only worked if user spoke in the same accent/language as during enrollment
- No language selection

### After (Multi-Language Flow)

```
User → Start Enrollment → Add Samples (per language) → Verify (any enrolled language)
        ↓                         ↓                            ↓
   Choose 2-3 languages    3 samples each language    Auto-matches best language
```

- Multiple embeddings per user (one per language)
- User explicitly declares primary + secondary languages
- Optional third language supported
- Verification tries multiple strategies to find a match
- Accent detection is advisory (never overrides user intent)

---

## New User Journey

### Enrollment Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                      ENROLLMENT JOURNEY                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. USER STARTS ENROLLMENT                                      │
│     ├── Selects primary language (e.g., Hindi - hi-IN)         │
│     ├── Selects secondary language (e.g., English - en-IN)     │
│     └── Optionally selects third language (e.g., Marathi)      │
│                                                                 │
│  2. USER PROVIDES SAMPLES FOR EACH LANGUAGE                     │
│     ├── Hindi: 3 voice samples speaking in Hindi               │
│     ├── English: 3 voice samples speaking in English           │
│     └── (Optional) Marathi: 3 voice samples                    │
│                                                                 │
│  3. SYSTEM CREATES SEPARATE VOICE PROFILES                      │
│     ├── Hindi voice profile (centroid)                         │
│     ├── English voice profile (centroid)                       │
│     └── (Optional) Marathi voice profile (centroid)            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Verification Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                     VERIFICATION JOURNEY                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. USER SUBMITS VOICE FOR VERIFICATION                         │
│     └── Can speak in ANY of their enrolled languages           │
│                                                                 │
│  2. SYSTEM APPLIES VERIFICATION STRATEGY                        │
│     ├── BEST OF ALL: Tries all languages, picks highest score  │
│     ├── ACCENT MATCHED: Uses detected language first           │
│     └── DECLARED FALLBACK: Tries primary → secondary → optional│
│                                                                 │
│  3. SYSTEM RETURNS RESULT                                       │
│     ├── Match/No Match decision                                │
│     ├── Which language matched                                 │
│     ├── Confidence scores                                      │
│     └── Detection info (advisory)                              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## New API Endpoints

### 1. Start Enrollment

**Purpose:** Initialize a multi-language enrollment session with language preferences.

```
POST /api/v1/voice/enroll/start
```

**Request:**
```json
{
    "user_id": "user_123",
    "primary_language": "hi-IN",
    "secondary_language": "en-IN",
    "optional_languages": ["mr-IN"]
}
```

**Response:**
```json
{
    "success": true,
    "user_id": "user_123",
    "message": "Enrollment session initialized",
    "primary_language": "hi-IN",
    "secondary_language": "en-IN",
    "optional_languages": ["mr-IN"],
    "samples_required_per_language": 3
}
```

**Notes:**
- Primary and secondary languages are mandatory
- Optional language is limited to 1
- Session must be started before adding samples

---

### 2. Add Enrollment Sample

**Purpose:** Submit a voice sample for a specific language during enrollment.

```
POST /api/v1/voice/enroll/sample
```

**Request:** (multipart/form-data)
| Field | Type | Description |
|-------|------|-------------|
| `user_id` | string | User identifier |
| `language` | string | BCP-47 language tag (e.g., hi-IN) |
| `audio` | file | Audio file (WAV, MP3, FLAC, OGG) |

**Response:**
```json
{
    "success": true,
    "user_id": "user_123",
    "message": "Sample 2 of 3 collected for hi-IN",
    "language": "hi-IN",
    "samples_collected": 2,
    "samples_required": 3,
    "language_complete": false,
    "detected_language": "unknown",
    "detection_confidence": 0.0
}
```

**Response when language is complete:**
```json
{
    "success": true,
    "user_id": "user_123",
    "message": "Enrollment complete for hi-IN. Voice profile created.",
    "language": "hi-IN",
    "samples_collected": 3,
    "samples_required": 3,
    "language_complete": true,
    "detected_language": "hi-IN",
    "detection_confidence": 0.85
}
```

**Notes:**
- Call once per sample (3 times per language)
- Language must be one of the declared languages from start enrollment
- System auto-finalizes when 3 samples are collected

---

### 3. Get Enrollment Status

**Purpose:** Check enrollment progress across all declared languages.

```
GET /api/v1/voice/enroll/status/{user_id}
```

**Response:**
```json
{
    "user_id": "user_123",
    "primary_language": "hi-IN",
    "secondary_language": "en-IN",
    "optional_languages": ["mr-IN"],
    "languages": [
        {
            "language_code": "hi-IN",
            "role": "primary",
            "samples_collected": 3,
            "samples_required": 3,
            "is_complete": true
        },
        {
            "language_code": "en-IN",
            "role": "secondary",
            "samples_collected": 1,
            "samples_required": 3,
            "is_complete": false
        },
        {
            "language_code": "mr-IN",
            "role": "optional",
            "samples_collected": 0,
            "samples_required": 3,
            "is_complete": false
        }
    ],
    "is_fully_enrolled": false,
    "can_finalize": false
}
```

**Response when fully enrolled:**
```json
{
    "user_id": "user_123",
    "primary_language": "hi-IN",
    "secondary_language": "en-IN",
    "optional_languages": [],
    "languages": [
        {
            "language_code": "hi-IN",
            "role": "primary",
            "samples_collected": 3,
            "samples_required": 3,
            "is_complete": true
        },
        {
            "language_code": "en-IN",
            "role": "secondary",
            "samples_collected": 3,
            "samples_required": 3,
            "is_complete": true
        }
    ],
    "is_fully_enrolled": true,
    "can_finalize": true
}
```

**Notes:**
- `is_fully_enrolled` is true when primary AND secondary are complete
- Optional languages don't block enrollment completion

---

### 4. Accent-Aware Verification

**Purpose:** Verify a user's voice with accent-aware strategies.

```
POST /api/v1/voice/accent/verify
```

**Request:** (multipart/form-data)
| Field | Type | Description |
|-------|------|-------------|
| `user_id` | string | User identifier |
| `audio` | file | Audio file for verification |
| `strategy` | string | `best_of_all` (default), `accent_matched`, or `declared_language_fallback` |

**Response (Match):**
```json
{
    "matched": true,
    "score": 0.87,
    "threshold": 0.80,
    "user_id": "user_123",
    "message": "Best-of-all language match",
    "strategy_used": "best_of_all",
    "detected_language": "hi-IN",
    "matched_language": "hi-IN",
    "confidence_level": "very_high",
    "primary_model": "ECAPA-TDNN",
    "primary_score": 0.89,
    "secondary_model": "X-Vector",
    "secondary_score": 0.82
}
```

**Response (No Match):**
```json
{
    "matched": false,
    "score": 0.45,
    "threshold": 0.80,
    "user_id": "user_123",
    "message": "Best-of-all language match",
    "strategy_used": "best_of_all",
    "detected_language": "unknown",
    "matched_language": null,
    "confidence_level": "low",
    "primary_model": "ECAPA-TDNN",
    "primary_score": 0.45,
    "secondary_model": "X-Vector",
    "secondary_score": 0.42
}
```

---

## Verification Strategies Explained

| Strategy | Behavior |
|----------|----------|
| **best_of_all** | Tries ALL enrolled languages, returns the highest matching score. Best for maximum flexibility. |
| **accent_matched** | First tries the detected language (if enrolled), then falls back to best_of_all. Fastest when accent detection is accurate. |
| **declared_language_fallback** | Tries languages in order: primary → secondary → optional. Respects user's declared preference. |

---

## Supported Languages

| Code | Language |
|------|----------|
| `en-IN` | English (India) |
| `hi-IN` | Hindi |
| `mr-IN` | Marathi |
| `ta-IN` | Tamil |
| `te-IN` | Telugu |
| `kn-IN` | Kannada |
| `ml-IN` | Malayalam |
| `gu-IN` | Gujarati |
| `pa-IN` | Punjabi |
| `bn-IN` | Bengali |

---

## Backward Compatibility

The old APIs still work but are now **deprecated**:

| Old Endpoint | Status | Behavior |
|--------------|--------|----------|
| `POST /api/v1/voice/enroll` | Deprecated | Routes through new flow if profile exists |
| `POST /api/v1/voice/verify` | Deprecated | Uses `best_of_all` strategy automatically |

**Recommendation:** Migrate to the new endpoints for multi-language support.

---

## Complete Example Flow

### Step 1: Start Enrollment
```bash
curl -X POST http://localhost:8000/api/v1/voice/enroll/start \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "ramesh_kumar",
    "primary_language": "hi-IN",
    "secondary_language": "en-IN"
  }'
```

### Step 2: Add Hindi Samples (3 times)
```bash
curl -X POST http://localhost:8000/api/v1/voice/enroll/sample \
  -F "user_id=ramesh_kumar" \
  -F "language=hi-IN" \
  -F "audio=@hindi_sample_1.wav"
```

### Step 3: Add English Samples (3 times)
```bash
curl -X POST http://localhost:8000/api/v1/voice/enroll/sample \
  -F "user_id=ramesh_kumar" \
  -F "language=en-IN" \
  -F "audio=@english_sample_1.wav"
```

### Step 4: Check Status
```bash
curl http://localhost:8000/api/v1/voice/enroll/status/ramesh_kumar
```

### Step 5: Verify (User can speak in Hindi OR English)
```bash
curl -X POST http://localhost:8000/api/v1/voice/accent/verify \
  -F "user_id=ramesh_kumar" \
  -F "strategy=best_of_all" \
  -F "audio=@verification_audio.wav"
```

---

## Key Principles

1. **User-declared language is authoritative** — The system never overrides the user's language choice
2. **Accent detection is advisory** — Used for routing and analytics, not decisions
3. **Language isolation** — Embeddings from different languages are never mixed
4. **Flexible verification** — Multiple strategies to maximize match success
5. **Backward compatible** — Old APIs continue to work for existing users

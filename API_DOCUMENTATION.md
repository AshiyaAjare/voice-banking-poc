# Voice-Print Backend API Documentation

Detailed documentation for the Voice Banking API, providing speaker enrollment and verification services with accent-aware multi-lingual support.

---

## Overview

The Voice-Print Backend provides a robust API for capturing, storing, and verifying speaker identities using voice embeddings. It supports both single-language and multi-lingual (accent-aware) flows, specifically tailored for Indian multilingual use cases.

- **Dual Model Comparison**: Utilizes both ECAPA-TDNN and X-Vector models for higher accuracy.
- **Accent-Aware**: Supports multiple language buckets (Hindi, English, Marathi, etc.) for a single user.
- **Backward Compatible**: Maintains legacy endpoints for existing integrations.

---

## Base URL

```
http://<host>:<port>
```
Default local development: `http://localhost:8000`

---

## Authentication

Current implementation uses open endpoints (CORS allowed for all origins). Ensure appropriate security layers (API keys, OAuth2) are added for production deployment.

---

## Core API Endpoints

### 1. Configuration

**GET `/api/v1/voice/config`**

Get enrollment and verification configuration for the frontend.

**Response:**
```json
{
  "min_enrollment_samples": 3,
  "max_enrollment_samples": 5,
  "enable_secondary_model": true,
  "primary_model_name": "ECAPA-TDNN",
  "secondary_model_name": "X-Vector",
  "similarity_threshold": 0.85
}
```

---

### 2. Multi-Language Enrollment (Recommended)

#### Start Enrollment Session
**POST `/api/v1/voice/enroll/start`**

Initializes a multi-language enrollment session.

**Request Body:**
| Field | Type | Description |
|-------|------|-------------|
| `user_id` | string | Unique identifier for the user |
| `primary_language` | string | BCP-47 language tag (e.g., `hi-IN`) |
| `secondary_language` | string | BCP-47 language tag (e.g., `en-IN`) |
| `optional_languages` | array[string] | Optional third language (max 1) |

**Example:**
```json
{
  "user_id": "user_123",
  "primary_language": "hi-IN",
  "secondary_language": "en-IN"
}
```

#### Add Enrollment Sample
**POST `/api/v1/voice/enroll/sample`**

Add a voice sample for a specific language bucket.

**Request (multipart/form-data):**
| Field | Type | Description |
|-------|------|-------------|
| `user_id` | string | User identifier |
| `language` | string | BCP-47 language tag |
| `audio` | file | Audio file (WAV, MP3, etc.) |

**Response:**
```json
{
  "success": true,
  "user_id": "user_123",
  "message": "Sample 1 of 3 collected for hi-IN",
  "language": "hi-IN",
  "samples_collected": 1,
  "samples_required": 3,
  "language_complete": false
}
```

#### Check Multi-Language Status
**GET `/api/v1/voice/enroll/status/{user_id}`**

Get enrollment progress across all declared languages.

---

### 3. Voice Verification

#### Accent-Aware Verification
**POST `/api/v1/voice/accent/verify`**

Verify a user's voice using accent-aware strategies.

**Request (multipart/form-data):**
| Field | Type | Description |
|-------|------|-------------|
| `user_id` | string | User identifier |
| `audio` | file | Audio file for verification |
| `strategy` | string | `best_of_all` \| `accent_matched` \| `declared_language_fallback` |

**Verification Strategies:**
- `best_of_all`: (Default) Tries all enrolled languages, picks best score.
- `accent_matched`: Uses detected language first, then falls back.
- `declared_language_fallback`: Tries primary → secondary → optional.

---

### 4. Enrollment Management

#### Check Status
**GET `/api/v1/voice/enrollment/{user_id}`**

Check if a user has an enrolled voice profile.

#### Delete Enrollment
**DELETE `/api/v1/voice/enrollment/{user_id}`**

Remove a user's voice enrollment from the system.

#### Cancel Pending Enrollment
**DELETE `/api/v1/voice/enrollment/{user_id}/cancel`**

Discard collected samples for an in-progress enrollment.

---

## Legacy Endpoints (Deprecated)

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/voice/enroll` | Single-sample/language enrollment |
| POST | `/api/v1/voice/verify` | Standard verification |

---

## Supported Languages (BCP-47)

- `en-IN`: English (India)
- `hi-IN`: Hindi
- `mr-IN`: Marathi
- `ta-IN`: Tamil
- `te-IN`: Telugu
- `kn-IN`: Kannada
- `ml-IN`: Malayalam
- `gu-IN`: Gujarati
- `pa-IN`: Punjabi
- `bn-IN`: Bengali

---

## Health Check

**GET `/health`**

Returns the operational status of the API.

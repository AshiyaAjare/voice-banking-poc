# 🎤 Voice Banking API

A voice biometric authentication system for banking applications using speaker recognition technology.

## Overview

Voice Banking API provides secure voice-based authentication by extracting and comparing speaker embeddings. It uses the **ECAPA-TDNN** model from SpeechBrain, trained on VoxCeleb, to create unique voiceprints for each user.

## Features

- **Multi-Sample Enrollment** - Robust voice profiles created from 5 audio samples
- **Centroid-Based Voice Print** - Multiple embeddings combined into a single robust speaker profile
- **Voice Verification** - Authenticate users by comparing their voice against enrolled profiles
- **Enrollment Management** - Check status, cancel pending, and delete voice enrollments
- **REST API** - Clean FastAPI-based endpoints with OpenAPI documentation

## Tech Stack

| Component | Technology |
|-----------|------------|
| Framework | FastAPI |
| ML Model | SpeechBrain ECAPA-TDNN |
| Audio Processing | torchaudio |
| Deep Learning | PyTorch |

## Installation

### Prerequisites

- Python 3.10+
- pip

### Setup

```bash
# Clone and navigate to project
cd voice-banking-draft

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

## Running the API

```bash
# Start the development server
uvicorn app.main:app --reload

# The API will be available at http://localhost:8000
```

### Interactive Documentation

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## API Endpoints

### Health Check
```http
GET /health
```
Returns API health status.

---

### Enroll Voice (Multi-Sample)
```http
POST /api/v1/voice/enroll
Content-Type: multipart/form-data

user_id: string (form field)
audio: file (audio file - WAV, MP3, FLAC, OGG)
```

Call this endpoint **5 times** with different audio samples to complete enrollment.

**Response (In Progress):**
```json
{
  "success": true,
  "user_id": "user123",
  "message": "Sample 3 of 5 collected successfully",
  "enrollment_complete": false,
  "samples_collected": 3,
  "samples_required": 5
}
```

**Response (Complete):**
```json
{
  "success": true,
  "user_id": "user123",
  "message": "Enrollment complete with 5 samples",
  "enrollment_complete": true,
  "samples_collected": 5,
  "samples_required": 5
}
```

---

### Verify Voice
```http
POST /api/v1/voice/verify
Content-Type: multipart/form-data

user_id: string (form field)
audio: file (audio file)
```

**Response:**
```json
{
  "matched": true,
  "score": 0.87,
  "threshold": 0.25,
  "user_id": "user123",
  "message": "Voice matched - authentication successful"
}
```

---

### Check Enrollment Status
```http
GET /api/v1/voice/enrollment/{user_id}
```

**Response:**
```json
{
  "enrolled": true,
  "user_id": "user123",
  "created_at": "2026-01-13T12:00:00",
  "enrollment_complete": true,
  "samples_collected": 5,
  "samples_required": 5
}
```

---

### Delete Enrollment
```http
DELETE /api/v1/voice/enrollment/{user_id}
```

**Response:**
```json
{
  "success": true,
  "user_id": "user123",
  "message": "Enrollment for user 'user123' deleted successfully"
}
```

---

### Cancel Pending Enrollment
```http
DELETE /api/v1/voice/enrollment/{user_id}/cancel
```

Cancel an in-progress enrollment and discard collected samples.

**Response:**
```json
{
  "success": true,
  "user_id": "user123",
  "message": "Pending enrollment cancelled, 3 samples discarded",
  "samples_discarded": 3
}
```

## Project Structure

```
voice-print-backend/
├── app/
│   ├── __init__.py
│   ├── main.py                # FastAPI application entry point
│   ├── config.py              # Application settings
│   ├── api/
│   │   ├── router.py          # API router aggregation
│   │   └── v1/
│   │       └── voice.py       # Voice authentication endpoints
│   ├── models/
│   │   └── schemas.py         # Pydantic request/response models
│   └── services/
│       ├── speaker_model.py   # SpeechBrain model wrapper
│       ├── voice_service.py   # Voice enrollment/verification logic
│       └── embedding_utils.py # Multi-sample centroid computation
├── storage/
│   ├── embeddings/            # Finalized user voice profiles
│   └── pending/               # In-progress enrollment samples
├── pretrained_models/         # Downloaded ML model files
└── requirements.txt
```

## Configuration

Configuration is managed via environment variables or a `.env` file:

| Variable | Default | Description |
|----------|---------|-------------|
| `APP_NAME` | Voice Banking API | Application name |
| `APP_VERSION` | 1.0.0 | API version |
| `DEBUG` | false | Enable debug mode |
| `SIMILARITY_THRESHOLD` | 0.25 | Voice match threshold (0-1) |
| `MIN_ENROLLMENT_SAMPLES` | 5 | Samples required to complete enrollment |
| `MAX_ENROLLMENT_SAMPLES` | 10 | Maximum samples allowed per enrollment |
| `ENROLLMENT_TIMEOUT_HOURS` | 24 | Auto-expire pending enrollments (hours) |

## How It Works

### Multi-Sample Enrollment

1. User provides 5 audio samples (one per API call)
2. Each sample is encoded by ECAPA-TDNN into a 192-dim embedding
3. Embeddings are stored in `storage/pending/{user_id}/`
4. When 5 samples are collected:
   - All embeddings are stacked: `[5, 192]`
   - **Centroid** (mean) is computed: `[1, 192]`
   - Result is L2 normalized for consistent cosine similarity
5. Final voice print is saved to `storage/embeddings/{user_id}.pt`

### Verification

1. User provides audio for authentication
2. Embedding is extracted from audio
3. Cosine similarity is computed against stored centroid
4. Match decision based on threshold (default: 0.25)

### Why Centroid?

- **Noise reduction**: Averaging across samples reduces the impact of any single noisy recording
- **Robustness**: The centroid better represents the speaker's "true" voice characteristics
- **Efficiency**: Simple mean + L2 normalization is computationally cheap

## License

MIT License

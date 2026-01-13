# 🎤 Voice Banking API

A voice biometric authentication system for banking applications using speaker recognition technology.

## Overview

Voice Banking API provides secure voice-based authentication by extracting and comparing speaker embeddings. It uses the **ECAPA-TDNN** model from SpeechBrain, trained on VoxCeleb, to create unique voiceprints for each user.

## Features

- **Voice Enrollment** - Register a user's voice profile from an audio sample
- **Voice Verification** - Authenticate users by comparing their voice against enrolled profiles
- **Enrollment Management** - Check status and delete voice enrollments
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

### Enroll Voice
```http
POST /api/v1/voice/enroll
Content-Type: multipart/form-data

user_id: string (form field)
audio: file (audio file - WAV, MP3, FLAC, OGG)
```

**Response:**
```json
{
  "success": true,
  "user_id": "user123",
  "message": "User 'user123' enrolled successfully"
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
  "created_at": "2026-01-13T12:00:00"
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

## Project Structure

```
voice-banking-draft/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI application entry point
│   ├── config.py            # Application settings
│   ├── api/
│   │   ├── router.py        # API router aggregation
│   │   └── v1/
│   │       └── voice.py     # Voice authentication endpoints
│   ├── models/
│   │   └── schemas.py       # Pydantic request/response models
│   └── services/
│       ├── speaker_model.py # SpeechBrain model wrapper
│       └── voice_service.py # Voice enrollment/verification logic
├── storage/
│   └── embeddings/          # Stored user voice embeddings
├── pretrained_models/       # Downloaded ML model files
├── audio/                   # Sample audio files for testing
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

## How It Works

1. **Enrollment**: User provides an audio sample → ECAPA-TDNN extracts a 192-dimensional embedding → Embedding is stored with user ID

2. **Verification**: User provides audio for authentication → Embedding is extracted → Cosine similarity is computed against stored embedding → Match decision based on threshold

## License

MIT License

# ScamShield AI - Honeypot API

Production-ready AI-powered scam detection honeypot for GUVI Hackathon.

## APIs

### 1. Voice Detection API
**Endpoint:** `POST /api/voice-detection`

Detects AI-generated vs human voice in audio files.

**Request:**
```json
{
  "language": "English",
  "audioFormat": "mp3",
  "audioBase64": "<base64-encoded-audio>"
}
```

**Response:**
```json
{
  "status": "success",
  "isHuman": false,
  "confidence": 0.85,
  "message": "Audio analysis complete",
  "timestamp": "2025-01-15T10:30:00Z"
}
```

### 2. Honeypot Scam Detection API
**Endpoint:** `POST /api/v1/scam-detection/`

Autonomous scam detection and engagement system.

**Request:**
```json
{
  "session_id": "unique-session-123",
  "message": "Hello, I'm calling from the bank...",
  "context": {
    "caller_id": "+1234567890",
    "call_timestamp": "2025-01-15T10:30:00Z"
  }
}
```

**Response:**
```json
{
  "status": "success",
  "reply": "Oh really? Which bank exactly?",
  "confidence_score": 0.75,
  "agent_handoff": true,
  "session_data": {
    "extracted_info": ["caller claims bank affiliation"],
    "risk_level": "high"
  }
}
```

## Authentication

All API calls require the `x-api-key` header:
```
x-api-key: sk_guvi_hackathon_2024
```

## Health Check

**Endpoint:** `GET /health`

Returns server status without authentication.

## Deployment

### Railway

1. Fork this repository
2. Connect to Railway
3. Set environment variables:
   - `API_KEYS=sk_guvi_hackathon_2024`
   - `PORT=8000`
4. Deploy

### Docker

```bash
docker build -t scamshield-ai .
docker run -p 8000:8000 -e API_KEYS=sk_guvi_hackathon_2024 scamshield-ai
```

## Rate Limiting

- 100 requests per 60 seconds per API key
- Returns 429 status when exceeded

## License

MIT License - GUVI Hackathon 2024

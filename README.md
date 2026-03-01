# Backend


FastAPI backend for the doppelganger interrogation game.

## What is implemented

- Game generation with Mistral: 1 real suspect + 2 impostors.
- Interrogation flow per suspect (5 base questions + 2 unlockable extras).
- Profanity filter (English/Spanish) for player questions.
- Sus-O-Scan:
  - Auto scan during interrogation.
  - On-demand scan endpoint.
- AI analysis endpoint and question suggestion endpoint.
- Final accusation endpoint.
- Text-to-speech with ElevenLabs.
- Game status and health endpoints.

## Stack

- Python + FastAPI
- Mistral API
- ElevenLabs API
- In-memory game state (no database yet)

## Setup

From `backend/`:

```bash
chmod +x setup.sh
./setup.sh
source venv/bin/activate
uvicorn main:app --reload --port 8000
```

Swagger: `http://localhost:8000/docs`

## Environment variables

Create `backend/.env`:

```env
MISTRAL_API_KEY=
ELEVENLABS_API_KEY=
MISTRAL_MODEL=mistral-large-latest
ENABLE_ANALYSIS=false
```

Notes:
- `MISTRAL_API_KEY` is required for core gameplay.
- `ELEVENLABS_API_KEY` is required only for `/api/game/narrate`.
- `ENABLE_ANALYSIS=true` adds inline analysis in interrogation responses.

## Main endpoints

- `POST /api/game/start`
- `POST /api/game/interrogate`
- `POST /api/game/unlock-extra`
- `POST /api/game/susoscan/scan`
- `POST /api/game/analyze`
- `POST /api/game/suggest`
- `POST /api/game/accuse`
- `POST /api/game/narrate`
- `GET /api/game/{game_id}/status`
- `GET /health`

## To do

- Add persistent storage (database).
- Add auth/rate limiting.
- Add automated tests.
- Add structured logging/metrics.
- Read model config from env consistently in all modules.
- Prepare Docker + production deployment.

# 🎬 VideoProcessApp — PhantomPilot

> **AI-powered video processing backend** built on the [OpenShorts](https://github.com/mutonby/openshorts) open-source stack.  
> Self-hosted · No watermarks · Deployed on GCP Cloud Run · Powered by Google Gemini.

---

## What It Does

VideoProcessApp is a **FastAPI backend** that turns long-form videos into viral short-form content. It exposes three AI pipelines via a simple REST API:

| Endpoint | What it does |
|---|---|
| `POST /api/clip-generator` | Extract the most viral clips from a long video (podcast, webinar, interview) |
| `POST /api/ai-shorts` | Generate a single vertical 9:16 short, reframed and captioned |
| `POST /api/youtube-studio` | Generate YouTube title, description, tags, and thumbnail |

All jobs are **async** — submit a job, get a `job_id`, poll for results.

---

## Tech Stack

```
┌─────────────────────────────────────────────────┐
│              FastAPI (Python 3.11)               │
├─────────────────────────────────────────────────┤
│  Transcription    faster-whisper (word-level)    │
│  Clip Selection   Google Gemini 2.0 Flash (LLM)  │
│  Scene Detection  PySceneDetect                  │
│  Face Tracking    MediaPipe BlazeFace            │
│  Person Fallback  YOLOv8n (Ultralytics)          │
│  9:16 Reframe     OpenCV + SmoothedCameraman     │
│  Video Rendering  FFmpeg                         │
├─────────────────────────────────────────────────┤
│  Storage          GCP Cloud Storage              │
│  Job Tracking     Supabase (PostgreSQL)          │
│  Auth             X-API-Key header               │
│  Cloud Host       GCP Cloud Run (Docker)         │
│  CI/CD            GCP Cloud Build                │
└─────────────────────────────────────────────────┘
```

---

## Project Structure

```
VideoProcessApp/
├── app/
│   ├── main.py                    # FastAPI app entry point + model preload
│   ├── auth.py                    # API Key header validation
│   ├── config.py                  # Pydantic settings (env vars)
│   ├── db/
│   │   └── supabase.py            # Job CRUD — create / get / list / update / cancel
│   ├── storage/
│   │   └── gcs.py                 # GCS download / upload / signed URLs
│   ├── models/
│   │   ├── requests.py            # Pydantic request models
│   │   └── responses.py           # Response models + JobStatus enum
│   ├── services/
│   │   ├── openshots_service.py   # Core engine: Whisper · Gemini · MediaPipe · FFmpeg
│   │   ├── clip_service.py        # Clip Generator pipeline
│   │   ├── shorts_service.py      # AI Shorts pipeline
│   │   └── studio_service.py      # YouTube Studio pipeline
│   └── routers/
│       ├── clip_generator.py      # POST /api/clip-generator
│       ├── ai_shorts.py           # POST /api/ai-shorts
│       ├── youtube_studio.py      # POST /api/youtube-studio
│       └── jobs.py                # GET + DELETE /api/jobs
├── Dockerfile                     # Multi-stage build (pre-bakes model weights)
├── docker-compose.yml             # Local development
├── cloudbuild.yaml                # GCP CI/CD pipeline
├── supabase_schema.sql            # Run once in Supabase SQL editor
├── requirements.txt
└── .env.example
```

---

## How Each Pipeline Works

### 🎯 Clip Generator
```
Video (GCS) → faster-whisper → Gemini LLM (viral moment selection)
           → FFmpeg cut × N clips → MediaPipe 9:16 reframe
           → GCS upload → Supabase job update
```

### ⚡ AI Shorts
```
Video (GCS) → faster-whisper → Gemini LLM (single best moment)
           → FFmpeg cut → MediaPipe 9:16 reframe
           → SRT captions → FFmpeg caption burn-in
           → GCS upload → Supabase job update
```

### 📺 YouTube Studio
```
Video (GCS) → faster-whisper → Gemini LLM (title + description + tags + thumbnail text)
           → FFmpeg thumbnail extract → GCS upload
           → Supabase job update
```

---

## Getting Started

### 1 — Prerequisites

- Python 3.11+
- Docker + Docker Compose
- A [GCP project](https://console.cloud.google.com) with Cloud Storage enabled
- A [Supabase](https://supabase.com) project
- A [Google AI Studio](https://aistudio.google.com) API key (Gemini)

### 2 — Clone & Configure

```bash
git clone https://github.com/your-org/VideoProcessApp
cd VideoProcessApp

# Create your .env from the example
cp .env.example .env
```

Open `.env` and fill in:

```env
API_KEY=your-secret-api-key

SUPABASE_URL=https://xxxx.supabase.co
SUPABASE_SERVICE_KEY=eyJ...

GCP_PROJECT_ID=your-project
GCP_BUCKET_NAME=your-bucket

GEMINI_API_KEY=AIza...
```

### 3 — Set Up Supabase

Run [`supabase_schema.sql`](./supabase_schema.sql) in your Supabase **SQL Editor**:

```sql
-- Creates the jobs table with status, result, and auto updated_at trigger
```

### 4 — GCP Service Account

```bash
gcloud iam service-accounts create video-process-sa
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:video-process-sa@$PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/storage.admin"
gcloud iam service-accounts keys create gcp-key.json \
  --iam-account=video-process-sa@$PROJECT_ID.iam.gserviceaccount.com
```

### 5 — Run Locally

```bash
docker-compose up --build
```

Server is live at **http://localhost:8080**  
Swagger docs at **http://localhost:8080/docs**

---

## API Reference

All endpoints require: `X-API-Key: <your-key>`

### Submit a Job

**Clip Generator:**
```bash
curl -X POST http://localhost:8080/api/clip-generator \
  -H "X-API-Key: your-key" \
  -H "Content-Type: application/json" \
  -d '{
    "video_url": "gs://your-bucket/input/podcast.mp4",
    "prompt": "Extract the most insightful and engaging moments",
    "options": { "max_clips": 5 }
  }'
```

**AI Shorts:**
```bash
curl -X POST http://localhost:8080/api/ai-shorts \
  -H "X-API-Key: your-key" \
  -H "Content-Type: application/json" \
  -d '{
    "video_url": "gs://your-bucket/input/interview.mp4",
    "prompt": "Create a 60-second viral short from the best moment",
    "options": { "target_duration": 60, "add_captions": true }
  }'
```

**YouTube Studio:**
```bash
curl -X POST http://localhost:8080/api/youtube-studio \
  -H "X-API-Key: your-key" \
  -H "Content-Type: application/json" \
  -d '{
    "video_url": "gs://your-bucket/input/tutorial.mp4",
    "prompt": "SEO-optimized metadata for a Python tutorial"
  }'
```

All three return immediately with:
```json
{ "job_id": "uuid", "status": "pending", "created_at": "..." }
```

### Poll for Results

```bash
curl http://localhost:8080/api/jobs/<job_id> \
  -H "X-API-Key: your-key"
```

```json
{
  "job_id": "...",
  "status": "completed",
  "result": {
    "clips": [
      {
        "clip_number": 1,
        "download_url": "https://...",
        "duration": 42.5,
        "viral_hook_text": "You won't believe this...",
        "tiktok_description": "...",
        "youtube_title": "..."
      }
    ]
  }
}
```

### Job Status Values

| Status | Meaning |
|---|---|
| `pending` | Job created, waiting to start |
| `processing` | Pipeline is running |
| `completed` | Done — result is available |
| `failed` | Error — check the `error` field |
| `cancelled` | Cancelled via DELETE |

### Other Endpoints

```
GET    /api/jobs?status=completed&type=clip_generator  → list jobs
DELETE /api/jobs/{job_id}                              → cancel job
GET    /health                                         → health check
GET    /docs                                           → Swagger UI
```

---

## Deploy to GCP Cloud Run

### One-time manual deploy

```bash
# Build and push
gcloud builds submit --tag gcr.io/$PROJECT_ID/video-process-app

# Deploy
gcloud run deploy video-process-app \
  --image gcr.io/$PROJECT_ID/video-process-app \
  --platform managed \
  --region us-central1 \
  --memory 4Gi \
  --cpu 2 \
  --timeout 3600 \
  --set-env-vars "GCP_PROJECT_ID=$PROJECT_ID,APP_ENV=production" \
  --set-secrets "API_KEY=video-process-api-key:latest,GEMINI_API_KEY=gemini-api-key:latest,SUPABASE_URL=supabase-url:latest,SUPABASE_SERVICE_KEY=supabase-key:latest"
```

### Auto-deploy via Cloud Build (CI/CD)

```bash
gcloud builds triggers create github \
  --repo-name=VideoProcessApp \
  --repo-owner=<your-org> \
  --branch-pattern=^main$ \
  --build-config=cloudbuild.yaml
```

Every push to `main` will automatically build, push, and deploy.

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `API_KEY` | ✅ | Secret key for `X-API-Key` header |
| `SUPABASE_URL` | ✅ | Supabase project URL |
| `SUPABASE_SERVICE_KEY` | ✅ | Supabase service-role key |
| `GCP_PROJECT_ID` | ✅ | GCP project ID |
| `GCP_BUCKET_NAME` | ✅ | GCS bucket for video I/O |
| `GEMINI_API_KEY` | ✅ | Google AI Studio API key |
| `GEMINI_MODEL` | — | Default: `gemini-2.0-flash` |
| `WHISPER_MODEL` | — | Default: `base` (`tiny`/`small`/`medium`/`large-v3`) |
| `CROP_MODE` | — | `TRACK` (face tracking) or `GENERAL` (blur background) |
| `MAX_CLIPS` | — | Max clips Gemini can select. Default: `10` |
| `GOOGLE_APPLICATION_CREDENTIALS` | — | Path to GCP service account JSON |
| `APP_ENV` | — | `development` or `production` |

---

## Credits

This backend is inspired by and built on the architecture of **[OpenShorts](https://github.com/mutonby/openshorts)** by [@mutonby](https://github.com/mutonby) — a free & open-source AI video platform.

Key components ported from OpenShorts:
- `SmoothedCameraman` — smooth 9:16 face-following crop
- `SpeakerTracker` — prevents rapid speaker switching during crop
- Gemini LLM prompt template for viral clip selection
- faster-whisper word-level transcript pipeline

---

## License

MIT — see [LICENSE](./LICENSE)


#Run it
# Deploy on Google Cloud Run


## Create artifact registry
```bash
gcloud artifacts repositories create video-process-app \
    --repository-format=docker \
    --location=europe-west1
```

## Buil image on cloud

```bash
gcloud builds submit \
    --tag europe-west1-docker.pkg.dev/${PROJECT_ID}/video-process-app/app:latest
```

## Deploy image on cloud run


```bash
gcloud run deploy video-process-app \
    --image europe-west1-docker.pkg.dev/phantompilot/video-process-app/app:latest \
    --region europe-west1 \
    --platform managed \
    --allow-unauthenticated \
    --memory 8Gi \
    --cpu 4 \
    --concurrency 1 \
    --max-instances 10 \
    --timeout 3600 \
    --service-account="service-storage@phantompilot.iam.gserviceaccount.com" \
    --set-env-vars "API_KEY=mytokentochangemen,SUPABASE_URL=https://iprbepulxpebwbquqxju.supabase.co,GCP_PROJECT_ID=phantompilot,GCP_BUCKET_NAME=phantompilot_videos,GEMINI_MODEL=gemini-2.0-flash,WHISPER_MODEL=base,CROP_MODE=TRACK,MAX_CLIPS=10,APP_ENV=production,LOG_LEVEL=info,REFRAME_QUALITY=LOW,WHISPER_BEAM_SIZE=1" \
    --set-secrets "SUPABASE_SERVICE_KEY=SUPABASE_SERVICE_KEY:latest,GEMINI_API_KEY=GEMINI_API_KEY:latest" 
```


# Add service account rght to secret management

```bash
gcloud iam service-accounts add-iam-policy-binding service-storage@phantompilot.iam.gserviceaccount.com --role roles/secretmanager.secretAccessor --member "serviceAccount:service-storage@phantompilot.iam.gserviceaccount.com" --project phantompilot
```
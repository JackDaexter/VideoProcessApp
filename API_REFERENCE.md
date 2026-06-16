# VideoProcessApp — API Reference

**Base URL:** `https://<cloud-run-url>`  
**Auth:** All requests require header `X-API-Key: <key>`  
**Pattern:** POST → get `job_id` → poll `GET /api/jobs/{job_id}` until `status: completed`

---

## POST /api/clip-generator
Cuts the most viral moments from a long video into 9:16 clips using Gemini + face tracking.

**Body:**
```json
{
  "video_url": "gs://bucket/video.mp4",       // required — GCS URI
  "prompt": "Extract the funniest moments",    // required — guides Gemini
  "options": {
    "max_clips": 5,                            // default 5, max 20
    "max_duration_per_clip": 60,               // seconds, default 60
    "min_duration_per_clip": 15               // seconds, default 15
  }
}
```

**Result (`result` field when completed):**
```json
{
  "total_clips": 4,
  "transcript": "...",
  "clips": [{
    "clip_number": 1,
    "download_url": "https://signed-url...",   // valid 12h
    "gcs_uri": "gs://bucket/output/.../clip_01.mp4",
    "start": 42.3,                             // in source video (seconds)
    "end": 84.7,
    "duration": 42.4,
    "viral_hook_text": "You won't believe this",
    "tiktok_description": "...",
    "instagram_description": "...",
    "youtube_title": "..."
  }]
}
```

---

## POST /api/ai-shorts
Picks the single best moment, cuts and reframes to 9:16, optionally burns captions.

**Body:**
```json
{
  "video_url": "gs://bucket/video.mp4",        // required
  "prompt": "Make a 60s viral short",          // required
  "options": {
    "target_duration": 60,                     // seconds, default 60
    "add_captions": true                       // burn subtitles, default true
  }
}
```

**Result:**
```json
{
  "download_url": "https://signed-url...",     // valid 12h
  "gcs_uri": "gs://bucket/output/.../short.mp4",
  "duration": 58.3,
  "captions_burned": true,
  "viral_hook_text": "...",
  "tiktok_description": "...",
  "instagram_description": "...",
  "youtube_title": "...",
  "transcript": "...",
  "segment": { "start": 120.5, "end": 178.8 } // position in source
}
```

---

## POST /api/youtube-studio
Transcribes a video and uses Gemini to generate full YouTube metadata + thumbnail.

**Body:**
```json
{
  "video_url": "gs://bucket/video.mp4",        // required
  "prompt": "SEO metadata for a Python tutorial", // required
  "channel_context": "We teach AI to devs"     // optional
}
```

**Result:**
```json
{
  "title": "...",
  "description": "...with chapter timestamps...",
  "tags": ["python", "tutorial", "..."],
  "thumbnail_text": "Learn Python TODAY",
  "thumbnail_download_url": "https://signed-url...", // JPEG, valid 12h
  "thumbnail_url": "gs://bucket/output/.../thumbnail.jpg",
  "transcript": "..."
}
```

---

## GET /api/jobs/{job_id}
Returns current job state.

```json
{
  "job_id": "uuid",
  "type": "clip_generator | ai_shorts | youtube_studio",
  "status": "pending | processing | completed | failed | cancelled",
  "result": { ... },   // present when completed
  "error": "...",      // present when failed
  "created_at": "...",
  "updated_at": "..."
}
```

## GET /api/jobs
List jobs. Query params: `status`, `type`, `limit` (default 20), `offset`.

## DELETE /api/jobs/{job_id}
Cancels a job (sets status to `cancelled`).

## GET /health *(no auth)*
```json
{ "status": "ok", "version": "2.0.0", "environment": "production" }
```

---

## Errors
| Code | Meaning |
|---|---|
| `403` | Bad or missing `X-API-Key` |
| `404` | Job not found |
| `422` | Invalid request body |
| `500` | Server error |

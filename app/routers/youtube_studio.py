"""
app/routers/youtube_studio.py — POST /api/youtube-studio endpoint.
"""

from datetime import datetime, timezone

from fastapi import APIRouter, BackgroundTasks, Depends

from app.auth import require_api_key
from app.db.supabase import create_job
from app.models.requests import YouTubeStudioRequest
from app.models.responses import JobCreatedResponse, JobStatus, JobType
from app.services.studio_service import run_youtube_studio

router = APIRouter(prefix="/api", tags=["YouTube Studio"])


@router.post(
    "/youtube-studio",
    response_model=JobCreatedResponse,
    status_code=202,
    summary="Generate YouTube metadata for a video",
    description=(
        "Submit a video to automatically generate a YouTube-ready title, description, "
        "tags, and thumbnail. Powered by Whisper transcription + NLP extraction. "
        "Poll GET /api/jobs/{job_id} for status and results."
    ),
)
async def youtube_studio(
    request: YouTubeStudioRequest,
    background_tasks: BackgroundTasks,
    _api_key: str = Depends(require_api_key),
) -> JobCreatedResponse:
    """
    Create a YouTube Studio metadata generation job.

    - **video_url**: GCS URI of the source video (gs://bucket/path)
    - **prompt**: Style/focus guidance for the metadata generation
    - **channel_context**: Optional branding/channel info to include
    """
    job = await create_job(
        job_type=JobType.YOUTUBE_STUDIO,
        input_url=request.video_url,
        prompt=request.prompt,
        options={"channel_context": request.channel_context} if request.channel_context else {},
    )

    background_tasks.add_task(run_youtube_studio, job["id"], request)

    return JobCreatedResponse(
        job_id=job["id"],
        status=JobStatus.PENDING,
        created_at=datetime.now(timezone.utc),
    )

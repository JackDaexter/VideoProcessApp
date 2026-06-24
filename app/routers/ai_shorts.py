"""
app/routers/ai_shorts.py — POST /api/ai-shorts endpoint.
"""

from datetime import datetime, timezone

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status

from app.auth import get_current_user
from app.db.supabase import create_job
from app.models.requests import AIShortsRequest
from app.models.responses import JobCreatedResponse, JobStatus, JobType
from app.services.shorts_service import run_ai_shorts
from app.storage.gcs import gcs_uri_belongs_to_user_upload

router = APIRouter(prefix="/api", tags=["AI Shorts"])


@router.post(
    "/ai-shorts",
    response_model=JobCreatedResponse,
    status_code=202,
    summary="Generate a vertical AI short from a video",
    description=(
        "Submit a video to generate a vertical 9:16 short (Reels/TikTok/YouTube Shorts style). "
        "Includes scene detection, caption burning, and automatic reformatting. "
        "Poll GET /api/jobs/{job_id} for status and download URL."
    ),
)
async def ai_shorts(
    request: AIShortsRequest,
    background_tasks: BackgroundTasks,
    user_id: str = Depends(get_current_user),
) -> JobCreatedResponse:
    """
    Create an AI Shorts generation job.

    - **video_url**: GCS URI of the source video (gs://bucket/path)
    - **prompt**: Guidance for which content to include in the short
    - **options**: target_duration, add_captions, aspect_ratio
    """
    if not gcs_uri_belongs_to_user_upload(request.video_url, user_id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="video_url must point to the current user's uploaded folder.",
        )

    job = await create_job(
        user_id=user_id,
        job_type=JobType.AI_SHORTS,
        input_url=request.video_url,
        prompt=request.prompt,
        options=request.options.model_dump(),
    )

    background_tasks.add_task(run_ai_shorts, job["id"], user_id, request)

    return JobCreatedResponse(
        job_id=job["id"],
        status=JobStatus.PENDING,
        created_at=datetime.now(timezone.utc),
    )

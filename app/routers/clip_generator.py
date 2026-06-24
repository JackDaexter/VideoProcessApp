"""
app/routers/clip_generator.py — POST /api/clip-generator endpoint.
"""

from datetime import datetime, timezone

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status

from app.auth import get_current_user
from app.db.supabase import create_job
from app.models.requests import ClipGeneratorRequest
from app.models.responses import JobCreatedResponse, JobStatus, JobType
from app.services.clip_service import run_clip_generator
from app.storage.gcs import gcs_uri_belongs_to_user_upload

router = APIRouter(prefix="/api", tags=["Clip Generator"])


@router.post(
    "/clip-generator",
    response_model=JobCreatedResponse,
    status_code=202,
    summary="Generate highlight clips from a video",
    description=(
        "Submit a video for clip generation. The job runs asynchronously. "
        "Poll GET /api/jobs/{job_id} for status and results."
    ),
)
async def clip_generator(
    request: ClipGeneratorRequest,
    background_tasks: BackgroundTasks,
    user_id: str = Depends(get_current_user),
) -> JobCreatedResponse:
    """
    Create a clip generator job.

    - **video_url**: GCS URI of the source video (gs://bucket/path)
    - **prompt**: Natural language description of clips to extract
    - **options**: Optional configuration (max_clips, duration bounds)
    """
    if not gcs_uri_belongs_to_user_upload(request.video_url, user_id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="video_url must point to the current user's uploaded folder.",
        )

    job = await create_job(
        user_id=user_id,
        job_type=JobType.CLIP_GENERATOR,
        input_url=request.video_url,
        prompt=request.prompt,
        options=request.options.model_dump(),
    )

    background_tasks.add_task(run_clip_generator, job["id"], user_id, request)

    return JobCreatedResponse(
        job_id=job["id"],
        status=JobStatus.PENDING,
        created_at=datetime.now(timezone.utc),
    )

"""
app/routers/jobs.py — Job management endpoints.

Endpoints:
  GET  /api/jobs             — List all jobs (with optional filters)
  GET  /api/jobs/{job_id}    — Get a single job status + result
  DELETE /api/jobs/{job_id}  — Cancel a job (soft delete → status=cancelled)
"""

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status

from app.auth import get_current_user
from app.db.supabase import delete_job, get_job, list_jobs
from app.models.responses import JobListResponse, JobResponse, JobStatus, JobType

router = APIRouter(prefix="/api/jobs", tags=["Job Management"])


def _parse_job_row(row: dict) -> JobResponse:
    """Convert a raw Supabase row dict to a JobResponse model."""
    return JobResponse(
        job_id=row["id"],
        user_id=row["user_id"],
        type=JobType(row["type"]),
        status=JobStatus(row["status"]),
        current_step=row.get("current_step"),
        input_url=row["input_url"],
        prompt=row["prompt"],
        options=row.get("options"),
        result=row.get("result"),
        error=row.get("error"),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


@router.get(
    "/{job_id}",
    response_model=JobResponse,
    summary="Get job status and result",
    description="Retrieve the current status and (when completed) the result of a processing job.",
)
async def get_job_status(
    job_id: str,
    user_id: str = Depends(get_current_user),
) -> JobResponse:
    """Poll this endpoint after submitting a job to check progress."""
    row = await get_job(job_id)
    if not row or row.get("user_id") != user_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job '{job_id}' not found.",
        )
    return _parse_job_row(row)


@router.get(
    "",
    response_model=JobListResponse,
    summary="List all jobs",
    description="Return a paginated list of jobs with optional status and type filters.",
)
async def list_all_jobs(
    job_status: Optional[str] = Query(
        default=None,
        alias="status",
        description="Filter by status: pending | processing | completed | failed | cancelled",
    ),
    job_type: Optional[str] = Query(
        default=None,
        alias="type",
        description="Filter by type: clip_generator | ai_shorts | youtube_studio",
    ),
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    user_id: str = Depends(get_current_user),
) -> JobListResponse:
    """List jobs with optional filtering and pagination."""
    rows = await list_jobs(
        user_id=user_id,
        status=job_status,
        job_type=job_type,
        limit=limit,
        offset=offset,
    )
    jobs = [_parse_job_row(r) for r in rows]
    return JobListResponse(jobs=jobs, total=len(jobs))


@router.delete(
    "/{job_id}",
    status_code=status.HTTP_200_OK,
    summary="Cancel a job",
    description="Cancel a job. Running jobs cannot be interrupted mid-process but will be marked as cancelled.",
)
async def cancel_job(
    job_id: str,
    user_id: str = Depends(get_current_user),
) -> dict:
    """Cancel / soft-delete a job by setting its status to 'cancelled'."""
    row = await get_job(job_id)
    if not row or row.get("user_id") != user_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job '{job_id}' not found.",
        )
    
    await delete_job(job_id)
    return {"message": f"Job '{job_id}' has been cancelled.", "job_id": job_id}

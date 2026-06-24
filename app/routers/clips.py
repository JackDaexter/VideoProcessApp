"""
app/routers/clips.py — Per-clip management endpoints for Clip Generator jobs.

Allows the frontend Shorts section to:
  - List all clip jobs (with their individual clips) for the current user
  - List clips for a specific job
  - Delete a single clip (GCS file removed, DB result array updated)
  - Delete an entire clip job and all its GCS files
"""

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query, status

from app.auth import get_current_user
from app.db.supabase import delete_job, get_job, list_user_clip_jobs, update_job_result
from app.models.responses import ClipItemResponse, ClipJobListResponse, ClipJobResponse, JobType
from app.storage.gcs import delete_gcs_prefix, delete_gcs_uri

router = APIRouter(tags=["Clips"])
log = structlog.get_logger(__name__)


def _parse_clip_job(row: dict) -> ClipJobResponse:
    result = row.get("result") or {}
    clips_raw = result.get("clips") or []
    clips = [ClipItemResponse(**c) for c in clips_raw]
    return ClipJobResponse(
        job_id=row["id"],
        user_id=row["user_id"],
        status=row["status"],
        input_url=row["input_url"],
        prompt=row["prompt"],
        total_clips=result.get("total_clips", len(clips)),
        transcript=result.get("transcript"),
        clips=clips,
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


# ── List all clip jobs ────────────────────────────────────────────────────────

@router.get(
    "/api/clips",
    response_model=ClipJobListResponse,
    summary="List the current user's clip generator jobs with their clips",
)
@router.get("/clips", response_model=ClipJobListResponse, include_in_schema=False)
async def list_clip_jobs(
    limit: int = Query(default=50, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    user_id: str = Depends(get_current_user),
) -> ClipJobListResponse:
    rows = await list_user_clip_jobs(user_id=user_id, limit=limit, offset=offset)
    jobs = [_parse_clip_job(row) for row in rows]
    return ClipJobListResponse(jobs=jobs, total=len(jobs))


# ── Get clips for a specific job ──────────────────────────────────────────────

@router.get(
    "/api/clips/{job_id}",
    response_model=ClipJobResponse,
    summary="Get all clips for a specific clip generator job",
)
@router.get("/clips/{job_id}", response_model=ClipJobResponse, include_in_schema=False)
async def get_clip_job(
    job_id: str,
    user_id: str = Depends(get_current_user),
) -> ClipJobResponse:
    row = await get_job(job_id)
    if not row or row.get("user_id") != user_id or row.get("type") != JobType.CLIP_GENERATOR.value:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Clip job '{job_id}' not found.",
        )
    return _parse_clip_job(row)


# ── Delete a single clip ───────────────────────────────────────────────────────

@router.delete(
    "/api/clips/{job_id}/clip/{clip_number}",
    status_code=status.HTTP_200_OK,
    summary="Delete a single clip from a job (removes GCS file and updates job result)",
)
@router.delete(
    "/clips/{job_id}/clip/{clip_number}",
    status_code=status.HTTP_200_OK,
    include_in_schema=False,
)
async def remove_clip(
    job_id: str,
    clip_number: int,
    user_id: str = Depends(get_current_user),
) -> dict:
    row = await get_job(job_id)
    if not row or row.get("user_id") != user_id or row.get("type") != JobType.CLIP_GENERATOR.value:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Clip job '{job_id}' not found.",
        )

    result = row.get("result") or {}
    clips: list = result.get("clips") or []

    # Find the target clip
    target = next((c for c in clips if c.get("clip_number") == clip_number), None)
    if not target:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Clip #{clip_number} not found in job '{job_id}'.",
        )

    # Delete the GCS file
    gcs_uri = target.get("gcs_uri")
    if gcs_uri:
        try:
            await delete_gcs_uri(gcs_uri)
        except Exception as exc:
            log.warning("clip_gcs_delete_failed", job_id=job_id, clip=clip_number, uri=gcs_uri, error=str(exc))

    # Remove from clips array and patch the DB result
    updated_clips = [c for c in clips if c.get("clip_number") != clip_number]
    updated_result = {**result, "clips": updated_clips, "total_clips": len(updated_clips)}
    await update_job_result(job_id, updated_result)

    log.info("clip_removed", job_id=job_id, clip_number=clip_number)
    return {
        "message": f"Clip #{clip_number} removed from job '{job_id}'.",
        "job_id": job_id,
        "clip_number": clip_number,
        "remaining_clips": len(updated_clips),
    }


# ── Delete an entire clip job ─────────────────────────────────────────────────

@router.delete(
    "/api/clips/{job_id}",
    status_code=status.HTTP_200_OK,
    summary="Delete an entire clip job and all its GCS files",
)
@router.delete("/clips/{job_id}", status_code=status.HTTP_200_OK, include_in_schema=False)
async def remove_clip_job(
    job_id: str,
    user_id: str = Depends(get_current_user),
) -> dict:
    row = await get_job(job_id)
    if not row or row.get("user_id") != user_id or row.get("type") != JobType.CLIP_GENERATOR.value:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Clip job '{job_id}' not found.",
        )

    # Delete all GCS files under output/{user_id}/{job_id}/clips/
    result = row.get("result") or {}
    clips: list = result.get("clips") or []

    deleted_files = 0
    for clip in clips:
        gcs_uri = clip.get("gcs_uri")
        if gcs_uri:
            try:
                await delete_gcs_uri(gcs_uri)
                deleted_files += 1
            except Exception as exc:
                log.warning(
                    "clip_job_gcs_delete_failed",
                    job_id=job_id,
                    clip=clip.get("clip_number"),
                    uri=gcs_uri,
                    error=str(exc),
                )

    # Also sweep any orphaned files in the GCS prefix (in case some weren't in result)
    try:
        await delete_gcs_prefix(user_id=user_id, job_id=job_id, sub_path="clips")
    except Exception as exc:
        log.warning("clip_prefix_sweep_failed", job_id=job_id, error=str(exc))

    await delete_job(job_id)

    log.info("clip_job_removed", job_id=job_id, gcs_files_deleted=deleted_files)
    return {
        "message": f"Clip job '{job_id}' and its files have been removed.",
        "job_id": job_id,
        "gcs_files_deleted": deleted_files,
    }

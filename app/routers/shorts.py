"""
app/routers/shorts.py — User-scoped shorts listing endpoints.
"""

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query, status

from app.auth import get_current_user
from app.db.supabase import delete_job, get_job, list_user_shorts
from app.models.responses import JobStatus, JobType, ShortListResponse, ShortResponse
from app.storage.gcs import delete_gcs_uri

router = APIRouter(tags=["Shorts"])
log = structlog.get_logger(__name__)


def _parse_short_row(row: dict) -> ShortResponse:
    return ShortResponse(
        job_id=row["id"],
        user_id=row["user_id"],
        status=row["status"],
        input_url=row["input_url"],
        prompt=row["prompt"],
        result=row.get("result") or {},
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


@router.get(
    "/api/shorts",
    response_model=ShortListResponse,
    summary="List the current user's completed shorts",
)
@router.get(
    "/shorts",
    response_model=ShortListResponse,
    include_in_schema=False,
)
async def list_shorts(
    limit: int = Query(default=50, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    user_id: str = Depends(get_current_user),
) -> ShortListResponse:
    rows = await list_user_shorts(user_id=user_id, limit=limit, offset=offset)
    shorts = [_parse_short_row(row) for row in rows]
    return ShortListResponse(shorts=shorts, total=len(shorts))

# Look like remove all shoort per jobs
@router.delete(
    "/api/shorts/{job_id}",
    status_code=status.HTTP_200_OK,
    summary="Remove one of the current user's shorts",
)
@router.delete(
    "/shorts/{job_id}",
    status_code=status.HTTP_200_OK,
    include_in_schema=False,
)
async def remove_short(
    job_id: str,
    user_id: str = Depends(get_current_user),
) -> dict:
    row = await get_job(job_id)
    if (
        not row
        or row.get("user_id") != user_id
        or row.get("type") != JobType.AI_SHORTS.value
    ):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Short '{job_id}' not found.",
        )

    result = row.get("result") or {}
    gcs_uri = result.get("short_url") or result.get("gcs_uri")
    if row.get("status") == JobStatus.COMPLETED.value and gcs_uri:
        try:
            await delete_gcs_uri(gcs_uri)
        except Exception as exc:
            log.warning("short_gcs_delete_failed", job_id=job_id, uri=gcs_uri, error=str(exc))

    await delete_job(job_id)
    return {"message": f"Short '{job_id}' has been removed.", "job_id": job_id}

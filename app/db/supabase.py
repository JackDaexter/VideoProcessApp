"""
app/db/supabase.py — Supabase client and job CRUD operations.

All database interaction for the jobs table goes through this module.
Uses the Supabase Python SDK with the service-role key (bypasses RLS).
"""

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from supabase import AsyncClient, acreate_client
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import get_settings
from app.models.responses import JobStatus, JobType

import structlog

log = structlog.get_logger(__name__)


# ── Client Singleton ──────────────────────────────────────────────────────────

_client: Optional[AsyncClient] = None


async def get_client() -> AsyncClient:
    """Return (or create) the async Supabase client singleton."""
    global _client
    if _client is None:
        settings = get_settings()
        _client = await acreate_client(
            settings.supabase_url,
            settings.supabase_service_key,
        )
    return _client


# ── Job CRUD ──────────────────────────────────────────────────────────────────

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=4))
async def create_job(
    job_type: JobType,
    input_url: str,
    prompt: str,
    options: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Insert a new job row with status=pending and return the full row.
    """
    client = await get_client()
    job_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()

    payload = {
        "id": job_id,
        "type": job_type.value,
        "status": JobStatus.PENDING.value,
        "input_url": input_url,
        "prompt": prompt,
        "options": options or {},
        "created_at": now,
        "updated_at": now,
    }

    response = await client.table("jobs").insert(payload).execute()
    row = response.data[0]
    log.info("job_created", job_id=job_id, type=job_type.value)
    return row


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=4))
async def get_job(job_id: str) -> Optional[Dict[str, Any]]:
    """Fetch a single job by ID. Returns None if not found."""
    client = await get_client()
    response = (
        await client.table("jobs").select("*").eq("id", job_id).execute()
    )
    if not response.data:
        return None
    return response.data[0]


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=4))
async def list_jobs(
    status: Optional[str] = None,
    job_type: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
) -> List[Dict[str, Any]]:
    """List jobs with optional filters."""
    client = await get_client()
    query = client.table("jobs").select("*").order("created_at", desc=True)

    if status:
        query = query.eq("status", status)
    if job_type:
        query = query.eq("type", job_type)

    query = query.range(offset, offset + limit - 1)
    response = await query.execute()
    return response.data


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=4))
async def update_job_status(
    job_id: str,
    status: JobStatus,
    result: Optional[Dict[str, Any]] = None,
    error: Optional[str] = None,
) -> Dict[str, Any]:
    """Update job status (and optionally result/error)."""
    client = await get_client()
    now = datetime.now(timezone.utc).isoformat()

    payload: Dict[str, Any] = {
        "status": status.value,
        "updated_at": now,
    }
    if result is not None:
        payload["result"] = result
    if error is not None:
        payload["error"] = error

    response = (
        await client.table("jobs").update(payload).eq("id", job_id).execute()
    )
    log.info("job_updated", job_id=job_id, status=status.value)
    return response.data[0]


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=4))
async def update_job_step(job_id: str, step: str) -> None:
    """Update current_step to reflect which pipeline stage is active."""
    client = await get_client()
    now = datetime.now(timezone.utc).isoformat()
    await client.table("jobs").update(
        {"current_step": step, "updated_at": now}
    ).eq("id", job_id).execute()
    log.info("job_step", job_id=job_id, step=step)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=4))
async def delete_job(job_id: str) -> bool:
    """
    Mark a job as cancelled (soft delete). Returns True if job existed.
    """
    client = await get_client()
    existing = await get_job(job_id)
    if not existing:
        return False

    now = datetime.now(timezone.utc).isoformat()
    await client.table("jobs").update(
        {"status": JobStatus.CANCELLED.value, "updated_at": now}
    ).eq("id", job_id).execute()

    log.info("job_cancelled", job_id=job_id)
    return True

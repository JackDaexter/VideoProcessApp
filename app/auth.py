"""
app/auth.py — API Key authentication dependency for FastAPI.

Every protected endpoint calls `Depends(require_api_key)`.
The client must pass the key in the `X-API-Key` HTTP header.
"""

from fastapi import Depends, HTTPException, Security, status
from fastapi.security import APIKeyHeader

from app.config import get_settings

_API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=True)


async def require_api_key(
    api_key: str = Security(_API_KEY_HEADER),
) -> str:
    """
    FastAPI dependency — validates the X-API-Key header.

    Raises:
        HTTPException 403: if the key is missing or incorrect.
    """
    settings = get_settings()
    if api_key != settings.api_key:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid or missing API key.",
        )
    return api_key

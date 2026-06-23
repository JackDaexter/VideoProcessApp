"""
app/auth.py — Firebase Authentication dependency for FastAPI.

Every protected endpoint calls `Depends(get_current_user)`.
The client must pass the Firebase JWT in the `Authorization: Bearer <token>` HTTP header.
"""

import firebase_admin
import firebase_admin.auth
from fastapi import Depends, HTTPException, Security, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

security = HTTPBearer(auto_error=True)


def ensure_firebase_app() -> bool:
    """Initialize Firebase Admin once before auth APIs are used."""
    try:
        firebase_admin.get_app()
        return False
    except ValueError:
        firebase_admin.initialize_app()
        return True


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Security(security),
) -> str:
    """
    FastAPI dependency — validates the Firebase JWT token.

    Returns:
        str: The Firebase user_id (uid).
        
    Raises:
        HTTPException 401: if the token is missing, invalid, or expired.
    """
    token = credentials.credentials
    try:
        ensure_firebase_app()
        decoded_token = firebase_admin.auth.verify_id_token(token)
        return decoded_token.get("uid")
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid authentication credentials: {str(e)}",
            headers={"WWW-Authenticate": "Bearer"},
        )

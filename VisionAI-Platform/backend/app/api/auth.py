"""VisionAI - Auth API Router"""

from datetime import timedelta
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

from app.core.security import authenticate_user, create_access_token
from app.core.config import settings

router = APIRouter()


class LoginRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    role: str


@router.post("/auth/login", response_model=TokenResponse)
async def login(req: LoginRequest):
    user = authenticate_user(req.username, req.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
        )
    token = create_access_token(
        {"sub": user["username"], "role": user["role"]},
        timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES),
    )
    return TokenResponse(access_token=token, role=user["role"])


@router.get("/auth/me")
async def me(current_user=None):
    from fastapi import Depends
    from app.core.security import get_current_user
    return {"username": current_user["username"], "role": current_user["role"]}

"""
JWT Authentication for Sentinel API.

Provides token-based authentication using python-jose.
Tokens are validated on each request to protected endpoints.
"""

import os
from datetime import UTC, datetime, timedelta
from typing import Annotated

import structlog
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from pydantic import BaseModel, Field

logger = structlog.get_logger()

# ============================================================================
# Configuration
# ============================================================================

# Secret key for JWT signing - MUST be set in production
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "CHANGE_ME_IN_PRODUCTION_USE_STRONG_KEY")
ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("JWT_EXPIRE_MINUTES", "30"))

# Aliases for backward compatibility with tests
JWT_SECRET_KEY = SECRET_KEY
JWT_ALGORITHM = ALGORITHM

# Warn if using default secret key
if SECRET_KEY == "CHANGE_ME_IN_PRODUCTION_USE_STRONG_KEY":
    logger.warning(
        "jwt_default_secret",
        message="Using default JWT secret key. Set JWT_SECRET_KEY in production!",
    )

# Security scheme for FastAPI docs
security_scheme = HTTPBearer(
    scheme_name="JWT",
    description="JWT Bearer token authentication",
    auto_error=False,  # Handle missing credentials manually to return 401
)


# ============================================================================
# Models
# ============================================================================


class TokenData(BaseModel):
    """Data extracted from a validated JWT token."""

    sub: str = Field(..., description="Subject (user identifier)")
    exp: datetime = Field(..., description="Expiration time")
    iat: datetime = Field(..., description="Issued at time")
    scopes: list[str] = Field(default_factory=list, description="Permission scopes")
    tenant_id: str | None = Field(None, description="Tenant identifier for multi-tenancy")

    @property
    def user_id(self) -> str:
        """Alias for sub (subject) field."""
        return self.sub

    @property
    def is_expired(self) -> bool:
        """Check if token has expired."""
        return datetime.now(UTC) > self.exp.replace(tzinfo=UTC)


class TokenResponse(BaseModel):
    """Response model for token generation."""

    access_token: str = Field(..., description="JWT access token")
    token_type: str = Field(default="bearer", description="Token type")
    expires_in: int = Field(..., description="Token lifetime in seconds")


# ============================================================================
# Token Creation
# ============================================================================


def create_access_token(
    subject: str,
    scopes: list[str] | None = None,
    tenant_id: str | None = None,
    expires_delta: timedelta | None = None,
) -> str:
    """
    Create a JWT access token.

    Args:
        subject: User identifier (typically user ID or email).
        scopes: Permission scopes for the token.
        tenant_id: Tenant identifier for multi-tenant deployments.
        expires_delta: Custom token lifetime. Defaults to ACCESS_TOKEN_EXPIRE_MINUTES.

    Returns:
        Encoded JWT token string.

    Example:
        >>> token = create_access_token(
        ...     subject="user@example.com",
        ...     scopes=["predict:read", "predict:write"],
        ... )
    """
    if expires_delta is None:
        expires_delta = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)

    now = datetime.now(UTC)
    expire = now + expires_delta

    payload = {
        "sub": subject,
        "exp": expire,
        "iat": now,
        "scopes": scopes or [],
    }

    if tenant_id:
        payload["tenant_id"] = tenant_id

    encoded_jwt: str = jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

    logger.info(
        "token_created",
        subject=subject,
        scopes=scopes,
        expires_at=expire.isoformat(),
    )

    return encoded_jwt


def decode_token(token: str) -> TokenData:
    """
    Decode and validate a JWT token.

    Args:
        token: The JWT token string.

    Returns:
        TokenData with decoded claims.

    Raises:
        HTTPException: If token is invalid or expired.
    """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])

        # Extract claims
        subject: str | None = payload.get("sub")
        if subject is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token missing subject claim",
                headers={"WWW-Authenticate": "Bearer"},
            )

        # Parse timestamps
        exp_timestamp = payload.get("exp")
        iat_timestamp = payload.get("iat")

        if exp_timestamp is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token missing expiration claim",
                headers={"WWW-Authenticate": "Bearer"},
            )

        exp = datetime.fromtimestamp(exp_timestamp, tz=UTC)
        iat = datetime.fromtimestamp(iat_timestamp, tz=UTC) if iat_timestamp else datetime.now(UTC)

        return TokenData(
            sub=subject,
            exp=exp,
            iat=iat,
            scopes=payload.get("scopes", []),
            tenant_id=payload.get("tenant_id"),
        )

    except JWTError as e:
        logger.warning("token_decode_failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )


# ============================================================================
# FastAPI Dependencies
# ============================================================================


async def verify_token(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(security_scheme)],
) -> TokenData:
    """
    FastAPI dependency to verify JWT token from Authorization header.

    Usage:
        @app.get("/protected")
        async def protected_route(token_data: TokenData = Depends(verify_token)):
            return {"user": token_data.user_id}

    Args:
        credentials: HTTP Bearer credentials extracted by FastAPI.

    Returns:
        TokenData with decoded and validated claims.

    Raises:
        HTTPException: 401 if token is missing, invalid, or expired.
    """
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token = credentials.credentials
    token_data = decode_token(token)

    # Check expiration
    if token_data.is_expired:
        logger.warning("token_expired", subject=token_data.sub)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
            headers={"WWW-Authenticate": "Bearer"},
        )

    logger.debug(
        "token_verified",
        subject=token_data.sub,
        scopes=token_data.scopes,
    )

    return token_data


def require_scopes(*required_scopes: str):
    """
    FastAPI dependency factory to require specific scopes.

    Usage:
        @app.post("/admin/action")
        async def admin_action(
            token_data: TokenData = Depends(require_scopes("admin:write"))
        ):
            ...

    Args:
        *required_scopes: Scopes that must be present in the token.

    Returns:
        FastAPI dependency function.
    """

    async def scope_checker(
        token_data: Annotated[TokenData, Depends(verify_token)],
    ) -> TokenData:
        """Check that token has required scopes."""
        token_scopes = set(token_data.scopes)
        required = set(required_scopes)

        if not required.issubset(token_scopes):
            missing = required - token_scopes
            logger.warning(
                "insufficient_scopes",
                subject=token_data.sub,
                required=list(required),
                missing=list(missing),
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Missing required scopes: {', '.join(missing)}",
            )

        return token_data

    return scope_checker


# ============================================================================
# Optional: API Key Authentication (for service-to-service)
# ============================================================================


class APIKeyAuth:
    """
    API Key authentication for service-to-service communication.

    Usage:
        api_key_auth = APIKeyAuth()

        @app.get("/internal/metrics")
        async def internal_metrics(
            validated: bool = Depends(api_key_auth.verify)
        ):
            ...
    """

    def __init__(self) -> None:
        """Initialize with API key from environment."""
        self.api_key = os.getenv("SENTINEL_API_KEY")
        if not self.api_key:
            logger.warning(
                "api_key_not_configured",
                message="SENTINEL_API_KEY not set. API key auth will reject all requests.",
            )

    async def verify(
        self,
        credentials: Annotated[HTTPAuthorizationCredentials, Depends(security_scheme)],
    ) -> bool:
        """
        Verify API key from Authorization header.

        Returns:
            True if API key is valid.

        Raises:
            HTTPException: 401 if API key is invalid.
        """
        if not self.api_key:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="API key authentication not configured",
            )

        if credentials.credentials != self.api_key:
            logger.warning("invalid_api_key")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API key",
            )

        return True

import os
import time
import jwt
from fastapi import HTTPException, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

# Secret key for JWT signing (use an environment variable in production)
JWT_SECRET = os.getenv("JWT_SECRET")
ALGORITHM = "HS256"

# HTTPBearer instance to parse the "Authorization: Bearer <token>" header
http_bearer = HTTPBearer()

def signJWT(user_id: str) -> dict:
    """
    Generates a JWT token with user_id and an expiration of 1 hour.
    """
    payload = {
        "user_id": user_id,
        "expires": time.time() + 3600  # Token valid for 1 hour
    }
    token = jwt.encode(payload, JWT_SECRET, algorithm=ALGORITHM)
    return {"access_token": token}

def decodeJWT(token: str) -> dict:
    """
    Decodes the JWT token. Raises 401 HTTPException if token is invalid or expired.
    """
    try:
        decoded_token = jwt.decode(token, JWT_SECRET, algorithms=[ALGORITHM])
        if decoded_token["expires"] < time.time():
            raise HTTPException(status_code=401, detail="Token expired")
        return decoded_token
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid token")

def get_current_user(credentials: HTTPAuthorizationCredentials = Security(http_bearer)) -> dict:
    """
    Dependency function to retrieve the current user by decoding the Bearer token.
    """
    return decodeJWT(credentials.credentials) 
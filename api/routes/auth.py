from fastapi import APIRouter

router = APIRouter()

# Simple auth endpoint — accepts any credentials and returns a mock token
# This lets the Chrome extension's login form work without a full auth system
@router.post("/api/auth/login")
def login(payload: dict):
    email = payload.get("email", "")
    password = payload.get("password", "")

    # For now, accept any non-empty email/password
    if email and password:
        return {
            "success": True,
            "token": "footballiq-static-token",
            "user": {
                "email": email,
                "role": "admin"
            }
        }

    return {"success": False, "error": "Email and password are required"}

"""
routes/auth.py — Real JWT authentication for PlusOne.

Endpoints:
  POST /api/auth/login                 — email + password → JWT token
  POST /api/auth/create-admin          — admin creates a new admin/user account
  GET  /api/auth/me                    — returns current user info (requires token)
  GET  /api/auth/users                 — list all users (admin only)
  DELETE /api/auth/users/{user_id}     — delete a user (admin only)
  PUT  /api/auth/users/{user_id}/role  — change a user's role (admin only)

Registration is admin-only: only an existing admin can create new accounts.
"""

from fastapi import APIRouter, HTTPException, Depends, status
from pydantic import BaseModel, EmailStr
from typing import Optional

from auth_utils import hash_password, verify_password, create_access_token
from routes.deps import get_current_user, require_admin
from database import get_connection

router = APIRouter()


# ── Request / Response models ─────────────────────────────────────────────────

class LoginRequest(BaseModel):
    email: str
    password: str

class CreateUserRequest(BaseModel):
    email: str
    password: str
    role: Optional[str] = "user"   # 'user' | 'admin'

class ChangeRoleRequest(BaseModel):
    role: str


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/api/auth/login")
def login(payload: LoginRequest):
    """
    Authenticate with email + password.
    Returns a signed JWT token on success.
    """
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT id, email, password_hash, role FROM users WHERE email = %s LIMIT 1",
            (payload.email.lower().strip(),),
        )
        user = cur.fetchone()
    finally:
        conn.close()

    # Generic error to avoid leaking whether the email exists
    if user is None or not verify_password(payload.password, user["password_hash"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password.",
        )

    token = create_access_token({"sub": user["email"], "role": user["role"]})
    return {
        "success": True,
        "token": token,
        "token_type": "bearer",
        "user": {
            "id":    user["id"],
            "email": user["email"],
            "role":  user["role"],
        },
    }


@router.post("/api/auth/create-user", status_code=201)
def create_user(
    payload: CreateUserRequest,
    _admin: dict = Depends(require_admin),
):
    """
    Admin-only: create a new user or admin account.
    Only existing admins can call this endpoint.
    """
    if payload.role not in ("user", "admin"):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="role must be 'user' or 'admin'.",
        )

    email = payload.email.lower().strip()
    if not email:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Email is required.",
        )
    if len(payload.password) < 8:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Password must be at least 8 characters.",
        )

    hashed = hash_password(payload.password)
    conn   = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO users (email, password_hash, role)
            VALUES (%s, %s, %s)
            RETURNING id, email, role, created_at
            """,
            (email, hashed, payload.role),
        )
        new_user = cur.fetchone()
        conn.commit()
    except Exception as exc:
        conn.rollback()
        if "unique" in str(exc).lower():
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"A user with email '{email}' already exists.",
            )
        raise HTTPException(status_code=500, detail="Failed to create user.")
    finally:
        conn.close()

    return {
        "success": True,
        "user": {
            "id":         new_user["id"],
            "email":      new_user["email"],
            "role":       new_user["role"],
            "created_at": new_user["created_at"].isoformat() if new_user["created_at"] else None,
        },
    }


@router.get("/api/auth/me")
def me(user: dict = Depends(get_current_user)):
    """Return the currently authenticated user's profile."""
    return {
        "id":    user["id"],
        "email": user["email"],
        "role":  user["role"],
    }


@router.get("/api/auth/users")
def list_users(_admin: dict = Depends(require_admin)):
    """Admin only — list all registered users."""
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT id, email, role, created_at FROM users ORDER BY id"
        )
        rows = cur.fetchall()
    finally:
        conn.close()

    return {
        "users": [
            {
                "id":         r["id"],
                "email":      r["email"],
                "role":       r["role"],
                "created_at": r["created_at"].isoformat() if r["created_at"] else None,
            }
            for r in rows
        ]
    }


@router.put("/api/auth/users/{user_id}/role")
def change_role(
    user_id: int,
    payload: ChangeRoleRequest,
    _admin: dict = Depends(require_admin),
):
    """Admin only — change a user's role between 'user' and 'admin'."""
    if payload.role not in ("user", "admin"):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="role must be 'user' or 'admin'.",
        )
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            "UPDATE users SET role = %s WHERE id = %s RETURNING id, email, role",
            (payload.role, user_id),
        )
        updated = cur.fetchone()
        conn.commit()
    except Exception:
        conn.rollback()
        raise HTTPException(status_code=500, detail="Failed to update role.")
    finally:
        conn.close()

    if updated is None:
        raise HTTPException(status_code=404, detail=f"User {user_id} not found.")

    return {"success": True, "user": dict(updated)}


@router.delete("/api/auth/users/{user_id}", status_code=200)
def delete_user(
    user_id: int,
    admin: dict = Depends(require_admin),
):
    """Admin only — delete a user account. Cannot delete yourself."""
    if admin["id"] == user_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="You cannot delete your own account.",
        )
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute("DELETE FROM users WHERE id = %s RETURNING id", (user_id,))
        deleted = cur.fetchone()
        conn.commit()
    except Exception:
        conn.rollback()
        raise HTTPException(status_code=500, detail="Failed to delete user.")
    finally:
        conn.close()

    if deleted is None:
        raise HTTPException(status_code=404, detail=f"User {user_id} not found.")

    return {"success": True, "deleted_id": user_id}

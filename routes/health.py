from fastapi import APIRouter
from database import get_connection

router = APIRouter()

@router.get("/health")
def health_check():
    try:
        conn = get_connection()
        conn.close()
        
        import sys
        sys_info = {
            "executable": sys.executable,
            "path": sys.path
        }
        
        httpx_info = "not imported"
        try:
            import httpx
            httpx_info = f"installed (version {getattr(httpx, '__version__', 'unknown')}) at {httpx.__file__}"
        except ImportError as e:
            httpx_info = f"ImportError: {str(e)}"
        except Exception as e:
            httpx_info = f"Crash: {str(e)}"
            
        return {
            "status": "healthy", 
            "version": "1.0.0", 
            "database": "connected",
            "sys_info": sys_info,
            "httpx": httpx_info
        }
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}

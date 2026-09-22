"""
FastAPI Server Launch Script for PAM CDS API.
Executes: uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
"""

import subprocess
import sys

if __name__ == "__main__":
    print("Launching PAM Early Prediction API Server...")
    print("Documentation available at: http://localhost:8000/docs")
    print("ReDoc available at:         http://localhost:8000/redoc")
    try:
        import uvicorn
        uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
    except KeyboardInterrupt:
        print("\nServer shutdown gracefully.")

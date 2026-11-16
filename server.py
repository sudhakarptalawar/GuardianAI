"""
GuardianAI — Server Entry Point
Runs the dashboard backend on http://localhost:8000
"""

import uvicorn
from api.main import app

if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("   🛡️ GuardianAI Console — Local Safety System")
    print("   Web UI: http://localhost:8000")
    print("   API Docs: http://localhost:8000/docs")
    print("=" * 50 + "\n")
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=False, log_level="info")

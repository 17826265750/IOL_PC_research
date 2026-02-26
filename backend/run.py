"""
Entry point for the CIPS 2008 Lifetime Prediction Backend.
"""
import uvicorn
from app.main import app


def main():
    """Run the FastAPI application."""
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )


if __name__ == "__main__":
    main()

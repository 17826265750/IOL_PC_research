"""
Entry point for the Power Module Lifetime Analysis Backend.

功率模块寿命分析软件 - 后端启动入口
Author: GSH
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

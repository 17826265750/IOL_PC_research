"""
FastAPI application entry point for CIPS 2008 Lifetime Prediction.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config import get_settings
from app.db.database import init_db
from app.core.models.model_factory import ModelFactory
from contextlib import asynccontextmanager
import logging

logger = logging.getLogger(__name__)

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup
    logger.info("Starting CIPS 2008 Lifetime Prediction API")

    # Initialize database
    try:
        init_db()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")

    # Register all lifetime models
    try:
        ModelFactory.register_all()
        logger.info(f"Registered {len(ModelFactory.list_models())} lifetime models")
    except Exception as e:
        logger.error(f"Failed to register models: {e}")

    yield

    # Shutdown
    logger.info("Shutting down CIPS 2008 Lifetime Prediction API")


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Backend API for IOL PC board lifetime prediction using CIPS 2008 standard",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Import and include routers
from app.api import prediction, rainflow, damage, analysis, experiments, export

# Include all API routers with /api prefix
app.include_router(
    prediction.router,
    prefix="/api",
    tags=["prediction"]
)
app.include_router(
    rainflow.router,
    prefix="/api",
    tags=["rainflow"]
)
app.include_router(
    damage.router,
    prefix="/api",
    tags=["damage"]
)
app.include_router(
    analysis.router,
    prefix="/api",
    tags=["analysis"]
)
app.include_router(
    experiments.router,
    prefix="/api",
    tags=["experiments"]
)
app.include_router(
    export.router,
    prefix="/api",
    tags=["export"]
)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        # Check database connection
        from app.db.database import SessionLocal
        db = SessionLocal()
        db.execute("SELECT 1")
        db.close()

        return {
            "status": "healthy",
            "app": settings.app_name,
            "version": settings.app_version,
            "database": "connected",
            "models_registered": len(ModelFactory.list_models())
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "app": settings.app_name,
            "version": settings.app_version,
            "error": str(e)
        }


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "CIPS 2008 IOL Lifetime Prediction API",
        "version": settings.app_version,
        "docs": "/api/docs",
        "endpoints": {
            "prediction": "/api/prediction",
            "rainflow": "/api/rainflow",
            "damage": "/api/damage",
            "analysis": "/api/analysis",
            "experiments": "/api/experiments",
            "export": "/api/export"
        }
    }

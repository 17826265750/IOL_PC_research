"""
SQLite database setup with SQLAlchemy.
"""
from sqlalchemy import create_engine, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from app.config import get_settings
import logging

logger = logging.getLogger(__name__)

settings = get_settings()

# Create SQLite engine
engine = create_engine(
    settings.database_url,
    connect_args={"check_same_thread": False}  # Needed for SQLite
)

# Create SessionLocal class
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create Base class for models
Base = declarative_base()


def get_db() -> Session:
    """
    Dependency function to get database session.

    Yields:
        Session: SQLAlchemy database session
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Initialize database tables."""
    try:
        from app.models import prediction, experiment, parameters, mission_profile

        # Create all tables
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created successfully")

        # Verify connection
        with engine.connect() as conn:
            result = conn.execute(text("SELECT name FROM sqlite_master WHERE type='table'"))
            tables = [row[0] for row in result]
            logger.info(f"Database tables: {tables}")

    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise


def reset_db():
    """Drop and recreate all tables (use with caution!)."""
    try:
        from app.models import prediction, experiment, parameters, mission_profile

        # Drop all tables
        Base.metadata.drop_all(bind=engine)
        logger.info("Database tables dropped")

        # Recreate tables
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables recreated")

    except Exception as e:
        logger.error(f"Failed to reset database: {e}")
        raise

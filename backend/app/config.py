"""
Configuration settings using Pydantic BaseSettings.

功率模块寿命分析软件 - 配置模块
Author: GSH
"""
from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache


class Settings(BaseSettings):
    """Application configuration settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )

    # Application
    app_name: str = "功率模块寿命分析软件"
    app_version: str = "1.0.0"
    debug: bool = True
    
    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    
    # Database
    database_url: str = "sqlite:///./cips_prediction.db"
    
    # CORS
    cors_origins: list[str] = [
        "http://localhost:5173",
        "http://localhost:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:3000"
    ]
    
    # File Upload
    max_upload_size: int = 10 * 1024 * 1024  # 10MB
    upload_dir: str = "./uploads"
    
    # Model Parameters
    default_damage_model: str = "miner"
    default_rainflow_bin_count: int = 64
    
    # Export
    export_dir: str = "./exports"


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()

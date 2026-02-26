"""
API routes package.

Exports all API routers for easy inclusion in the main application.
"""
from app.api import prediction, rainflow, damage, analysis, experiments, export

__all__ = [
    "prediction",
    "rainflow",
    "damage",
    "analysis",
    "experiments",
    "export"
]

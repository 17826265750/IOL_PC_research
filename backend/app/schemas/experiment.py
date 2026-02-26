"""
Pydantic schemas for Experiment model.
"""
from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional


class ExperimentBase(BaseModel):
    """Base experiment schema."""
    name: str = Field(..., description="Experiment name")
    description: Optional[str] = Field(None, description="Experiment description")
    temperature: float = Field(..., description="Test temperature in Celsius", gt=-273.15)
    humidity: Optional[float] = Field(None, description="Relative humidity %", ge=0, le=100)
    temperature_cycles: Optional[int] = Field(None, description="Number of temp cycles", ge=0)


class ExperimentCreate(ExperimentBase):
    """Schema for creating an experiment."""
    cycles: Optional[int] = Field(None, ge=0)
    failures: int = Field(default=0, ge=0)
    test_duration_hours: Optional[float] = Field(None, ge=0)
    data_file_path: Optional[str] = None
    data_format: Optional[str] = None


class ExperimentUpdate(BaseModel):
    """Schema for updating an experiment."""
    name: Optional[str] = None
    description: Optional[str] = None
    temperature: Optional[float] = None
    humidity: Optional[float] = None
    temperature_cycles: Optional[int] = None
    cycles: Optional[int] = None
    failures: Optional[int] = None
    test_duration_hours: Optional[float] = None
    mean_time_to_failure: Optional[float] = None
    data_file_path: Optional[str] = None


class ExperimentResponse(ExperimentBase):
    """Schema for experiment response."""
    id: int
    cycles: Optional[int] = None
    failures: int
    test_duration_hours: Optional[float] = None
    mean_time_to_failure: Optional[float] = None
    data_file_path: Optional[str] = None
    data_format: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True

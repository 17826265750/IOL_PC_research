"""
Pydantic schemas for Prediction model.
"""
from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional


class PredictionBase(BaseModel):
    """Base prediction schema."""
    name: str = Field(..., description="Prediction name/identifier")
    model_type: str = Field(default="CIPS_ARRHENIUS", description="Type of lifetime model")
    notes: Optional[str] = Field(None, description="Additional notes")


class PredictionCreate(PredictionBase):
    """Schema for creating a prediction."""
    parameters_id: Optional[int] = Field(None, description="Model parameters ID")
    mission_profile_id: Optional[int] = Field(None, description="Mission profile ID")


class PredictionUpdate(BaseModel):
    """Schema for updating a prediction."""
    name: Optional[str] = None
    model_type: Optional[str] = None
    predicted_lifetime_years: Optional[float] = None
    predicted_lifetime_cycles: Optional[int] = None
    total_damage: Optional[float] = None
    confidence_level: Optional[float] = None
    notes: Optional[str] = None
    parameters_id: Optional[int] = None
    mission_profile_id: Optional[int] = None


class PredictionResponse(PredictionBase):
    """Schema for prediction response."""
    id: int
    predicted_lifetime_years: Optional[float] = None
    predicted_lifetime_cycles: Optional[int] = None
    total_damage: Optional[float] = None
    confidence_level: Optional[float] = None
    parameters_id: Optional[int] = None
    mission_profile_id: Optional[int] = None
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True

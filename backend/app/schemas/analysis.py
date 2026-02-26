"""
Pydantic schemas for advanced analysis modules.
"""
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime


class SensitivityAnalysisRequest(BaseModel):
    """Request for sensitivity analysis."""
    base_parameters: Dict[str, Any] = Field(..., description="Base model parameters")
    parameter_ranges: Dict[str, tuple[float, float]] = Field(..., description="Parameter ranges to analyze")
    samples_per_parameter: int = Field(default=10, ge=2, le=100, description="Samples per parameter")


class SensitivityResult(BaseModel):
    """Result for a single parameter sensitivity."""
    parameter_name: str
    sensitivity_coefficient: float
    min_lifetime: float
    max_lifetime: float
    percent_change: float


class SensitivityAnalysisResponse(BaseModel):
    """Response for sensitivity analysis."""
    results: List[SensitivityResult]
    base_lifetime: float
    most_sensitive_parameter: str


class AccelerationFactorRequest(BaseModel):
    """Request for acceleration factor calculation."""
    test_temperature: float = Field(..., description="Test temperature in Celsius")
    use_temperature: float = Field(..., description="Use temperature in Celsius")
    activation_energy: float = Field(default=0.7, description="Activation energy in eV")


class AccelerationFactorResponse(BaseModel):
    """Response for acceleration factor calculation."""
    acceleration_factor: float
    test_time_hours: Optional[float] = Field(None, description="Test duration in hours")
    equivalent_use_hours: Optional[float] = Field(None, description="Equivalent use condition hours")


class WeibullAnalysisRequest(BaseModel):
    """Request for Weibull reliability analysis."""
    failure_times: List[float] = Field(..., description="List of failure times in hours")
    suspended_count: int = Field(default=0, ge=0, description="Number of suspended units")


class WeibullAnalysisResponse(BaseModel):
    """Response for Weibull analysis."""
    shape_parameter: float = Field(..., description="Weibull shape (beta)")
    scale_parameter: float = Field(..., description="Weibull scale (eta) in hours")
    characteristic_life: float = Field(..., description="Characteristic life (63.2% failure)")
    mtbf: Optional[float] = Field(None, description="Mean time between failures")


class AnalysisRequest(BaseModel):
    """Generic analysis request."""
    analysis_type: str = Field(..., description="Type of analysis to perform")
    input_data: Dict[str, Any] = Field(..., description="Analysis input data")


class AnalysisResponse(BaseModel):
    """Generic analysis response."""
    analysis_type: str
    results: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    status: str = Field(default="completed")
    errors: Optional[List[str]] = None

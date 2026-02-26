"""
Pydantic schemas for Rainflow cycle counting.
"""
from pydantic import BaseModel, Field
from typing import List, Optional


class DataPoint(BaseModel):
    """Single data point for time-series analysis."""
    time: float = Field(..., description="Time in seconds")
    value: float = Field(..., description="Temperature or stress value")


class CycleCount(BaseModel):
    """Individual cycle count from rainflow analysis."""
    stress_range: float = Field(..., description="Stress/temperature range")
    mean_value: float = Field(..., description="Mean stress/temperature")
    cycles: float = Field(..., description="Number of cycles (can be fractional)")


class RainflowRequest(BaseModel):
    """Request schema for rainflow cycle counting."""
    data_points: List[DataPoint] = Field(..., description="Time-series data points")
    bin_count: int = Field(default=64, ge=8, le=256, description="Number of bins for histogram")
    method: str = Field(default="ASTM", description="Rainflow algorithm variant")


class RainflowResponse(BaseModel):
    """Response schema for rainflow analysis."""
    cycles: List[CycleCount] = Field(..., description="Cycle count results")
    total_cycles: float = Field(..., description="Total number of cycles")
    max_range: float = Field(..., description="Maximum stress/temperature range")
    summary: dict = Field(default_factory=dict, description="Summary statistics")


class HistogramBin(BaseModel):
    """Histogram bin for cycle distribution."""
    range_min: float
    range_max: float
    cycle_count: float
    damage_contribution: Optional[float] = None


class RainflowHistogramRequest(BaseModel):
    """Request for rainflow histogram analysis."""
    cycles: List[CycleCount]
    bin_count: int = Field(default=64, ge=8, le=256)


class RainflowHistogramResponse(BaseModel):
    """Response for rainflow histogram."""
    bins: List[HistogramBin]
    total_cycles: float
    cumulative_damage: Optional[float] = None

"""
Pydantic schemas for damage accumulation analysis.
"""
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from app.schemas.rainflow import CycleCount


class DamageModel(BaseModel):
    """Base damage model parameters."""
    model_type: str = Field(..., description="Model type (MINER, GOODMAN, etc.)")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Model-specific parameters")


class SNCurve(BaseModel):
    """S-N curve parameters for fatigue life calculation."""
    intercept: float = Field(..., description="S-N curve intercept (stress cycles)")
    slope: float = Field(..., description="S-N curve slope (Basquin exponent)")


# ============================================================================
# Safety Margin Schemas
# ============================================================================

class SafetyMarginRequest(BaseModel):
    """Request schema for safety margin calculation."""
    design_life_cycles: float = Field(..., gt=0, description="Required design life in cycles")
    predicted_life_cycles: float = Field(..., gt=0, description="Predicted life in cycles")
    safety_factor: float = Field(default=1.0, gt=0, le=10.0, description="Safety factor to apply")
    minimum_acceptable_margin: float = Field(default=0.0, description="Minimum acceptable margin percentage")


class SafetyMarginResponse(BaseModel):
    """Response schema for safety margin calculation."""
    safety_factor: float = Field(..., description="Applied safety factor")
    design_life_cycles: float = Field(..., description="Required design life in cycles")
    predicted_life_cycles: float = Field(..., description="Predicted life in cycles")
    margin_percentage: float = Field(..., description="Safety margin as percentage")
    margin_value: float = Field(..., description="Absolute margin (predicted - design)")
    utilization: float = Field(..., description="Ratio of used life to available life")
    is_acceptable: bool = Field(..., description="Whether the margin meets requirements")
    adequacy_level: str = Field(..., description="Design adequacy category")
    recommendation: str = Field(..., description="Design recommendation")
    inspection_recommendation: Optional[str] = Field(None, description="Inspection frequency recommendation")


class StatisticalSafetyMarginRequest(BaseModel):
    """Request schema for statistical safety margin calculation."""
    design_life_cycles: float = Field(..., gt=0, description="Required design life in cycles")
    predicted_life_mean: float = Field(..., gt=0, description="Mean of predicted life distribution")
    predicted_life_std: float = Field(..., ge=0, description="Standard deviation of predicted life")
    safety_factor: float = Field(default=1.0, gt=0, le=10.0, description="Safety factor to apply")
    minimum_acceptable_margin: float = Field(default=0.0, description="Minimum acceptable margin percentage")


class StatisticalSafetyMarginResponse(BaseModel):
    """Response schema for statistical safety margin calculation."""
    mean_margin: float = Field(..., description="Mean safety margin percentage")
    std_margin: float = Field(..., description="Standard deviation of margin")
    percentile_5: float = Field(..., description="5th percentile (conservative)")
    percentile_95: float = Field(..., description="95th percentile (optimistic)")
    probability_acceptable: float = Field(..., description="Probability that margin is acceptable")


class RequiredSafetyFactorRequest(BaseModel):
    """Request schema for calculating required safety factor."""
    design_life_cycles: float = Field(..., gt=0, description="Required design life in cycles")
    predicted_life_cycles: float = Field(..., gt=0, description="Predicted life in cycles")
    target_margin: float = Field(default=0.0, ge=-99, le=1000, description="Target margin percentage")


class RequiredSafetyFactorResponse(BaseModel):
    """Response schema for required safety factor calculation."""
    required_safety_factor: float = Field(..., description="Required safety factor to achieve target margin")


# ============================================================================
# Lifetime Curve Schemas
# ============================================================================

class CurvePoint(BaseModel):
    """A single point on a lifetime curve."""
    x_value: float = Field(..., description="X-axis value (e.g., temperature, cycles)")
    y_value: float = Field(..., description="Y-axis value (e.g., cycles to failure)")
    log_x: Optional[float] = Field(None, description="Log-transformed x value")
    log_y: Optional[float] = Field(None, description="Log-transformed y value")


class ModelCurve(BaseModel):
    """Curve data for a single model."""
    model_type: str = Field(..., description="Model type identifier")
    model_name: str = Field(..., description="Human-readable model name")
    points: List[CurvePoint] = Field(..., description="Curve data points")
    equation: str = Field(..., description="Model equation")


class LifetimeCurveRequest(BaseModel):
    """Request schema for lifetime curve generation."""
    model_types: List[str] = Field(
        default=["cips-2008"],
        description="List of model types to generate curves for"
    )
    parameter_to_vary: str = Field(
        ...,
        description="Parameter to vary (e.g., 'delta_Tj', 'Tj_max', 't_on')"
    )
    parameter_values: List[float] = Field(
        ...,
        min_length=2,
        description="Values of the parameter to calculate"
    )
    fixed_parameters: Dict[str, float] = Field(
        default_factory=dict,
        description="Fixed parameters for the calculation"
    )
    log_scale_x: bool = Field(default=False, description="Use log scale for x-axis")
    log_scale_y: bool = Field(default=True, description="Use log scale for y-axis")


class LifetimeCurveResponse(BaseModel):
    """Response schema for lifetime curve generation."""
    curves: List[ModelCurve] = Field(..., description="Generated curves for each model")
    x_axis_label: str = Field(..., description="Label for x-axis")
    y_axis_label: str = Field(..., description="Label for y-axis")
    title: str = Field(..., description="Chart title")


# ============================================================================
# Remaining Life Schemas
# ============================================================================

class DegradationHistoryPoint(BaseModel):
    """A single point in degradation history."""
    cycles: float = Field(..., ge=0, description="Cumulative cycles at this point")
    damage: float = Field(..., ge=0, le=1, description="Damage value (0-1) at this point")
    time: Optional[float] = Field(None, ge=0, description="Elapsed time in hours")


class RemainingLifeRequest(BaseModel):
    """Request schema for remaining life evaluation."""
    current_damage: float = Field(..., ge=0, le=1, description="Current accumulated damage (0-1)")
    model_type: str = Field(..., description="Lifetime model to use")
    model_parameters: Dict[str, float] = Field(
        default_factory=dict,
        description="Parameters for the lifetime model"
    )
    operating_conditions: Dict[str, float] = Field(
        ...,
        description="Current operating conditions (range, mean)"
    )
    degradation_history: Optional[List[DegradationHistoryPoint]] = Field(
        None,
        description="Historical degradation data for trend analysis"
    )
    cycle_frequency: float = Field(default=1.0, gt=0, description="Cycles per hour")
    method: str = Field(
        default="auto",
        description="Estimation method: 'auto', 'linear', 'exponential', 'constant'"
    )


class ConfidenceInterval(BaseModel):
    """Confidence interval for remaining life estimate."""
    lower_bound: float = Field(..., description="Lower bound of confidence interval")
    upper_bound: float = Field(..., description="Upper bound of confidence interval")
    confidence_level: float = Field(default=0.95, description="Confidence level (e.g., 0.95 for 95%)")


class RemainingLifeResponse(BaseModel):
    """Response schema for remaining life evaluation."""
    estimated_cycles_remaining: float = Field(..., description="Estimated cycles until failure")
    estimated_time_remaining: float = Field(..., description="Estimated time until failure (hours)")
    estimated_days_remaining: Optional[float] = Field(None, description="Estimated time until failure (days)")
    estimated_years_remaining: Optional[float] = Field(None, description="Estimated time until failure (years)")
    health_index: float = Field(..., ge=0, le=1, description="Health indicator (0-1, 1 = perfect)")
    degradation_rate: float = Field(..., description="Rate of degradation (damage per cycle)")
    confidence_interval: Optional[ConfidenceInterval] = Field(None, description="Confidence bounds")
    method_used: str = Field(..., description="Method used for estimation")
    is_failed: bool = Field(..., description="Whether damage indicates failure")
    damage_state: str = Field(..., description="Current damage state category")
    recommendation: str = Field(..., description="Actionable recommendation")


class DamageRequest(BaseModel):
    """Request schema for damage analysis."""
    cycles: List[CycleCount] = Field(..., description="Rainflow cycle counts")
    sn_curve: SNCurve = Field(..., description="S-N curve parameters")
    damage_model: str = Field(default="MINER", description="Damage accumulation model")
    safety_factor: float = Field(default=1.0, ge=0.1, le=10.0, description="Safety factor for life prediction")


class DamageResult(BaseModel):
    """Individual damage result for a stress level."""
    stress_range: float
    cycles: float
    allowable_cycles: float
    damage_ratio: float
    cumulative_damage: float


class DamageResponse(BaseModel):
    """Response schema for damage analysis."""
    total_damage: float = Field(..., description="Total accumulated damage")
    remaining_life_fraction: float = Field(..., description="Remaining life fraction (1 - total_damage)")
    predicted_cycles: Optional[float] = Field(None, description="Predicted cycles to failure")
    details: List[DamageResult] = Field(..., description="Detailed damage breakdown")
    safety_factor: float = Field(..., description="Applied safety factor")
    is_failed: bool = Field(..., description="Whether damage exceeds 1.0")


class LifetimePredictionRequest(BaseModel):
    """Request for lifetime prediction using CIPS 2008 models."""
    mission_profile_id: int = Field(..., description="Mission profile ID")
    model_type: str = Field(..., description="Lifetime model to use")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Model parameters")
    safety_factor: float = Field(default=1.0, ge=0.1, le=10.0)


class LifetimePredictionResponse(BaseModel):
    """Response for lifetime prediction."""
    predicted_lifetime_years: float
    predicted_lifetime_cycles: Optional[float] = None
    confidence_level: Optional[float] = None
    damage_at_eol: float = Field(..., description="Damage at end of life")
    model_used: str
    safety_factor: float

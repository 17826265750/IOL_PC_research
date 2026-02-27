"""
Pydantic schemas for Rainflow cycle counting.
"""
from pydantic import BaseModel, Field
from typing import Dict, List, Optional


class DataPoint(BaseModel):
    """Single data point for time-series analysis."""
    time: Optional[float] = Field(default=None, description="Time in seconds")
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


class LifeCurvePoint(BaseModel):
    """Single life-curve point for Miner damage estimation."""
    delta_tj: float = Field(..., gt=0, description="Temperature swing ΔTj")
    nf: float = Field(..., gt=0, description="Cycles to failure at this ΔTj")


class FosterElement(BaseModel):
    """Single RC element of a Foster thermal network."""
    R: float = Field(..., gt=0, description="Thermal resistance R (K/W)")
    tau: float = Field(..., gt=0, description="Time constant τ (s)")


class RainflowPipelineRequest(BaseModel):
    """One-stop request: power/zth -> Tj -> rainflow -> damage."""
    # --- Single-source inputs (backward compatible) ---
    junction_temperature: Optional[List[float]] = Field(
        default=None, description="Precomputed junction temperature series")
    power_curve: Optional[List[float]] = Field(
        default=None, description="Power curve time series (W)")
    thermal_impedance_curve: Optional[List[float]] = Field(
        default=None, description="Thermal impedance/response curve (sampled)")
    foster_params: Optional[List[FosterElement]] = Field(
        default=None,
        description="Foster RC network elements (alternative to sampled Zth)")

    # --- Multi-source inputs ---
    power_curves: Optional[List[List[float]]] = Field(
        default=None,
        description="Power curves per heat source [n_sources][n_steps]")
    zth_matrix: Optional[List[List[List[FosterElement]]]] = Field(
        default=None,
        description="Foster RC matrix [n_nodes][n_sources][n_rc_elements]. "
                    "zth_matrix[i][j] = thermal path from source j to node i")
    source_names: Optional[List[str]] = Field(
        default=None,
        description="Names for each heat source, e.g. ['IGBT', 'Diode']")
    target_node: int = Field(
        default=0, ge=0,
        description="Which node's Tj to use for rainflow analysis (0-based)")

    # --- Common thermal params ---
    ambient_temperature: float = Field(
        default=25.0, description="Ambient temperature (°C)")
    response_type: str = Field(
        default='impulse',
        description="Thermal response type: 'impulse' or 'step'")
    dt: float = Field(
        default=1.0, gt=0,
        description="Sampling interval (s) for power / Zth curves")

    # --- Rainflow params ---
    bin_count: int = Field(default=64, ge=8, le=256)
    rearrange: bool = Field(
        default=False,
        description="Rearrange reversals to start at max peak "
                    "(recommended for repeating mission profiles)")
    n_band: int = Field(
        default=20, ge=2, le=200,
        description="Number of bands for From-To matrix")
    y_min: Optional[float] = Field(
        default=None,
        description="Lower bound of temperature band range (auto if None)")
    y_max: Optional[float] = Field(
        default=None,
        description="Upper bound of temperature band range (auto if None)")
    ignore_below: float = Field(
        default=0.0, ge=0,
        description="Ignore cycles with delta_Tj below this value")

    # --- Damage: manual life curve ---
    life_curve: Optional[List[LifeCurvePoint]] = Field(
        default=None, description="ΔTj-Nf life curve for Miner damage")
    reference_delta_tj: Optional[float] = Field(
        default=None, gt=0,
        description="Reference ΔTj for equivalent cycle conversion")

    # --- Damage: lifetime model ---
    lifetime_model: Optional[str] = Field(
        default=None,
        description="Registered lifetime model name, e.g. 'coffin-manson', "
                    "'cips-2008'. When set, model-based CDI replaces life_curve.")
    model_params: Optional[Dict[str, float]] = Field(
        default=None,
        description="Constant model parameters (A, alpha, K, beta1-6, "
                    "Ea, t_on, I, V, D, f …)")
    safety_factor: float = Field(
        default=1.0, gt=0,
        description="Safety factor f_safe for CDI calculation")


class RainflowPipelineResponse(BaseModel):
    """One-stop pipeline result."""
    junction_temperature: List[float]
    thermal_summary: dict = Field(default_factory=dict)
    cycles: List[CycleCount]
    matrix_rows: List[dict]
    total_cycles: float
    max_range: float
    summary: dict = Field(default_factory=dict)
    damage: Optional[dict] = None
    from_to_matrix: Optional[dict] = None
    amplitude_histogram: Optional[dict] = None
    residual: Optional[List[float]] = None
    # --- Multi-source ---
    all_junction_temperatures: Optional[dict] = Field(
        default=None,
        description="Tj series per node: {'IGBT': [...], 'Diode': [...]}")
    # --- Model-based damage ---
    model_damage: Optional[dict] = Field(
        default=None,
        description="Model-based CDI result with per-cycle details")

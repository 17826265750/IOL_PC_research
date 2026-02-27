"""
Rainflow cycle counting endpoints.

Provides endpoints for:
- Rainflow cycle counting from time-series data
- Histogram generation
- Cycle matrix computation
- Equivalent constant amplitude calculation
"""
from fastapi import APIRouter, HTTPException, status
from typing import List, Dict, Any
import numpy as np
import logging

from app.core.rainflow import (
    rainflow_counting,
    find_peaks_and_valleys,
    get_cycle_matrix,
    get_histogram_data,
    calculate_equivalent_constant_amplitude,
    get_cumulative_cycles,
    compute_junction_temperature,
    compute_junction_temperature_foster,
    compute_junction_temperature_multi_source,
    compute_thermal_summary,
    build_cycle_matrix_table,
    estimate_damage_from_life_curve,
    compute_model_based_damage,
    compute_from_to_matrix,
    compute_amplitude_histogram,
    Cycle,
)
from app.schemas.rainflow import (
    RainflowRequest,
    RainflowResponse,
    RainflowHistogramRequest,
    RainflowHistogramResponse,
    CycleCount,
    DataPoint,
    HistogramBin,
    RainflowPipelineRequest,
    RainflowPipelineResponse,
)


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/rainflow", tags=["rainflow"])


@router.post("/count", response_model=RainflowResponse)
async def count_cycles(request: RainflowRequest):
    """
    Perform rainflow cycle counting on time-series data.

    Implements the ASTM E1049 three-point rainflow counting method.
    Returns cycle counts, ranges, means, and summary statistics.
    """
    try:
        # Extract time-series values
        values = [dp.value for dp in request.data_points]

        if len(values) < 3:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="At least 3 data points required for rainflow counting"
            )

        # Perform rainflow counting
        result = rainflow_counting(values, bin_count=request.bin_count)

        # Convert to response format
        cycles = [
            CycleCount(
                stress_range=cycle.range,
                mean_value=cycle.mean,
                cycles=cycle.count
            )
            for cycle in result.cycles
        ]

        total_cycles = sum(cycle.count for cycle in result.cycles)
        max_range = max((cycle.range for cycle in result.cycles), default=0.0)

        # Calculate summary statistics
        ranges = np.array([cycle.range for cycle in result.cycles]) if result.cycles else np.array([])
        summary = {
            "total_cycles": float(total_cycles),
            "unique_cycles": len(result.cycles),
            "max_range": float(max_range) if max_range > 0 else 0.0,
            "mean_range": float(np.mean(ranges)) if len(ranges) > 0 else 0.0,
            "std_range": float(np.std(ranges)) if len(ranges) > 1 else 0.0,
            "residual_points": len(result.residual) if result.residual else 0,
        }

        return RainflowResponse(
            cycles=cycles,
            total_cycles=total_cycles,
            max_range=max_range,
            summary=summary
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in rainflow counting: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Rainflow counting failed: {str(e)}"
        )


@router.post("/histogram")
async def generate_histogram(request: RainflowHistogramRequest) -> RainflowHistogramResponse:
    """
    Generate histogram bins from cycle counts.

    Creates binned cycle distribution for visualization.
    """
    try:
        # Convert schema cycles to core cycles
        cycles = [
            Cycle(
                range=cycle.stress_range,
                mean=cycle.mean_value,
                count=cycle.cycles,
                min_val=cycle.mean_value - cycle.stress_range / 2,
                max_val=cycle.mean_value + cycle.stress_range / 2
            )
            for cycle in request.cycles
        ]

        # Get histogram data
        hist_data = get_histogram_data(cycles, bin_count=request.bin_count)

        # Convert to response format
        bins = []
        for i in range(len(hist_data['counts'])):
            bins.append(HistogramBin(
                range_min=float(hist_data['bins'][i]),
                range_max=float(hist_data['bins'][i + 1]),
                cycle_count=float(hist_data['counts'][i]),
                damage_contribution=None
            ))

        total_cycles = float(sum(hist_data['counts']))

        return RainflowHistogramResponse(
            bins=bins,
            total_cycles=total_cycles,
            cumulative_damage=None
        )

    except Exception as e:
        logger.error(f"Error generating histogram: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Histogram generation failed: {str(e)}"
        )


@router.post("/matrix")
async def get_cycle_matrix_data(
    cycles: List[CycleCount],
    bin_count: int = 64
) -> Dict[str, Any]:
    """
    Generate cycle matrix (range vs mean) for visualization.

    Returns 2D histogram data for heatmap visualization.
    """
    try:
        if bin_count < 8 or bin_count > 256:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="bin_count must be between 8 and 256"
            )

        # Convert to core cycles
        core_cycles = [
            Cycle(
                range=cycle.stress_range,
                mean=cycle.mean_value,
                count=cycle.cycles,
                min_val=cycle.mean_value - cycle.stress_range / 2,
                max_val=cycle.mean_value + cycle.stress_range / 2
            )
            for cycle in cycles
        ]

        # Get cycle matrix
        matrix = get_cycle_matrix(core_cycles, bin_count=bin_count)

        # Find ranges for axes
        ranges = np.array([c.range for c in core_cycles]) if core_cycles else np.array([1])
        means = np.array([c.mean for c in core_cycles]) if core_cycles else np.array([0])

        range_min, range_max = float(ranges.min()), float(ranges.max())
        mean_min, mean_max = float(means.min()), float(means.max())

        return {
            "matrix": matrix.tolist(),
            "bin_count": bin_count,
            "range_axis": {"min": range_min, "max": range_max},
            "mean_axis": {"min": mean_min, "max": mean_max},
            "shape": matrix.shape
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating cycle matrix: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Cycle matrix generation failed: {str(e)}"
        )


@router.post("/equivalent")
async def calculate_equivalent_amplitude(
    cycles: List[CycleCount],
    exponent: float = 3.0
) -> Dict[str, float]:
    """
    Calculate equivalent constant amplitude stress range.

    Uses the damage-equivalent approach where variable amplitude
    loading is converted to an equivalent constant amplitude.
    """
    try:
        # Convert to core cycles
        core_cycles = [
            Cycle(
                range=cycle.stress_range,
                mean=cycle.mean_value,
                count=cycle.cycles,
                min_val=cycle.mean_value - cycle.stress_range / 2,
                max_val=cycle.mean_value + cycle.stress_range / 2
            )
            for cycle in cycles
        ]

        equivalent = calculate_equivalent_constant_amplitude(core_cycles, exponent)

        return {
            "equivalent_range": float(equivalent),
            "exponent": exponent
        }

    except Exception as e:
        logger.error(f"Error calculating equivalent amplitude: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Equivalent amplitude calculation failed: {str(e)}"
        )


@router.post("/cumulative")
async def get_cumulative_distribution(cycles: List[CycleCount]) -> Dict[str, Any]:
    """
    Calculate cumulative cycle count sorted by range.

    Returns cumulative distribution function for assessing
    contribution of larger vs smaller cycles.
    """
    try:
        # Convert to core cycles
        core_cycles = [
            Cycle(
                range=cycle.stress_range,
                mean=cycle.mean_value,
                count=cycle.cycles,
                min_val=cycle.mean_value - cycle.stress_range / 2,
                max_val=cycle.mean_value + cycle.stress_range / 2
            )
            for cycle in cycles
        ]

        cumulative = get_cumulative_cycles(core_cycles)

        # Sort cycles by range for paired response
        sorted_cycles = sorted(core_cycles, key=lambda c: c.range)

        return {
            "ranges": [float(c.range) for c in sorted_cycles],
            "cumulative_counts": cumulative.tolist(),
            "total_cumulative": float(cumulative[-1]) if len(cumulative) > 0 else 0.0
        }

    except Exception as e:
        logger.error(f"Error calculating cumulative distribution: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Cumulative distribution calculation failed: {str(e)}"
        )


@router.post("/peaks")
async def extract_peaks_valleys(data_points: List[DataPoint]) -> Dict[str, Any]:
    """
    Extract peak and valley points from time series.

    Removes consecutive points that don't represent reversals.
    """
    try:
        values = [dp.value for dp in data_points]

        if len(values) < 2:
            return {
                "peaks_valleys": values,
                "count": len(values),
                "reduction": 0
            }

        extracted = find_peaks_and_valleys(values)

        return {
            "peaks_valleys": extracted,
            "count": len(extracted),
            "reduction": len(values) - len(extracted),
            "original_count": len(values)
        }

    except Exception as e:
        logger.error(f"Error extracting peaks and valleys: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Peak/valley extraction failed: {str(e)}"
        )


@router.post("/analyze")
async def analyze_time_series(data_points: List[DataPoint]) -> Dict[str, Any]:
    """
    Perform comprehensive time-series analysis.

    Returns peaks/valleys, rainflow cycles, and statistics.
    """
    try:
        values = [dp.value for dp in data_points]

        if len(values) < 3:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="At least 3 data points required"
            )

        # Extract peaks and valleys
        peaks_valleys = find_peaks_and_valleys(values)

        # Perform rainflow counting
        result = rainflow_counting(values)

        # Calculate statistics
        arr = np.array(values)
        stats = {
            "count": len(values),
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "range": float(np.max(arr) - np.min(arr)),
            "peaks_valleys_count": len(peaks_valleys),
            "rainflow_cycles": len(result.cycles),
            "total_cycles": float(sum(c.count for c in result.cycles))
        }

        return {
            "statistics": stats,
            "peaks_valleys": peaks_valleys,
            "cycles": [
                {
                    "range": float(c.range),
                    "mean": float(c.mean),
                    "count": float(c.count),
                    "min": float(c.min_val),
                    "max": float(c.max_val)
                }
                for c in result.cycles
            ],
            "residual": result.residual
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in time-series analysis: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Analysis failed: {str(e)}"
        )


@router.post('/pipeline', response_model=RainflowPipelineResponse)
async def run_rainflow_pipeline(request: RainflowPipelineRequest) -> RainflowPipelineResponse:
    """Run one-stop workflow: input -> Tj -> rainflow -> matrix -> damage.

    Supports four input modes:
    1. Pre-computed junction_temperature series.
    2. power_curve + foster_params  (Foster RC network — single source).
    3. power_curve + thermal_impedance_curve (convolution — single source).
    4. power_curves + zth_matrix  (multi-heat-source matrix Foster convolution).

    Damage calculation supports two strategies:
    a. Manual life_curve (ΔTj-Nf log-interpolation).
    b. Registered lifetime_model with model_params (ModelFactory).
    """
    try:
        all_tj: dict | None = None  # multi-source results

        # ---- Step 1: obtain Tj series ----------------------------------
        if request.junction_temperature is not None:
            tj_series = [float(v) for v in request.junction_temperature]

        elif (request.power_curves is not None
              and request.zth_matrix is not None):
            # ---- Mode 4: multi-source matrix Foster convolution --------
            zth_mat = [
                [
                    [{'R': e.R, 'tau': e.tau} for e in cell]
                    for cell in row
                ]
                for row in request.zth_matrix
            ]
            all_tj_list = compute_junction_temperature_multi_source(
                power_curves=request.power_curves,
                zth_matrix=zth_mat,
                ambient_temperature=request.ambient_temperature,
                dt=request.dt,
            )
            names = request.source_names or [
                f'Node_{i}' for i in range(len(all_tj_list))
            ]
            all_tj = {
                names[i]: all_tj_list[i]
                for i in range(len(all_tj_list))
            }
            # Select target node for downstream analysis
            target = min(request.target_node, len(all_tj_list) - 1)
            tj_series = all_tj_list[target]

        elif request.foster_params is not None and request.power_curve is not None:
            tj_series = compute_junction_temperature_foster(
                power_curve=request.power_curve,
                foster_params=[{'R': e.R, 'tau': e.tau}
                               for e in request.foster_params],
                ambient_temperature=request.ambient_temperature,
                dt=request.dt,
            )
        elif (request.power_curve is not None
              and request.thermal_impedance_curve is not None):
            tj_series = compute_junction_temperature(
                power_curve=request.power_curve,
                thermal_impedance_curve=request.thermal_impedance_curve,
                ambient_temperature=request.ambient_temperature,
                response_type=request.response_type,
                dt=request.dt,
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=('Provide junction_temperature, '
                        'power_curve + foster_params, '
                        'power_curve + thermal_impedance_curve, '
                        'or power_curves + zth_matrix'),
            )

        if len(tj_series) < 3:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail='At least 3 junction temperature points are required',
            )

        # ---- Step 2: thermal summary -----------------------------------
        thermal_summary = compute_thermal_summary(tj_series)

        # ---- Step 3: rainflow counting ---------------------------------
        rf_result = rainflow_counting(
            tj_series,
            bin_count=request.bin_count,
            rearrange=request.rearrange,
        )

        cycles = [
            CycleCount(
                stress_range=float(c.range),
                mean_value=float(c.mean),
                cycles=float(c.count),
            )
            for c in rf_result.cycles
        ]

        total_cycles = float(sum(c.count for c in rf_result.cycles))
        max_range = float(max((c.range for c in rf_result.cycles), default=0.0))
        matrix_rows = build_cycle_matrix_table(rf_result.cycles, decimals=2)

        ranges = (np.array([c.range for c in rf_result.cycles])
                  if rf_result.cycles else np.array([]))
        summary = {
            'total_cycles': total_cycles,
            'unique_cycles': len(rf_result.cycles),
            'max_range': max_range,
            'mean_range': float(np.mean(ranges)) if len(ranges) > 0 else 0.0,
            'std_range': float(np.std(ranges)) if len(ranges) > 1 else 0.0,
            'residual_points': len(rf_result.residual) if rf_result.residual else 0,
        }

        # ---- Step 3b: From-To matrix & amplitude histogram -------------
        from_to = compute_from_to_matrix(
            rf_result.reversals or [],
            n_band=request.n_band,
            y_min=request.y_min,
            y_max=request.y_max,
        )
        amp_hist = compute_amplitude_histogram(
            rf_result.cycles,
            n_bins=request.n_band,
            ignore_below=request.ignore_below,
        )

        # ---- Step 4: Miner damage (optional) ---------------------------
        damage = None
        model_damage = None

        if request.lifetime_model and request.model_params:
            # Strategy B: model-based CDI
            model_damage = compute_model_based_damage(
                rf_result.cycles,
                model_name=request.lifetime_model,
                model_params=request.model_params,
                safety_factor=request.safety_factor,
            )
            # Also expose as `damage` for backward compat display
            damage = model_damage

        elif request.life_curve:
            # Strategy A: manual life-curve interpolation
            damage = estimate_damage_from_life_curve(
                rf_result.cycles,
                life_curve=[{'delta_tj': item.delta_tj, 'nf': item.nf}
                            for item in request.life_curve],
                reference_delta_tj=request.reference_delta_tj,
            )

        return RainflowPipelineResponse(
            junction_temperature=tj_series,
            thermal_summary=thermal_summary,
            cycles=cycles,
            matrix_rows=matrix_rows,
            total_cycles=total_cycles,
            max_range=max_range,
            summary=summary,
            damage=damage,
            from_to_matrix=from_to,
            amplitude_histogram=amp_hist,
            residual=rf_result.residual,
            all_junction_temperatures=all_tj,
            model_damage=model_damage,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f'Error in rainflow pipeline: {e}')
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f'Rainflow pipeline failed: {str(e)}',
        )

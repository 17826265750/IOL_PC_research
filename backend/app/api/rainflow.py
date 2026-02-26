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
    find_range_mean_pairs,
    calculate_equivalent_constant_amplitude,
    get_cumulative_cycles,
    Cycle,
    RainflowResult
)
from app.schemas.rainflow import (
    RainflowRequest,
    RainflowResponse,
    RainflowHistogramRequest,
    RainflowHistogramResponse,
    CycleCount,
    DataPoint,
    HistogramBin
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
        counts = np.array([cycle.count for cycle in result.cycles]) if result.cycles else np.array([])

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
            "exponent": exponent,
            "note": "Equivalent constant amplitude that causes same damage"
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

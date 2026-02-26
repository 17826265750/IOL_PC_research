"""
Rainflow cycle counting algorithm implementation per ASTM E1049.
Used for extracting stress/temperature cycles from irregular time series.

The algorithm implements the three-point counting method from ASTM E1049-85
Section 5.4.4, which is the most common rainflow cycle counting technique.

References:
    ASTM E1049-85 (2017), "Standard Practices for Cycle Counting in
    Fatigue Analysis", DOI: 10.1520/E1049-85R17
"""
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
import warnings


@dataclass
class Cycle:
    """Represents a counted cycle.

    Attributes:
        range: Peak-to-peak range (stress/temperature difference)
        mean: Mean value of the cycle
        count: Cycle count (0.5 for half cycles, 1.0 for full)
        min_val: Minimum value of the cycle
        max_val: Maximum value of the cycle
    """
    range: float
    mean: float
    count: float
    min_val: float
    max_val: float

    def __repr__(self) -> str:
        return (f"Cycle(range={self.range:.4g}, mean={self.mean:.4g}, "
                f"count={self.count:.4g}, min={self.min_val:.4g}, "
                f"max={self.max_val:.4g})")


@dataclass
class RainflowResult:
    """Complete rainflow counting result.

    Attributes:
        cycles: List of Cycle objects
        cycle_matrix: 2D histogram matrix (range vs mean)
        residual: Residual points that couldn't form complete cycles
    """
    cycles: List[Cycle]
    cycle_matrix: Optional[np.ndarray] = None
    residual: Optional[List[float]] = None


def rainflow_counting(data: List[float], bin_count: int = 64) -> RainflowResult:
    """
    Perform rainflow cycle counting on time series data.

    Implements the "three-point" rainflow method from ASTM E1049 for cycle
    counting. This extracts complete and half cycles from irregular loading
    histories.

    The algorithm works by:
    1. Extracting peaks and valleys from the time series
    2. Using a three-point comparison to identify cycles
    3. Storing complete cycles and continuing with the remainder

    Args:
        data: List of stress/temperature values
        bin_count: Number of bins for cycle matrix (default: 64)

    Returns:
        RainflowResult containing cycles, matrix, and residuals

    Examples:
        >>> data = [0, 10, -5, 8, -3, 7, -2, 5]
        >>> result = rainflow_counting(data)
        >>> for cycle in result.cycles:
        ...     print(cycle)
    """
    if not data:
        warnings.warn("Empty data provided to rainflow_counting")
        return RainflowResult(cycles=[], cycle_matrix=None, residual=[])

    data_array = np.asarray(data, dtype=float)

    if len(data_array) == 1:
        warnings.warn("Single data point - no cycles possible")
        return RainflowResult(cycles=[], cycle_matrix=None, residual=data_array.tolist())

    if np.allclose(data_array, data_array[0]):
        warnings.warn("All data points are identical - no cycles")
        return RainflowResult(cycles=[], cycle_matrix=None, residual=data_array.tolist())

    # Extract peaks and valleys
    peaks_valleys = find_peaks_and_valleys(data_array.tolist())

    if len(peaks_valleys) < 3:
        # Need at least 3 points to form a cycle
        return RainflowResult(cycles=[], residual=peaks_valleys)

    # Perform the three-point rainflow counting
    cycles, residual = _three_point_rainflow(peaks_valleys)

    # Generate cycle matrix if cycles exist
    cycle_matrix = None
    if cycles and bin_count > 0:
        cycle_matrix = get_cycle_matrix(cycles, bin_count)

    return RainflowResult(cycles=cycles, cycle_matrix=cycle_matrix, residual=residual)


def _three_point_rainflow(data: List[float]) -> Tuple[List[Cycle], List[float]]:
    """
    Internal implementation of ASTM E1049 three-point rainflow counting.

    Args:
        data: List of peak and valley values

    Returns:
        Tuple of (cycles list, residual list)
    """
    if len(data) < 3:
        return [], data[:]

    # Convert to numpy array for easier manipulation
    Y = np.array(data, dtype=float)
    k = 0
    cycles: List[Cycle] = []

    while True:
        # Check if we can form a cycle
        while k < len(Y) - 2:
            # Three consecutive points
            A = Y[k]
            B = Y[k + 1]
            C = Y[k + 2]

            range_AB = abs(B - A)
            range_BC = abs(C - B)

            # Check if a cycle can be formed (ASTM E1049 rule)
            if range_AB >= range_BC:
                # Form a cycle from points A, B, C
                cycle_range = range_AB
                cycle_mean = (A + C) / 2
                cycle_min = min(A, C)
                cycle_max = max(A, C)

                # Check if this is the last three points
                if k == len(Y) - 3:
                    # Last cycle
                    cycles.append(Cycle(
                        range=cycle_range,
                        mean=cycle_mean,
                        count=1.0,  # Full cycle
                        min_val=cycle_min,
                        max_val=cycle_max
                    ))
                    # Remove last two points
                    Y = Y[:k + 1]
                    k = 0
                else:
                    # Inner cycle
                    cycles.append(Cycle(
                        range=cycle_range,
                        mean=cycle_mean,
                        count=0.5,  # Half cycle
                        min_val=cycle_min,
                        max_val=cycle_max
                    ))
                    # Remove middle point
                    Y = np.delete(Y, k + 1)
                    k = max(0, k - 1)  # Go back to check previous points
            else:
                k += 1

        # Check if we've processed all possible cycles
        if len(Y) < 3:
            break

        # Handle remaining points
        k = 0

    return cycles, Y.tolist()


def find_peaks_and_valleys(data: List[float]) -> List[float]:
    """
    Extract peak and valley points from time series.

    This function removes consecutive points that don't represent reversals
    in the loading direction. Only points where the slope changes sign
    (or the endpoints) are kept.

    Args:
        data: List of stress/temperature values

    Returns:
        List of peak and valley values

    Examples:
        >>> data = [0, 2, 5, 3, -1, 2, 4, 1]
        >>> find_peaks_and_valleys(data)
        [0, 5, -1, 4, 1]
    """
    if not data:
        return []

    data_array = np.asarray(data, dtype=float)

    if len(data_array) <= 2:
        return data_array.tolist()

    # Calculate differences between consecutive points
    diffs = np.diff(data_array)

    # Find where the sign changes (including zero handling)
    # A peak/valley occurs where diff changes sign
    sign_changes = np.zeros(len(data_array), dtype=bool)
    sign_changes[0] = True  # First point is always included
    sign_changes[-1] = True  # Last point is always included

    for i in range(1, len(diffs)):
        # Check for sign change or zero difference
        if diffs[i] == 0:
            # Continue in same direction as last non-zero
            j = i - 1
            while j >= 0 and diffs[j] == 0:
                j -= 1
            if j < 0:
                sign_changes[i] = True
            # else: not a turning point
        elif diffs[i - 1] == 0:
            # Previous was zero - this could be a turning point
            j = i - 2
            while j >= 0 and diffs[j] == 0:
                j -= 1
            if j < 0 or diffs[j] * diffs[i] < 0:
                sign_changes[i] = True
        elif diffs[i - 1] * diffs[i] < 0:
            # Sign change - this is a peak or valley
            sign_changes[i] = True

    # Filter the data
    result = data_array[sign_changes].tolist()

    # Handle edge case of all equal values
    if len(result) == 1 and len(data_array) > 1:
        return [data_array[0], data_array[-1]]

    return result


def get_cycle_matrix(cycles: List[Cycle], bin_count: int = 64) -> np.ndarray:
    """
    Generate cycle matrix (range vs mean) for visualization.

    Creates a 2D histogram where:
    - X-axis represents the mean stress/temperature
    - Y-axis represents the range (peak-to-peak)
    - Z-values represent the accumulated cycle count

    Args:
        cycles: List of Cycle objects from rainflow_counting
        bin_count: Number of bins for each dimension (default: 64)

    Returns:
        2D numpy array with cycle counts in each bin

    Examples:
        >>> cycles = [Cycle(10, 50, 1.0, 45, 55), Cycle(15, 60, 0.5, 52, 67)]
        >>> matrix = get_cycle_matrix(cycles, bin_count=32)
    """
    if not cycles:
        return np.zeros((bin_count, bin_count))

    ranges = np.array([c.range for c in cycles])
    means = np.array([c.mean for c in cycles])
    weights = np.array([c.count for c in cycles])

    if len(ranges) == 0:
        return np.zeros((bin_count, bin_count))

    # Determine bin edges
    range_min, range_max = ranges.min(), ranges.max()
    mean_min, mean_max = means.min(), means.max()

    # Add small margin to avoid edge cases
    if range_max > range_min:
        range_min -= 0.01 * (range_max - range_min)
        range_max += 0.01 * (range_max - range_min)
    else:
        range_min -= 1.0
        range_max += 1.0

    if mean_max > mean_min:
        mean_min -= 0.01 * (mean_max - mean_min)
        mean_max += 0.01 * (mean_max - mean_min)
    else:
        mean_min -= 1.0
        mean_max += 1.0

    range_edges = np.linspace(range_min, range_max, bin_count + 1)
    mean_edges = np.linspace(mean_min, mean_max, bin_count + 1)

    # Create 2D histogram
    matrix, _, _ = np.histogram2d(
        means, ranges,
        bins=[mean_edges, range_edges],
        weights=weights
    )

    # Transpose to get range on y-axis (rows) and mean on x-axis (cols)
    return matrix.T


def get_histogram_data(cycles: List[Cycle], bin_count: int = 64) -> dict:
    """
    Generate histogram data for cycle distribution.

    Creates a histogram of cycle ranges weighted by their counts,
    useful for visualizing the distribution of cycle magnitudes.

    Args:
        cycles: List of Cycle objects from rainflow_counting
        bin_count: Number of histogram bins (default: 64)

    Returns:
        dict with 'bins' (edge values) and 'counts' arrays,
        and 'centers' (bin center values)

    Examples:
        >>> cycles = [Cycle(10, 50, 1.0, 45, 55), Cycle(15, 60, 0.5, 52, 67)]
        >>> hist = get_histogram_data(cycles, bin_count=10)
        >>> hist['bins']      # Array of bin edges
        >>> hist['counts']    # Array of cycle counts per bin
    """
    if not cycles:
        return {
            'bins': np.zeros(bin_count + 1),
            'counts': np.zeros(bin_count),
            'centers': np.zeros(bin_count)
        }

    ranges = np.array([c.range for c in cycles])
    weights = np.array([c.count for c in cycles])

    if len(ranges) == 0:
        return {
            'bins': np.zeros(bin_count + 1),
            'counts': np.zeros(bin_count),
            'centers': np.zeros(bin_count)
        }

    range_min, range_max = ranges.min(), ranges.max()

    if range_max > range_min:
        range_min -= 0.01 * (range_max - range_min)
        range_max += 0.01 * (range_max - range_min)
    else:
        range_min -= 1.0
        range_max += 1.0

    counts, bins = np.histogram(
        ranges,
        bins=bin_count,
        range=(range_min, range_max),
        weights=weights
    )

    centers = (bins[:-1] + bins[1:]) / 2

    return {
        'bins': bins,
        'counts': counts,
        'centers': centers
    }


def find_range_mean_pairs(cycles: List[Cycle]) -> List[Tuple[float, float, float]]:
    """
    Extract (range, mean, count) tuples from cycles.

    Useful for exporting cycle data or creating custom visualizations.

    Args:
        cycles: List of Cycle objects

    Returns:
        List of (range, mean, count) tuples
    """
    return [(c.range, c.mean, c.count) for c in cycles]


def calculate_equivalent_constant_amplitude(
    cycles: List[Cycle],
    exponent: float = 3.0
) -> float:
    """
    Calculate equivalent constant amplitude stress range.

    Uses the damage-equivalent approach where a variable amplitude
    loading is converted to an equivalent constant amplitude that
    would cause the same damage.

    Based on: (Σ n_i * ΔS_i^m)^(1/m) where m is the S-N exponent

    Args:
        cycles: List of Cycle objects
        exponent: S-N curve exponent (default: 3.0 for steel)

    Returns:
        Equivalent constant amplitude range
    """
    if not cycles:
        return 0.0

    total_cycles = sum(c.count for c in cycles)

    if total_cycles == 0:
        return 0.0

    weighted_sum = sum(c.count * (c.range ** exponent) for c in cycles)
    equivalent_range = (weighted_sum / total_cycles) ** (1 / exponent)

    return equivalent_range


def get_cumulative_cycles(cycles: List[Cycle]) -> np.ndarray:
    """
    Calculate cumulative cycle count sorted by range.

    Returns a cumulative distribution function useful for
    assessing the contribution of larger vs smaller cycles.

    Args:
        cycles: List of Cycle objects

    Returns:
        Array of cumulative cycle counts (sorted by increasing range)
    """
    if not cycles:
        return np.array([])

    # Sort cycles by range
    sorted_cycles = sorted(cycles, key=lambda c: c.range)

    # Get ranges and cumulative counts
    ranges = np.array([c.range for c in sorted_cycles])
    counts = np.array([c.count for c in sorted_cycles])
    cumulative = np.cumsum(counts)

    return cumulative

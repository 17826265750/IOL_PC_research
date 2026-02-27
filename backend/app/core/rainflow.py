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
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
import warnings
import logging

logger = logging.getLogger(__name__)


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
    """Complete rainflow counting result."""
    cycles: List[Cycle]
    cycle_matrix: Optional[np.ndarray] = None
    residual: Optional[List[float]] = None
    reversals: Optional[List[float]] = None


def rainflow_counting(
    data: List[float],
    bin_count: int = 64,
    rearrange: bool = False,
) -> RainflowResult:
    """
    Perform rainflow cycle counting on time series data.

    Implements the three-point rainflow method from ASTM E1049-85 §5.4.4.
    Extracts full and half cycles from irregular loading histories.

    Args:
        data: List of stress/temperature values.
        bin_count: Number of bins for cycle matrix (default: 64).
        rearrange: If True, rotate the reversal sequence so that it
            starts (and ends) at the highest peak.  Recommended for
            repeating mission profiles — it converts the outermost
            cycle from two half-cycles into one full cycle.

    Returns:
        RainflowResult containing cycles, matrix, and residuals.
    """
    if not data:
        warnings.warn("Empty data provided to rainflow_counting")
        return RainflowResult(cycles=[], cycle_matrix=None, residual=[])

    data_array = np.asarray(data, dtype=float)

    if len(data_array) == 1:
        warnings.warn("Single data point - no cycles possible")
        return RainflowResult(cycles=[], cycle_matrix=None,
                              residual=data_array.tolist(),
                              reversals=data_array.tolist())

    if np.allclose(data_array, data_array[0]):
        warnings.warn("All data points are identical - no cycles")
        return RainflowResult(cycles=[], cycle_matrix=None,
                              residual=data_array.tolist(),
                              reversals=data_array.tolist())

    # Extract peaks and valleys
    peaks_valleys = find_peaks_and_valleys(data_array.tolist())

    if len(peaks_valleys) < 3:
        # Need at least 3 points to form a cycle
        return RainflowResult(cycles=[], residual=peaks_valleys,
                              reversals=peaks_valleys)

    # Optional: rearrange for repeating profiles
    if rearrange and len(peaks_valleys) > 2:
        peaks_valleys = _rearrange_to_start_at_max(peaks_valleys)

    # Perform the three-point rainflow counting (ASTM E1049)
    cycles, residual = _three_point_rainflow(peaks_valleys)

    # Generate cycle matrix if cycles exist
    cycle_matrix = None
    if cycles and bin_count > 0:
        cycle_matrix = get_cycle_matrix(cycles, bin_count)

    return RainflowResult(cycles=cycles, cycle_matrix=cycle_matrix, residual=residual, reversals=peaks_valleys)


def _three_point_rainflow(data: List[float]) -> Tuple[List[Cycle], List[float]]:
    """
    ASTM E1049-85 §5.4.4 Rainflow Counting — three-point method.

    The algorithm maintains a list of reversal points and repeatedly
    applies the following rule to the three most-recent entries
    (denoted A, B, C from oldest to newest):

        Y = |B − A|   (the "previous" range)
        X = |C − B|   (the "current" range)

        • X ≥ Y → count Y.  Half-cycle if Y starts at the very first
                   reversal; full cycle otherwise.  Remove A and B
                   from the list and re-check.
        • X < Y → advance to the next reversal point.

    After all points have been examined, each adjacent pair still on
    the stack is counted as a half-cycle (Step 6 of ASTM E1049).

    Args:
        data: List of reversal (peak/valley) values.

    Returns:
        Tuple of (cycles list, residual stack)
    """
    if len(data) < 3:
        return [], data[:]

    stack = list(data)
    cycles: List[Cycle] = []

    idx = 2  # need ≥ 3 points on the stack before we can test

    while idx < len(stack):
        A = stack[idx - 2]
        B = stack[idx - 1]
        C = stack[idx]

        Y = abs(B - A)  # older ("previous") range
        X = abs(C - B)  # newer ("current")  range

        if X >= Y:
            # Y closes ─────────────────────────────────────────
            is_starting = (idx - 2 == 0)
            cycles.append(Cycle(
                range=float(Y),
                mean=float((A + B) / 2.0),
                count=0.5 if is_starting else 1.0,
                min_val=float(min(A, B)),
                max_val=float(max(A, B)),
            ))
            # remove the two points that formed Y
            del stack[idx - 2]   # removes A
            del stack[idx - 2]   # removes B (shifted into A's slot)
            idx = max(2, idx - 2)  # step back and re-check
        else:
            idx += 1  # X < Y → advance

    # Step 6 — residual: count each adjacent pair as a half-cycle
    for j in range(len(stack) - 1):
        a, b = stack[j], stack[j + 1]
        cycles.append(Cycle(
            range=float(abs(b - a)),
            mean=float((a + b) / 2.0),
            count=0.5,
            min_val=float(min(a, b)),
            max_val=float(max(a, b)),
        ))

    return cycles, stack


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


def compute_junction_temperature(
    power_curve: List[float],
    thermal_impedance_curve: List[float],
    ambient_temperature: float = 25.0,
    response_type: str = 'impulse',
    dt: float = 1.0,
) -> List[float]:
    """Compute junction temperature from power and thermal impedance curve.

    Args:
        power_curve: Power time series (W), sampled at interval *dt*.
        thermal_impedance_curve: Thermal impedance/response sequence,
            sampled at the same interval *dt*.
        ambient_temperature: Ambient temperature (°C).
        response_type: ``'impulse'`` — *thermal_impedance_curve* is the
            impulse response h(t);  ``'step'`` — it is the unit-step
            (thermal impedance Zth) response.
        dt: Sampling interval in seconds (default 1.0).  Only used
            when *response_type* is ``'impulse'``.

    Returns:
        Junction temperature series Tj (°C).
    """
    if not power_curve:
        return []
    if not thermal_impedance_curve:
        raise ValueError("thermal_impedance_curve cannot be empty")

    p = np.asarray(power_curve, dtype=float)
    z = np.asarray(thermal_impedance_curve, dtype=float)

    if len(p) < 2:
        return (p * z[0] + ambient_temperature).tolist()

    if response_type not in ('impulse', 'step'):
        raise ValueError("response_type must be 'impulse' or 'step'")

    if response_type == 'step':
        # Convert step response Zth(t) to impulse response via
        # finite difference.  The dt factors cancel in the
        # convolution, so no extra scaling is needed.
        h = np.diff(np.concatenate(([0.0], z)))
        delta_t_arr = np.convolve(p, h, mode='full')[:len(p)]
    else:
        # Impulse response: ΔT[n] = dt · Σ_k P[k]·h[n-k]
        delta_t_arr = np.convolve(p, z, mode='full')[:len(p)] * dt

    tj = delta_t_arr + float(ambient_temperature)
    return tj.tolist()


def compute_junction_temperature_foster(
    power_curve: List[float],
    foster_params: List[Dict[str, float]],
    ambient_temperature: float = 25.0,
    dt: float = 1.0,
) -> List[float]:
    """Compute Tj using a Foster RC thermal network (state-space simulation).

    Each element of *foster_params* must have keys ``'R'`` (K/W) and
    ``'tau'`` (s).  The Foster model:

        Zth(t) = Σ Rᵢ · (1 − exp(−t / τᵢ))

    is evaluated via the exact discrete-time state-space update::

        Tᵢ[n] = Tᵢ[n−1]·exp(−Δt/τᵢ) + Rᵢ·(1 − exp(−Δt/τᵢ))·P[n]
        Tj[n]  = Tₐₘᵦ + Σᵢ Tᵢ[n]

    which is numerically stable and exact for piecewise-constant P.

    Args:
        power_curve: Power dissipation time series (W).
        foster_params: List of ``{'R': float, 'tau': float}``.
        ambient_temperature: Ambient temperature (°C).
        dt: Sampling interval (s).

    Returns:
        Junction temperature series Tj (°C).
    """
    if not power_curve:
        return []
    if not foster_params:
        raise ValueError('foster_params cannot be empty')

    p = np.asarray(power_curve, dtype=float)
    n_steps = len(p)
    n_rc = len(foster_params)

    # Pre-compute per-element constants
    exp_decay = np.array(
        [np.exp(-dt / rc['tau']) for rc in foster_params], dtype=float
    )
    gain = np.array(
        [rc['R'] * (1.0 - np.exp(-dt / rc['tau'])) for rc in foster_params],
        dtype=float,
    )

    states = np.zeros(n_rc, dtype=float)
    tj = np.empty(n_steps, dtype=float)

    for n in range(n_steps):
        states = states * exp_decay + gain * p[n]
        tj[n] = ambient_temperature + states.sum()

    return tj.tolist()


def compute_junction_temperature_multi_source(
    power_curves: List[List[float]],
    zth_matrix: List[List[List[Dict[str, float]]]],
    ambient_temperature: float = 25.0,
    dt: float = 1.0,
) -> List[List[float]]:
    """Compute Tj for a multi-heat-source system via matrix Foster RC convolution.

    Supports thermal coupling between multiple heat sources (e.g. IGBT + Diode
    in a half-bridge module).  The thermal network is described by a matrix of
    Foster RC networks::

        [Zth_1→1  Zth_2→1]   [P_1]   [ΔT_1]
        [Zth_1→2  Zth_2→2] × [P_2] = [ΔT_2]

    Each cell ``zth_matrix[i][j]`` is a list of Foster RC elements
    ``{'R': float, 'tau': float}`` for the thermal path from source *j*
    to node *i*.  The state-space update per element *k* is::

        state_{j→i,k}[n] = state_{j→i,k}[n-1]·exp(-dt/τ_k)
                          + R_k·(1 − exp(-dt/τ_k))·P_j[n]

    Temperature at node *i*::

        T_i[n] = T_amb + Σ_j Σ_k state_{j→i,k}[n]

    Args:
        power_curves: Power time series per source [n_sources][n_steps].
        zth_matrix: Foster RC matrix [n_nodes][n_sources][n_rc_elements].
        ambient_temperature: Ambient temperature (°C).
        dt: Sampling interval (s).

    Returns:
        Tj time series per node [n_nodes][n_steps].
    """
    n_sources = len(power_curves)
    if n_sources == 0:
        raise ValueError('power_curves cannot be empty')
    n_nodes = len(zth_matrix)
    if n_nodes == 0:
        raise ValueError('zth_matrix cannot be empty')
    for row_idx, row in enumerate(zth_matrix):
        if len(row) != n_sources:
            raise ValueError(
                f'zth_matrix row {row_idx} has {len(row)} entries, '
                f'expected {n_sources} (one per source)'
            )

    # Convert power curves to numpy arrays; pad shorter ones to common length
    p_arrays = [np.asarray(pc, dtype=float) for pc in power_curves]
    n_steps = max(len(pa) for pa in p_arrays)
    for idx in range(n_sources):
        if len(p_arrays[idx]) < n_steps:
            p_arrays[idx] = np.pad(
                p_arrays[idx], (0, n_steps - len(p_arrays[idx]))
            )

    # Pre-compute exponential decay and gain for every (node, source) cell
    cells: List[List[Optional[Dict[str, np.ndarray]]]] = []
    for i in range(n_nodes):
        row = []
        for j in range(n_sources):
            rc_list = zth_matrix[i][j]
            if not rc_list:
                row.append(None)
                continue
            exp_decay = np.array(
                [np.exp(-dt / rc['tau']) for rc in rc_list], dtype=float
            )
            gain = np.array(
                [rc['R'] * (1.0 - np.exp(-dt / rc['tau'])) for rc in rc_list],
                dtype=float,
            )
            row.append({'exp_decay': exp_decay, 'gain': gain,
                        'n_rc': len(rc_list)})
        cells.append(row)

    # State-space simulation
    result: List[List[float]] = []
    for i in range(n_nodes):
        states = [
            np.zeros(cells[i][j]['n_rc'], dtype=float) if cells[i][j] else None
            for j in range(n_sources)
        ]
        tj = np.empty(n_steps, dtype=float)
        for n in range(n_steps):
            temp_sum = 0.0
            for j in range(n_sources):
                cell = cells[i][j]
                if cell is None:
                    continue
                st = states[j]
                st[:] = st * cell['exp_decay'] + cell['gain'] * p_arrays[j][n]
                temp_sum += st.sum()
            tj[n] = ambient_temperature + temp_sum
        result.append(tj.tolist())

    return result


def _rearrange_to_start_at_max(reversals: List[float]) -> List[float]:
    """Rotate a reversal sequence so it starts (and ends) at the
    highest peak.  For repeating mission profiles this ensures the
    outermost cycle is counted as a full cycle instead of two halves."""
    if len(reversals) <= 2:
        return reversals
    idx = int(np.argmax(reversals))
    # wrap: [peak → end] + [start+1 → peak] + [peak]
    rotated = reversals[idx:] + reversals[1:idx + 1]
    return rotated


def compute_thermal_summary(tj_series: List[float]) -> Dict[str, float]:
    """Compute thermal-design statistics from a Tj series.

    Returns dict with keys:
        tj_max, tj_min, tj_mean, tj_range, delta_tj_max
    """
    if not tj_series:
        return {
            'tj_max': 0.0, 'tj_min': 0.0, 'tj_mean': 0.0,
            'tj_range': 0.0, 'delta_tj_max': 0.0,
        }
    arr = np.asarray(tj_series, dtype=float)
    tj_max = float(arr.max())
    tj_min = float(arr.min())
    return {
        'tj_max': tj_max,
        'tj_min': tj_min,
        'tj_mean': float(arr.mean()),
        'tj_range': tj_max - tj_min,
        'delta_tj_max': tj_max - tj_min,
    }


def compute_from_to_matrix(
    reversals: List[float],
    n_band: int = 20,
    y_min: Optional[float] = None,
    y_max: Optional[float] = None,
) -> Dict[str, Any]:
    """Compute From-To (transition) matrix from reversal points.

    Counts transitions between temperature bands across consecutive
    reversal points.  Row = 'from' band, column = 'to' band.

    Args:
        reversals: Peak/valley reversal point values.
        n_band: Number of equally-spaced temperature bands.
        y_min: Lower bound of band range (defaults to min of reversals).
        y_max: Upper bound of band range (defaults to max of reversals).

    Returns:
        Dict with keys 'matrix', 'band_values', 'n_band', 'y_min', 'y_max'.
    """
    if not reversals or len(reversals) < 2:
        return {
            'matrix': np.zeros((n_band, n_band), dtype=int).tolist(),
            'band_values': [],
            'n_band': n_band,
            'y_min': float(y_min or 0),
            'y_max': float(y_max or 0),
        }

    arr = np.asarray(reversals, dtype=float)
    if y_min is None:
        y_min = float(arr.min())
    if y_max is None:
        y_max = float(arr.max())
    if y_max <= y_min:
        y_max = y_min + 1.0

    band_width = (y_max - y_min) / n_band
    band_values = [round(y_min + (i + 0.5) * band_width, 4)
                   for i in range(n_band)]

    def _band_idx(val: float) -> int:
        return max(0, min(n_band - 1, int((val - y_min) / band_width)))

    matrix = np.zeros((n_band, n_band), dtype=int)
    for k in range(len(arr) - 1):
        matrix[_band_idx(arr[k]), _band_idx(arr[k + 1])] += 1

    return {
        'matrix': matrix.tolist(),
        'band_values': band_values,
        'n_band': n_band,
        'y_min': float(y_min),
        'y_max': float(y_max),
    }


def compute_amplitude_histogram(
    cycles: List[Cycle],
    n_bins: int = 20,
    ignore_below: float = 0.0,
) -> Dict[str, Any]:
    """Compute temperature-amplitude (ΔTj) distribution histogram.

    Args:
        cycles: Extracted rainflow cycles.
        n_bins: Number of histogram bins.
        ignore_below: Ignore cycles with range below this value.

    Returns:
        Dict with bin_centers, counts_full, counts_half,
        counts_total, bin_edges.
    """
    filtered = [c for c in cycles if c.range >= ignore_below] \
        if ignore_below > 0 else cycles
    if not filtered:
        return {
            'bin_centers': [],
            'counts_full': [],
            'counts_half': [],
            'counts_total': [],
            'bin_edges': [],
        }

    ranges = np.array([c.range for c in filtered])
    r_min, r_max = float(ranges.min()), float(ranges.max())
    if r_max <= r_min:
        r_max = r_min + 1.0

    bin_edges = np.linspace(r_min, r_max, n_bins + 1)
    bin_centers = ((bin_edges[:-1] + bin_edges[1:]) / 2).tolist()
    bin_width = (r_max - r_min) / n_bins

    counts_full = np.zeros(n_bins)
    counts_half = np.zeros(n_bins)
    for c in filtered:
        idx = int((c.range - r_min) / bin_width)
        idx = max(0, min(n_bins - 1, idx))
        if c.count >= 1.0:
            counts_full[idx] += c.count
        else:
            counts_half[idx] += c.count

    return {
        'bin_centers': bin_centers,
        'counts_full': counts_full.tolist(),
        'counts_half': counts_half.tolist(),
        'counts_total': (counts_full + counts_half).tolist(),
        'bin_edges': bin_edges.tolist(),
    }


def build_cycle_matrix_table(cycles: List[Cycle], decimals: int = 2) -> List[Dict[str, float]]:
    """Aggregate cycles into ΔTj-mean-count matrix rows."""
    agg: Dict[Tuple[float, float], float] = {}

    for c in cycles:
        key = (round(float(c.range), decimals), round(float(c.mean), decimals))
        agg[key] = agg.get(key, 0.0) + float(c.count)

    rows = [
        {
            'delta_tj': k[0],
            'mean_tj': k[1],
            'count': float(v),
        }
        for k, v in agg.items()
    ]
    rows.sort(key=lambda item: (item['delta_tj'], item['mean_tj']))
    return rows


def estimate_damage_from_life_curve(
    cycles: List[Cycle],
    life_curve: List[Dict[str, float]],
    reference_delta_tj: Optional[float] = None
) -> Dict[str, Any]:
    """Estimate Miner damage using a ΔTj-Nf life curve.

    life_curve item format:
        {'delta_tj': float, 'nf': float}
    """
    if not life_curve:
        return {
            'total_damage_per_block': 0.0,
            'blocks_to_failure': None,
            'equivalent_cycles_per_block': None,
            'reference_delta_tj': None,
        }

    delta_vals = np.asarray([float(item['delta_tj']) for item in life_curve], dtype=float)
    nf_vals = np.asarray([float(item['nf']) for item in life_curve], dtype=float)

    valid = (delta_vals > 0) & (nf_vals > 0)
    delta_vals = delta_vals[valid]
    nf_vals = nf_vals[valid]

    if len(delta_vals) < 2:
        raise ValueError("life_curve requires at least 2 valid points (delta_tj>0, nf>0)")

    order = np.argsort(delta_vals)
    delta_vals = delta_vals[order]
    nf_vals = nf_vals[order]

    log_delta = np.log(delta_vals)
    log_nf = np.log(nf_vals)

    def _nf_at(delta_tj: float) -> float:
        x = np.log(max(delta_tj, 1e-12))
        y = np.interp(x, log_delta, log_nf, left=log_nf[0], right=log_nf[-1])
        return float(np.exp(y))

    total_damage = 0.0
    for c in cycles:
        n_i = float(c.count)
        if n_i <= 0 or c.range <= 0:
            continue
        nf_i = _nf_at(float(c.range))
        if nf_i > 0:
            total_damage += n_i / nf_i

    blocks_to_failure = None if total_damage <= 0 else float(1.0 / total_damage)

    ref_delta = float(reference_delta_tj) if reference_delta_tj and reference_delta_tj > 0 else float(np.min(delta_vals))
    nf_ref = _nf_at(ref_delta)

    equivalent_cycles_per_block = float(total_damage * nf_ref) if nf_ref > 0 else None

    return {
        'total_damage_per_block': float(total_damage),
        'blocks_to_failure': blocks_to_failure,
        'equivalent_cycles_per_block': equivalent_cycles_per_block,
        'reference_delta_tj': ref_delta,
        'nf_reference': nf_ref,
    }


def compute_model_based_damage(
    cycles: List[Cycle],
    model_name: str,
    model_params: Dict[str, float],
    safety_factor: float = 1.0,
) -> Dict[str, Any]:
    """Compute Miner cumulative damage index using a registered lifetime model.

    .. math::

        CDI = f_{safe} \\times \\sum_i \\frac{n_i}{N_{f,i}}

    For each rainflow cycle the model-specific ``delta_Tj`` is taken from the
    cycle range, and absolute temperatures ``Tj_max``, ``Tj_min``,
    ``Tj_mean`` are derived from the cycle mean (°C → K).  User-supplied
    *model_params* provide constant parameters (A, alpha, K, β1–β6, Ea,
    t_on, I, V, D, f …).

    Args:
        cycles: Rainflow-counted cycles (range / mean / count).
        model_name: Registered lifetime model name, e.g. ``'coffin-manson'``,
            ``'cips-2008'``.
        model_params: Constant model parameters supplied by user.
        safety_factor: Safety factor *f_safe* (≥ 1, default 1.0).

    Returns:
        Dict with ``total_damage_per_block``, ``blocks_to_failure``,
        ``safety_factor``, ``model_used``, and per-cycle ``cycle_details``.
    """
    from app.core.models.model_factory import ModelFactory

    model = ModelFactory.get_model(model_name)

    total_damage = 0.0
    cycle_details: List[Dict[str, Any]] = []

    for c in cycles:
        n_i = float(c.count)
        if n_i <= 0 or c.range <= 0:
            continue

        # Build params: user-supplied constants + cycle-derived values
        params = dict(model_params)
        params['delta_Tj'] = float(c.range)

        # Derive absolute temperatures in Kelvin from cycle mean (°C)
        # Always override – these are per-cycle quantities, not constants
        tj_mean_k = float(c.mean) + 273.15
        tj_max_k = tj_mean_k + float(c.range) / 2.0
        tj_min_k = tj_mean_k - float(c.range) / 2.0
        params['Tj_max'] = tj_max_k
        params['Tj_min'] = tj_min_k
        params['Tj_mean'] = tj_mean_k

        try:
            nf_i = model.calculate_cycles_to_failure(**params)
            if nf_i <= 0:
                continue
            damage_i = n_i / nf_i
            total_damage += damage_i
            cycle_details.append({
                'delta_tj': float(c.range),
                'mean_tj': float(c.mean),
                'count': n_i,
                'nf': nf_i,
                'damage': damage_i,
            })
        except Exception as exc:
            logger.warning(
                'Model %s skipped cycle (range=%.2f, mean=%.2f): %s',
                model_name, c.range, c.mean, exc,
            )
            continue

    total_damage *= safety_factor

    return {
        'total_damage_per_block': float(total_damage),
        'blocks_to_failure': (1.0 / total_damage) if total_damage > 0 else None,
        'equivalent_cycles_per_block': None,
        'reference_delta_tj': None,
        'nf_reference': None,
        'safety_factor': safety_factor,
        'model_used': model_name,
        'cycle_details': cycle_details,
    }


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

    # Get cumulative counts
    counts = np.array([c.count for c in sorted_cycles])
    cumulative = np.cumsum(counts)

    return cumulative

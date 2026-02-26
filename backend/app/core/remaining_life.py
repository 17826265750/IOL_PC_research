"""
Remaining life assessment based on degradation state.

This module provides tools for estimating remaining useful life (RUL)
based on current damage state, operating conditions, and historical
degradation trends.
"""
from typing import Dict, Optional, List, Tuple, Callable, Any
from dataclasses import dataclass
import numpy as np
import warnings
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit


@dataclass
class RemainingLifeResult:
    """Remaining life assessment result.

    Attributes:
        estimated_cycles_remaining: Estimated cycles until failure
        estimated_time_remaining: Estimated time until failure (hours)
        health_index: Health indicator (0-1, where 1 = perfect health)
        degradation_rate: Rate of degradation (damage per unit time)
        confidence_interval: Optional (lower, upper) bounds for cycles remaining
        method_used: Method used for estimation
    """
    estimated_cycles_remaining: float
    estimated_time_remaining: float
    health_index: float
    degradation_rate: float
    confidence_interval: Optional[Tuple[float, float]] = None
    method_used: str = "linear"

    def __repr__(self) -> str:
        return (f"RemainingLifeResult(cycles={self.estimated_cycles_remaining:.0f}, "
                f"time={self.estimated_time_remaining:.1f}h, "
                f"health={self.health_index:.2%}, "
                f"method={self.method_used})")


@dataclass
class DegradationPoint:
    """A single point in degradation history.

    Attributes:
        cycles: Cumulative cycles at this point
        damage: Damage value (0-1) at this point
        time: Optional elapsed time (e.g., operating hours)
    """
    cycles: float
    damage: float
    time: Optional[float] = None


def assess_remaining_life(
    current_damage: float,
    lifetime_model: Callable[[float, float, Dict[str, Any]], float],
    model_params: Dict[str, Any],
    operating_conditions: Dict[str, float],
    degradation_history: Optional[List[DegradationPoint]] = None,
    cycle_frequency: float = 1.0,
    method: str = 'auto'
) -> RemainingLifeResult:
    """
    Assess remaining life based on current damage state.

    This function uses multiple approaches to estimate remaining life:
    1. **Linear extrapolation**: If history provided, fit line and extrapolate
    2. **Exponential extrapolation**: For materials with accelerating degradation
    3. **Constant rate**: Use current conditions to estimate remaining cycles

    Args:
        current_damage: Current accumulated damage (0-1)
        lifetime_model: Function that calculates cycles to failure
        model_params: Parameters for the lifetime model
        operating_conditions: Current operating conditions dict with:
                             - 'range': Peak-to-peak stress/temperature
                             - 'mean': Mean stress/temperature
        degradation_history: Optional list of historical degradation points
        cycle_frequency: Cycles per hour (default: 1.0)
        method: Estimation method ('auto', 'linear', 'exponential', 'constant')

    Returns:
        RemainingLifeResult with remaining life estimates

    Examples:
        >>> history = [DegradationPoint(1000, 0.1), DegradationPoint(2000, 0.2)]
        >>> result = assess_remaining_life(0.25, model, params,
        ...                                {'range': 100, 'mean': 0},
        ...                                history, cycle_frequency=60)
    """
    # Clamp damage to valid range
    current_damage = max(0.0, min(1.0, current_damage))

    # If already failed
    if current_damage >= 1.0:
        return RemainingLifeResult(
            estimated_cycles_remaining=0.0,
            estimated_time_remaining=0.0,
            health_index=0.0,
            degradation_rate=0.0,
            method_used="failed"
        )

    # Calculate health index
    health_idx = calculate_health_index(current_damage)

    # Determine best method
    if method == 'auto':
        if degradation_history and len(degradation_history) >= 3:
            # Use extrapolation if sufficient history
            method = 'linear'
        else:
            method = 'constant'

    # Estimate remaining cycles based on method
    if method in ('linear', 'exponential') and degradation_history:
        cycles_remaining, degradation_rate, conf_interval = _extrapolate_from_history(
            current_damage,
            degradation_history,
            method=method
        )
    else:
        # Constant rate based on current conditions
        cycle_range = operating_conditions.get('range', 0.0)
        cycle_mean = operating_conditions.get('mean', 0.0)

        try:
            cycles_to_failure = lifetime_model(cycle_range, cycle_mean, model_params)
        except Exception as e:
            warnings.warn(f"Error in lifetime model: {e}")
            cycles_to_failure = float('inf')

        # Assume damage accumulated linearly to current state
        if cycles_to_failure > 0 and cycles_to_failure != float('inf'):
            current_cycles = current_damage * cycles_to_failure
            cycles_remaining = cycles_to_failure - current_cycles
            degradation_rate = 1.0 / cycles_to_failure  # damage per cycle
        else:
            cycles_remaining = float('inf')
            degradation_rate = 0.0

        conf_interval = None

    # Calculate time remaining
    if cycles_remaining == float('inf'):
        time_remaining = float('inf')
    elif cycle_frequency <= 0:
        time_remaining = float('inf')
    else:
        time_remaining = cycles_remaining / cycle_frequency

    return RemainingLifeResult(
        estimated_cycles_remaining=cycles_remaining,
        estimated_time_remaining=time_remaining,
        health_index=health_idx,
        degradation_rate=degradation_rate,
        confidence_interval=conf_interval,
        method_used=method
    )


def _extrapolate_from_history(
    current_damage: float,
    history: List[DegradationPoint],
    method: str = 'linear'
) -> Tuple[float, float, Optional[Tuple[float, float]]]:
    """
    Extrapolate remaining life from degradation history.

    Args:
        current_damage: Current damage level
        history: List of degradation points
        method: 'linear' or 'exponential'

    Returns:
        Tuple of (cycles_remaining, degradation_rate, confidence_interval)
    """
    # Extract cycles and damage values
    cycles = np.array([p.cycles for p in history])
    damages = np.array([p.damage for p in history])

    # Filter out points beyond current damage
    valid_mask = damages <= current_damage
    cycles = cycles[valid_mask]
    damages = damages[valid_mask]

    if len(cycles) < 2:
        # Not enough data for extrapolation
        return (float('inf'), 0.0, None)

    try:
        if method == 'linear':
            # Linear fit: damage = a * cycles + b
            coeffs = np.polyfit(cycles, damages, 1)
            slope = coeffs[0]

            if slope <= 0:
                # Non-positive slope indicates non-degrading behavior
                return (float('inf'), 0.0, None)

            # Extrapolate to damage = 1.0
            cycles_to_failure = (1.0 - coeffs[1]) / slope
            cycles_remaining = cycles_to_failure - cycles[-1]

            # Calculate confidence interval using prediction error
            if len(cycles) >= 3:
                # Standard error of the estimate
                predicted = np.polyval(coeffs, cycles)
                residuals = damages - predicted
                std_error = np.sqrt(np.sum(residuals**2) / (len(cycles) - 2))

                # 95% confidence interval (approximately ±2 SE)
                ci_lower = (1.0 - coeffs[1] - 2*std_error) / slope - cycles[-1]
                ci_upper = (1.0 - coeffs[1] + 2*std_error) / slope - cycles[-1]
                conf_interval = (max(0, ci_lower), max(0, ci_upper))
            else:
                conf_interval = None

            degradation_rate = slope

        elif method == 'exponential':
            # Exponential fit: damage = 1 - exp(-k * cycles)
            # Rearranged: ln(1 - damage) = -k * cycles

            valid_exp = damages < 1.0
            if np.sum(valid_exp) < 2:
                return (float('inf'), 0.0, None)

            exp_cycles = cycles[valid_exp]
            exp_damages = damages[valid_exp]

            # Linear regression on ln(1 - damage) vs cycles
            y_data = np.log(1 - exp_damages)

            coeffs = np.polyfit(exp_cycles, y_data, 1)
            k = -coeffs[0]

            if k <= 0:
                return (float('inf'), 0.0, None)

            # For exponential, cycles to failure is theoretically infinite
            # Use damage = 0.99 as practical failure point
            target_damage = 0.99
            cycles_to_99 = -np.log(1 - target_damage) / k
            cycles_remaining = cycles_to_99 - cycles[-1]

            # Calculate confidence interval
            if len(exp_cycles) >= 3:
                predicted = np.polyval(coeffs, exp_cycles)
                residuals = y_data - predicted
                std_error = np.sqrt(np.sum(residuals**2) / (len(exp_cycles) - 2))

                # Confidence bounds on k
                k_lower = -(-coeffs[0] + 2*std_error)
                k_upper = -(-coeffs[0] - 2*std_error)

                ci_lower = -np.log(1 - target_damage) / k_upper - cycles[-1]
                ci_upper = -np.log(1 - target_damage) / k_lower - cycles[-1]
                conf_interval = (max(0, ci_lower), max(0, ci_upper))
            else:
                conf_interval = None

            # Current degradation rate
            degradation_rate = k * (1 - current_damage)

        else:
            return (float('inf'), 0.0, None)

        if cycles_remaining < 0:
            cycles_remaining = 0.0

        return (cycles_remaining, degradation_rate, conf_interval)

    except Exception as e:
        warnings.warn(f"Error in extrapolation: {e}")
        return (float('inf'), 0.0, None)


def calculate_health_index(current_damage: float) -> float:
    """
    Calculate health index from current damage.

    Health Index is defined as:
    HI = 1 - current_damage

    Where:
    - 1.0 represents pristine condition (no damage)
    - 0.0 represents complete failure
    - Values are clamped to [0, 1]

    Args:
        current_damage: Current accumulated damage (0-1)

    Returns:
        Health index value between 0 and 1

    Examples:
        >>> calculate_health_index(0.0)
        1.0
        >>> calculate_health_index(0.5)
        0.5
        >>> calculate_health_index(1.0)
        0.0
    """
    return max(0.0, min(1.0, 1.0 - current_damage))


def extrapolate_degradation(
    degradation_history: list,
    forecast_cycles: int,
    method: str = 'linear'
) -> np.ndarray:
    """
    Extrapolate degradation trend using linear or exponential fit.

    Args:
        degradation_history: List of DegradationPoint or dict objects
        forecast_cycles: Number of cycles to forecast ahead
        method: Fitting method ('linear' or 'exponential')

    Returns:
        Array of predicted damage values for each forecast cycle

    Examples:
        >>> history = [DegradationPoint(0, 0), DegradationPoint(1000, 0.1)]
        >>> forecast = extrapolate_degradation(history, forecast_cycles=2000)
        >>> forecast[1000]  # Predicted damage at cycle 1000
    """
    if not degradation_history:
        return np.array([0.0] * (forecast_cycles + 1))

    # Parse history
    if isinstance(degradation_history[0], dict):
        cycles = np.array([p['cycles'] for p in degradation_history])
        damages = np.array([p.get('damage', 0) for p in degradation_history])
    else:
        cycles = np.array([p.cycles for p in degradation_history])
        damages = np.array([p.damage for p in degradation_history])

    if len(cycles) < 2:
        # Not enough data - return constant last value
        return np.full(forecast_cycles + 1, damages[-1] if len(damages) > 0 else 0)

    try:
        last_cycle = cycles[-1]

        if method == 'linear':
            coeffs = np.polyfit(cycles, damages, 1)

            forecast_x = np.arange(last_cycle, last_cycle + forecast_cycles + 1)
            forecast = np.polyval(coeffs, forecast_x)

            # Clip to valid range
            forecast = np.clip(forecast, 0, 1)

        elif method == 'exponential':
            # Fit: damage = 1 - exp(-k * cycles)
            valid = damages < 1.0
            if np.sum(valid) < 2:
                return np.full(forecast_cycles + 1, damages[-1])

            valid_cycles = cycles[valid]
            valid_damages = damages[valid]

            y_data = np.log(1 - valid_damages)
            coeffs = np.polyfit(valid_cycles, y_data, 1)
            k = -coeffs[0]

            forecast_x = np.arange(last_cycle, last_cycle + forecast_cycles + 1)
            forecast = 1 - np.exp(-k * forecast_x)

            # Clip to valid range
            forecast = np.clip(forecast, 0, 0.999)

        else:
            raise ValueError(f"Unknown method: {method}")

        return forecast

    except Exception as e:
        warnings.warn(f"Error in degradation extrapolation: {e}")
        return np.full(forecast_cycles + 1, damages[-1])


def calculate_remaining_life_distribution(
    degradation_history: List[DegradationPoint],
    num_samples: int = 1000,
    confidence_levels: List[float] = None
) -> Dict[str, float]:
    """
    Calculate probabilistic remaining life distribution using bootstrap.

    Args:
        degradation_history: Historical degradation points
        num_samples: Number of bootstrap samples
        confidence_levels: List of confidence levels (e.g., [0.05, 0.5, 0.95])

    Returns:
        Dictionary with statistics (mean, std, and requested percentiles)
    """
    if confidence_levels is None:
        confidence_levels = [0.05, 0.25, 0.5, 0.75, 0.95]

    if len(degradation_history) < 3:
        return {
            'mean': float('inf'),
            'std': 0.0,
            **{f'p{int(level*100)}': float('inf') for level in confidence_levels}
        }

    cycles = np.array([p.cycles for p in degradation_history])
    damages = np.array([p.damage for p in degradation_history])

    remaining_lives = []

    # Bootstrap sampling
    for _ in range(num_samples):
        # Resample with replacement
        indices = np.random.choice(len(cycles), len(cycles), replace=True)
        boot_cycles = cycles[indices]
        boot_damages = damages[indices]

        try:
            # Fit linear model
            coeffs = np.polyfit(boot_cycles, boot_damages, 1)

            if coeffs[0] > 0:
                cycles_to_failure = (1.0 - coeffs[1]) / coeffs[0]
                remaining = cycles_to_failure - cycles[-1]
                remaining_lives.append(max(0, remaining))
        except:
            pass

    if not remaining_lives:
        return {
            'mean': float('inf'),
            'std': 0.0,
            **{f'p{int(level*100)}': float('inf') for level in confidence_levels}
        }

    remaining_lives = np.array(remaining_lives)

    result = {
        'mean': float(np.mean(remaining_lives)),
        'std': float(np.std(remaining_lives)),
    }

    for level in confidence_levels:
        percentile = np.percentile(remaining_lives, level * 100)
        result[f'p{int(level*100)}'] = float(percentile)

    return result


def estimate_remaining_life_weibull(
    current_damage: float,
    shape_parameter: float,
    scale_parameter: float
) -> float:
    """
    Estimate remaining life using Weibull distribution.

    The Weibull distribution is commonly used for reliability analysis:
    - Shape parameter (β): Determines failure rate behavior
      β < 1: Decreasing failure rate (infant mortality)
      β = 1: Constant failure rate (random failures)
      β > 1: Increasing failure rate (wear-out)

    Args:
        current_damage: Current damage level (0-1)
        shape_parameter: Weibull shape parameter (β)
        scale_parameter: Weibull scale parameter (η), cycles to 63.2% failure

    Returns:
        Estimated remaining cycles

    Examples:
        >>> remaining = estimate_remaining_life_weibull(0.5, 2.0, 10000)
    """
    if current_damage >= 1.0:
        return 0.0

    if shape_parameter <= 0 or scale_parameter <= 0:
        raise ValueError("Shape and scale parameters must be positive")

    # For Weibull, the reliability function is R(t) = exp(-(t/η)^β)
    # We want to find remaining time such that the conditional reliability
    # matches the remaining damage fraction

    # Using approximation for remaining life
    remaining_fraction = 1.0 - current_damage

    # Inverse Weibull CDF approximation
    if shape_parameter == 1.0:
        # Exponential case
        remaining_cycles = -scale_parameter * np.log(remaining_fraction)
    else:
        # General Weibull case (approximation)
        remaining_cycles = scale_parameter * ((-np.log(remaining_fraction)) ** (1/shape_parameter))

    return remaining_cycles

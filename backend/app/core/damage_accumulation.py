"""
Linear damage accumulation using Miner's rule.

功率模块寿命分析软件 - 线性损伤累积模块
Author: GSH

Total damage = Σ (n_i / N_i) where failure occurs when damage >= 1

This module implements the Palmgren-Miner linear damage hypothesis,
which assumes that damage accumulates linearly with cycle ratio
and that damage from different stress ranges is additive.

References:
    - Miner, M.A. (1945). "Cumulative damage in fatigue"
      Journal of Applied Mechanics, 12(3), A159-A164.
    - ASTM E739-10, "Standard Practice for Statistical Analysis
      of Linear or Linearized Stress-Life (S-N) and Strain-Life
      (e-N) Fatigue Data"
"""
from typing import List, Dict, Optional, Callable, Any
from dataclasses import dataclass, field
import numpy as np
import warnings


@dataclass
class DamageResult:
    """Result of damage accumulation analysis.

    Attributes:
        total_damage: Total accumulated damage (0 to ∞, failure at 1.0)
        remaining_life_fraction: Fraction of life remaining (0 to 1)
        is_critical: True if damage >= 1.0 (predicted failure)
        details: List of individual cycle damage contributions
        confidence_interval: Optional (lower, upper) bounds for damage
    """
    total_damage: float
    remaining_life_fraction: float
    is_critical: bool
    details: List[dict] = field(default_factory=list)
    confidence_interval: Optional[tuple] = None

    def __repr__(self) -> str:
        status = "CRITICAL" if self.is_critical else "OK"
        return (f"DamageResult(total={self.total_damage:.4f}, "
                f"remaining={self.remaining_life_fraction:.2%}, "
                f"status={status})")


@dataclass
class DamageSensitivity:
    """Sensitivity analysis result for damage calculation.

    Attributes:
        parameter_name: Name of the varied parameter
        parameter_values: Array of tested parameter values
        resulting_damage: Array of total damage for each parameter value
        sensitivity_coefficient: Rate of damage change per parameter unit
    """
    parameter_name: str
    parameter_values: np.ndarray
    resulting_damage: np.ndarray
    sensitivity_coefficient: float


def calculate_miner_damage(
    cycles: List[Dict[str, float]],
    lifetime_model: Callable[[float, float, Dict[str, Any]], float],
    model_params: Dict[str, Any],
    include_half_cycles: bool = True
) -> DamageResult:
    """
    Calculate cumulative damage using Miner's linear rule.

    For each cycle condition:
    1. Calculate Nf (cycles to failure) using the lifetime model
    2. Calculate damage contribution: n_i / Nf_i
    3. Sum all damage contributions

    Args:
        cycles: List of cycle dictionaries with keys:
                - 'range': Peak-to-peak stress/temperature range
                - 'mean': Mean stress/temperature value
                - 'count': Number of cycles at this condition
        lifetime_model: Function that calculates cycles to failure.
                       Signature: (range: float, mean: float, params: dict) -> float
        model_params: Parameters for the lifetime model (e.g., material constants)
        include_half_cycles: Whether to include half-cycle counts (default: True)

    Returns:
        DamageResult with total damage and details

    Examples:
        >>> def my_s_n_model(rnge, mn, params):
        ...     # S-N curve: N = (S_f / S)^b
        ...     return (params['fatigue_limit'] / rnge) ** params['exponent']
        >>>
        >>> cycles = [{'range': 100, 'mean': 0, 'count': 1000},
        ...           {'range': 150, 'mean': 50, 'count': 500}]
        >>> params = {'fatigue_limit': 500, 'exponent': 3}
        >>> result = calculate_miner_damage(cycles, my_s_n_model, params)
    """
    if not cycles:
        return DamageResult(
            total_damage=0.0,
            remaining_life_fraction=1.0,
            is_critical=False,
            details=[]
        )

    total_damage = 0.0
    details = []

    for i, cycle in enumerate(cycles):
        cycle_range = cycle.get('range', 0.0)
        cycle_mean = cycle.get('mean', 0.0)
        cycle_count = cycle.get('count', 0.0)

        # Skip if no cycles or invalid range
        if cycle_count <= 0 or cycle_range <= 0:
            continue

        # Optionally exclude half cycles
        if not include_half_cycles and cycle_count < 1.0:
            continue

        # Calculate cycles to failure using the lifetime model
        try:
            cycles_to_failure = lifetime_model(cycle_range, cycle_mean, model_params)
        except Exception as e:
            warnings.warn(f"Error calculating Nf for cycle {i}: {e}")
            cycles_to_failure = float('inf')

        # Calculate damage contribution
        if cycles_to_failure <= 0:
            # Model returned invalid value - assume infinite life
            damage_contribution = 0.0
        else:
            damage_contribution = cycle_count / cycles_to_failure

        total_damage += damage_contribution

        details.append({
            'cycle_index': i,
            'range': cycle_range,
            'mean': cycle_mean,
            'count': cycle_count,
            'cycles_to_failure': cycles_to_failure,
            'damage_contribution': damage_contribution,
            'damage_fraction': 0.0 if total_damage == 0 else damage_contribution / total_damage
        })

    remaining_life = max(0.0, 1.0 - total_damage)
    is_critical = total_damage >= 1.0

    return DamageResult(
        total_damage=total_damage,
        remaining_life_fraction=remaining_life,
        is_critical=is_critical,
        details=details
    )


def estimate_remaining_cycles(
    current_damage: float,
    lifetime_model: Callable[[float, float, Dict[str, Any]], float],
    model_params: Dict[str, Any],
    cycle_condition: Dict[str, float]
) -> float:
    """
    Estimate remaining cycles based on current damage state.

    Calculates how many more cycles at the given condition can be
    sustained before reaching failure (damage = 1.0).

    Args:
        current_damage: Current accumulated damage (0-1)
        lifetime_model: Function that calculates cycles to failure
        model_params: Parameters for the lifetime model
        cycle_condition: Current operating conditions with keys:
                        - 'range': Peak-to-peak stress/temperature range
                        - 'mean': Mean stress/temperature value

    Returns:
        Estimated remaining cycles until failure

    Examples:
        >>> current_d = 0.5  # 50% damage already accumulated
        >>> condition = {'range': 100, 'mean': 0}
        >>> remaining = estimate_remaining_cycles(current_d, model, params, condition)
    """
    if current_damage >= 1.0:
        return 0.0

    if current_damage < 0:
        warnings.warn("Negative damage value detected, treating as 0")
        current_damage = 0.0

    cycle_range = cycle_condition.get('range', 0.0)
    cycle_mean = cycle_condition.get('mean', 0.0)

    if cycle_range <= 0:
        return float('inf')

    # Calculate cycles to failure at this condition
    try:
        cycles_to_failure = lifetime_model(cycle_range, cycle_mean, model_params)
    except Exception as e:
        warnings.warn(f"Error calculating Nf: {e}")
        return 0.0

    if cycles_to_failure <= 0:
        # Model suggests infinite life
        return float('inf')

    # Remaining damage capacity
    remaining_damage = 1.0 - current_damage

    # Remaining cycles = remaining damage * Nf
    remaining_cycles = remaining_damage * cycles_to_failure

    return remaining_cycles


def calculate_damage_rate(
    cycles: List[Dict[str, float]],
    time_period: float,
    lifetime_model: Callable[[float, float, Dict[str, Any]], float],
    model_params: Dict[str, Any]
) -> float:
    """
    Calculate damage accumulation rate per unit time.

    Useful for tracking how quickly damage accumulates during operation.

    Args:
        cycles: List of cycle dictionaries
        time_period: Time period over which cycles occurred (e.g., hours)
        lifetime_model: Function that calculates cycles to failure
        model_params: Parameters for the lifetime model

    Returns:
        Damage rate (damage units per time unit)

    Examples:
        >>> cycles = [{'range': 100, 'mean': 0, 'count': 1000}]
        >>> rate = calculate_damage_rate(cycles, time_period=24, model=model, params=params)
        >>> print(f"Damage rate: {rate:.4f} per hour")
    """
    if time_period <= 0:
        raise ValueError("Time period must be positive")

    damage_result = calculate_miner_damage(cycles, lifetime_model, model_params)

    return damage_result.total_damage / time_period


def perform_sensitivity_analysis(
    cycles: List[Dict[str, float]],
    lifetime_model: Callable[[float, float, Dict[str, Any]], float],
    base_params: Dict[str, Any],
    param_name: str,
    param_range: np.ndarray,
    relative_variation: bool = True
) -> DamageSensitivity:
    """
    Perform sensitivity analysis on a model parameter.

    Calculates how damage changes as a specific parameter is varied
    across a range of values.

    Args:
        cycles: List of cycle dictionaries
        lifetime_model: Function that calculates cycles to failure
        base_params: Base parameters for the lifetime model
        param_name: Name of parameter to vary
        param_range: Array of parameter values to test
        relative_variation: If True, param_range is relative multiplier
                           from 0.5 to 1.5 of base value

    Returns:
        DamageSensitivity with parameter values and resulting damage
    """
    damages = []

    for value in param_range:
        test_params = base_params.copy()

        if relative_variation and param_name in base_params:
            test_params[param_name] = base_params[param_name] * value
        else:
            test_params[param_name] = value

        result = calculate_miner_damage(cycles, lifetime_model, test_params)
        damages.append(result.total_damage)

    damages = np.array(damages)

    # Calculate sensitivity coefficient (slope)
    if len(param_range) > 1 and len(damages) > 1:
        # Linear regression slope
        valid_mask = np.isfinite(damages)
        if np.sum(valid_mask) > 1:
            slope = np.polyfit(param_range[valid_mask], damages[valid_mask], 1)[0]
        else:
            slope = 0.0
    else:
        slope = 0.0

    return DamageSensitivity(
        parameter_name=param_name,
        parameter_values=param_range,
        resulting_damage=damages,
        sensitivity_coefficient=slope
    )


def calculate_confidence_interval(
    cycles: List[Dict[str, float]],
    lifetime_model: Callable[[float, float, Dict[str, Any]], float],
    model_params: Dict[str, Any],
    scatter_factor: float = 1.0,
    confidence_level: float = 0.95
) -> tuple:
    """
    Calculate confidence interval for damage estimate.

    Uses a scatter factor to account for variability in the S-N data.
    The scatter factor is typically derived from statistical analysis
    of test data (e.g., 2x standard deviation).

    Args:
        cycles: List of cycle dictionaries
        lifetime_model: Function that calculates cycles to failure
        model_params: Parameters for the lifetime model
        scatter_factor: Scatter factor for Nf (e.g., 1.2 for 20% variability)
        confidence_level: Confidence level (default: 0.95 for 95%)

    Returns:
        Tuple of (lower_bound, upper_bound) damage estimates
    """
    base_result = calculate_miner_damage(cycles, lifetime_model, model_params)
    base_damage = base_result.total_damage

    # For 95% confidence, use ±2 standard deviations
    # Scatter factor typically represents ±1 SD
    if confidence_level >= 0.95:
        n_std = 2.0
    elif confidence_level >= 0.68:
        n_std = 1.0
    else:
        n_std = 1.96  # Default to 95%

    # Adjust damage based on scatter
    # Lower bound: more conservative (less cycles to failure)
    # Upper bound: less conservative (more cycles to failure)

    # Damage is inversely proportional to Nf
    # If Nf varies by factor f, damage varies by factor 1/f

    n_std_scatter = 1.0 + n_std * (scatter_factor - 1.0)

    lower_bound = base_damage * n_std_scatter
    upper_bound = base_damage / n_std_scatter

    return (max(0.0, lower_bound), upper_bound)


def combine_damage_states(
    damage_states: List[float],
    weights: Optional[List[float]] = None
) -> float:
    """
    Combine multiple damage states (e.g., from different components).

    Uses either simple averaging or weighted averaging based on
    component importance.

    Args:
        damage_states: List of damage values (0-1 for each component)
        weights: Optional weights for each component

    Returns:
        Combined damage value

    Examples:
        >>> d = [0.5, 0.3, 0.7]  # Three components
        >>> combined = combine_damage_states(d)  # Simple average
        >>> weighted = combine_damage_states(d, weights=[0.5, 0.3, 0.2])
    """
    if not damage_states:
        return 0.0

    damages = np.array(damage_states)

    if weights is None:
        return float(np.mean(damages))

    weights = np.array(weights)
    if len(weights) != len(damages):
        raise ValueError("Weights must have same length as damage states")

    if np.sum(weights) == 0:
        return float(np.mean(damages))

    return float(np.sum(damages * weights) / np.sum(weights))


def predict_time_to_failure(
    current_damage: float,
    damage_rate: float,
    time_unit: str = 'hours'
) -> float:
    """
    Predict time to failure based on current damage and damage rate.

    Args:
        current_damage: Current accumulated damage (0-1)
        damage_rate: Damage accumulation rate per time unit
        time_unit: Time unit label for reporting

    Returns:
        Estimated time to failure in the same units as damage_rate

    Examples:
        >>> current_d = 0.5
        >>> rate = 0.01  # 1% damage per hour
        >>> ttf = predict_time_to_failure(current_d, rate)
        >>> print(f"Time to failure: {ttf:.1f} hours")
    """
    if current_damage >= 1.0:
        return 0.0

    if damage_rate <= 0:
        return float('inf')

    remaining_damage = 1.0 - current_damage
    time_to_failure = remaining_damage / damage_rate

    return time_to_failure


def adjust_for_sequence_effect(
    cycles: List[Dict[str, float]],
    lifetime_model: Callable[[float, float, Dict[str, Any]], float],
    model_params: Dict[str, Any],
    sequence_factor: float = 1.0
) -> DamageResult:
    """
    Adjust damage calculation for load sequence effects.

    Miner's rule assumes sequence independence, but in reality,
    high-low load sequences can cause different damage than
    low-high sequences. This applies an empirical correction.

    Args:
        cycles: List of cycle dictionaries (should be sorted by application order)
        lifetime_model: Function that calculates cycles to failure
        model_params: Parameters for the lifetime model
        sequence_factor: Empirical correction factor (1.0 = no correction)

    Returns:
        DamageResult with sequence-adjusted damage

    Note:
        This is an empirical correction. The sequence_factor should
        be determined from experimental data for the specific material
        and loading conditions.
    """
    base_result = calculate_miner_damage(cycles, lifetime_model, model_params)

    if sequence_factor == 1.0:
        return base_result

    # Apply sequence correction to total damage
    adjusted_damage = base_result.total_damage * sequence_factor
    adjusted_remaining = max(0.0, 1.0 - adjusted_damage)

    # Adjust individual damage contributions proportionally
    adjusted_details = []
    for detail in base_result.details:
        adjusted_detail = detail.copy()
        adjusted_detail['damage_contribution'] *= sequence_factor
        if adjusted_damage > 0:
            adjusted_detail['damage_fraction'] = (
                adjusted_detail['damage_contribution'] / adjusted_damage
            )
        adjusted_details.append(adjusted_detail)

    return DamageResult(
        total_damage=adjusted_damage,
        remaining_life_fraction=adjusted_remaining,
        is_critical=adjusted_damage >= 1.0,
        details=adjusted_details
    )

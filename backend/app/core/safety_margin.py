"""
Safety margin calculation for lifetime prediction.

This module provides tools for calculating and interpreting safety margins
in lifetime prediction, including safety factors, design margins, and
acceptability criteria.
"""
from dataclasses import dataclass
from typing import Optional, List, Tuple
import numpy as np


@dataclass
class SafetyMarginResult:
    """Safety margin calculation result.

    Attributes:
        safety_factor: Applied safety factor
        design_life_cycles: Required design life in cycles
        predicted_life_cycles: Predicted life in cycles
        margin_percentage: Safety margin as percentage
        is_acceptable: Whether the margin meets requirements
        margin_value: Absolute margin (predicted - design)
        utilization: Ratio of used life to available life
    """
    safety_factor: float
    design_life_cycles: float
    predicted_life_cycles: float
    margin_percentage: float
    is_acceptable: bool
    margin_value: float
    utilization: float

    def __repr__(self) -> str:
        status = "ACCEPTABLE" if self.is_acceptable else "UNACCEPTABLE"
        return (f"SafetyMarginResult(design={self.design_life_cycles:.0f}, "
                f"predicted={self.predicted_life_cycles:.0f}, "
                f"margin={self.margin_percentage:.1f}%, "
                f"status={status})")


@dataclass
class SafetyMarginDistribution:
    """Safety margin result with statistical distribution.

    Attributes:
        mean_margin: Mean safety margin
        std_margin: Standard deviation of margin
        percentile_5: 5th percentile (conservative)
        percentile_95: 95th percentile (optimistic)
        probability_acceptable: Probability that margin is acceptable
    """
    mean_margin: float
    std_margin: float
    percentile_5: float
    percentile_95: float
    probability_acceptable: float


def calculate_safety_margin(
    design_life: float,
    predicted_life: float,
    safety_factor: float = 1.0,
    minimum_acceptable_margin: float = 0.0
) -> SafetyMarginResult:
    """
    Calculate safety margin between design and predicted life.

    The safety margin is calculated as:
    Margin = (Predicted Life / (Design Life × Safety Factor)) - 1

    A margin >= 0 (or margin >= minimum_acceptable_margin) indicates
    the design is acceptable.

    Args:
        design_life: Required design life in cycles
        predicted_life: Predicted life in cycles
        safety_factor: Safety factor to apply (default: 1.0)
        minimum_acceptable_margin: Minimum acceptable margin (default: 0.0)

    Returns:
        SafetyMarginResult with calculated margins and acceptability

    Examples:
        >>> result = calculate_safety_margin(design_life=10000,
        ...                                  predicted_life=15000,
        ...                                  safety_factor=1.2)
        >>> print(f"Margin: {result.margin_percentage:.1f}%")
        >>> print(f"Acceptable: {result.is_acceptable}")
    """
    if design_life <= 0:
        raise ValueError("Design life must be positive")

    if predicted_life <= 0:
        raise ValueError("Predicted life must be positive")

    if safety_factor <= 0:
        raise ValueError("Safety factor must be positive")

    # Adjusted design life accounting for safety factor
    adjusted_design_life = design_life * safety_factor

    # Calculate margin value (absolute)
    margin_value = predicted_life - adjusted_design_life

    # Calculate margin percentage
    margin_percentage = (predicted_life / adjusted_design_life - 1) * 100

    # Calculate utilization (inverse of margin)
    if predicted_life > 0:
        utilization = adjusted_design_life / predicted_life
    else:
        utilization = float('inf')

    # Check acceptability
    is_acceptable = margin_percentage >= minimum_acceptable_margin

    return SafetyMarginResult(
        safety_factor=safety_factor,
        design_life_cycles=design_life,
        predicted_life_cycles=predicted_life,
        margin_percentage=margin_percentage,
        is_acceptable=is_acceptable,
        margin_value=margin_value,
        utilization=utilization
    )


def calculate_required_safety_factor(
    design_life: float,
    predicted_life: float,
    target_margin: float = 0.0
) -> float:
    """
    Calculate required safety factor to achieve target margin.

    Args:
        design_life: Required design life in cycles
        predicted_life: Predicted life in cycles
        target_margin: Target margin percentage (default: 0.0)

    Returns:
        Required safety factor

    Examples:
        >>> sf = calculate_required_safety_factor(10000, 12000, target_margin=20)
        >>> print(f"Required safety factor: {sf:.2f}")
    """
    if design_life <= 0 or predicted_life <= 0:
        raise ValueError("Life values must be positive")

    if target_margin <= -100:
        raise ValueError("Target margin must be greater than -100%")

    # Rearranging margin formula:
    # margin% = (predicted / (design * SF) - 1) * 100
    # SF = predicted / (design * (1 + margin%/100))

    required_sf = predicted_life / (design_life * (1 + target_margin / 100))

    return required_sf


def calculate_statistical_safety_margin(
    design_life: float,
    predicted_life_mean: float,
    predicted_life_std: float,
    safety_factor: float = 1.0,
    minimum_acceptable_margin: float = 0.0
) -> SafetyMarginDistribution:
    """
    Calculate safety margin with statistical uncertainty.

    Accounts for uncertainty in the predicted life by treating it as
    a random variable with specified mean and standard deviation.

    Args:
        design_life: Required design life in cycles
        predicted_life_mean: Mean of predicted life distribution
        predicted_life_std: Standard deviation of predicted life
        safety_factor: Safety factor to apply
        minimum_acceptable_margin: Minimum acceptable margin

    Returns:
        SafetyMarginDistribution with statistical margins

    Examples:
        >>> result = calculate_statistical_safety_margin(
        ...     design_life=10000, predicted_life_mean=12000,
        ...     predicted_life_std=2000, safety_factor=1.0
        ... )
        >>> print(f"Mean margin: {result.mean_margin:.1f}%")
        >>> print(f"P5 margin: {result.percentile_5:.1f}%")
    """
    if design_life <= 0:
        raise ValueError("Design life must be positive")

    if predicted_life_mean <= 0 or predicted_life_std < 0:
        raise ValueError("Predicted life parameters invalid")

    adjusted_design = design_life * safety_factor

    # Mean margin
    mean_margin = (predicted_life_mean / adjusted_design - 1) * 100

    # Std of margin (assuming normal distribution and linear transformation)
    margin_std = (predicted_life_std / adjusted_design) * 100

    # Calculate percentiles (assuming normal distribution)
    from scipy import stats

    percentile_5 = mean_margin + stats.norm.ppf(0.05) * margin_std
    percentile_95 = mean_margin + stats.norm.ppf(0.95) * margin_std

    # Probability of being acceptable
    if margin_std > 0:
        z_score = (minimum_acceptable_margin - mean_margin) / margin_std
        probability_acceptable = 1 - stats.norm.cdf(z_score)
    else:
        probability_acceptable = 1.0 if mean_margin >= minimum_acceptable_margin else 0.0

    return SafetyMarginDistribution(
        mean_margin=mean_margin,
        std_margin=margin_std,
        percentile_5=percentile_5,
        percentile_95=percentile_95,
        probability_acceptable=probability_acceptable
    )


def calculate_partial_safety_factors(
    uncertainty_contributions: dict,
    target_reliability_index: float = 3.0
) -> dict:
    """
    Calculate partial safety factors using First Order Reliability Method (FORM).

    This implements a simplified FORM approach where partial safety factors
    are calculated for different sources of uncertainty (load, material,
    geometry, etc.).

    Args:
        uncertainty_contributions: Dictionary of uncertainty sources with:
                                   - 'coefficient_of_variation': COV for each source
                                   - 'sensitivity': Sensitivity factor (0-1)
        target_reliability_index: Target reliability index β (default: 3.0 for Pf ≈ 0.001)

    Returns:
        Dictionary of partial safety factors for each source

    Examples:
        >>> contributions = {
        ...     'load': {'cov': 0.1, 'sensitivity': 0.7},
        ...     'material': {'cov': 0.15, 'sensitivity': 0.5}
        ... }
        >>> psf = calculate_partial_safety_factors(contributions, target_reliability_index=3.0)
    """
    partial_factors = {}

    for source, params in uncertainty_contributions.items():
        cov = params.get('coefficient_of_variation', 0.0)
        sensitivity = params.get('sensitivity', 1.0)

        if cov < 0:
            raise ValueError(f"COV must be non-negative for {source}")

        if sensitivity < 0 or sensitivity > 1:
            raise ValueError(f"Sensitivity must be in [0, 1] for {source}")

        # Partial safety factor using FORM approximation
        # γ = 1 + α × β × V
        # where α is the sensitivity factor, β is reliability index, V is COV

        if source.endswith('_load') or source.startswith('load_'):
            # For loads: factor > 1 (conservative: increase design load)
            factor = 1 + sensitivity * target_reliability_index * cov
        else:
            # For resistance/material: factor < 1 (conservative: reduce strength)
            factor = 1 / (1 + sensitivity * target_reliability_index * cov)

        partial_factors[source] = factor

    return partial_factors


def combine_safety_factors(
    partial_factors: dict,
    factor_type: str = 'product'
) -> float:
    """
    Combine multiple partial safety factors into a single factor.

    Args:
        partial_factors: Dictionary of partial safety factors
        factor_type: Combination method ('product', 'root_sum_square', 'max')

    Returns:
        Combined safety factor

    Examples:
        >>> factors = {'load': 1.2, 'material': 1.1, 'geometry': 1.05}
        >>> combined = combine_safety_factors(factors, factor_type='product')
    """
    if not partial_factors:
        return 1.0

    values = list(partial_factors.values())

    if factor_type == 'product':
        # Multiplicative combination (most conservative)
        combined = np.prod(values)
    elif factor_type == 'root_sum_square':
        # SRSS combination
        combined = np.sqrt(1 + sum((v - 1)**2 for v in values))
    elif factor_type == 'max':
        # Maximum of all factors
        combined = max(values)
    else:
        raise ValueError(f"Unknown factor_type: {factor_type}")

    return combined


def assess_design_adequacy(
    safety_margin_result: SafetyMarginResult,
    inspection_interval: Optional[float] = None
) -> dict:
    """
    Assess overall design adequacy based on safety margin.

    Provides additional context and recommendations based on the
    calculated safety margin.

    Args:
        safety_margin_result: Result from calculate_safety_margin
        inspection_interval: Optional inspection interval in cycles

    Returns:
        Dictionary with assessment results and recommendations

    Examples:
        >>> result = calculate_safety_margin(10000, 15000, 1.2)
        >>> assessment = assess_design_adequacy(result, inspection_interval=1000)
    """
    margin = safety_margin_result.margin_percentage
    utilization = safety_margin_result.utilization

    # Determine adequacy level
    if margin < 0:
        level = "INADEQUATE"
        color = "red"
        recommendation = "Design does not meet requirements. Redesign required."
    elif margin < 10:
        level = "MARGINAL"
        color = "yellow"
        recommendation = "Design barely meets requirements. Consider enhancement."
    elif margin < 30:
        level = "ADEQUATE"
        color = "green"
        recommendation = "Design meets requirements with reasonable margin."
    elif margin < 100:
        level = "GOOD"
        color = "green"
        recommendation = "Design has good safety margin."
    else:
        level = "EXCELLENT"
        color = "green"
        recommendation = "Design has excellent safety margin."

    # Inspection recommendation
    inspection_recommendation = None
    if inspection_interval is not None:
        remaining_life_ratio = safety_margin_result.predicted_life_cycles / safety_margin_result.design_life_cycles
        if remaining_life_ratio > 2:
            inspection_freq = "Low frequency inspection acceptable"
        elif remaining_life_ratio > 1.5:
            inspection_freq = "Standard inspection frequency"
        elif remaining_life_ratio > 1.2:
            inspection_freq = "Increased inspection frequency recommended"
        else:
            inspection_freq = "Frequent inspection and monitoring required"

        inspection_recommendation = inspection_freq

    return {
        'adequacy_level': level,
        'color_code': color,
        'recommendation': recommendation,
        'inspection_recommendation': inspection_recommendation,
        'margin_category': _categorize_margin(margin),
        'utilization_percentage': utilization * 100
    }


def _categorize_margin(margin: float) -> str:
    """Categorize margin into standard categories."""
    if margin < 0:
        return "critical"
    elif margin < 10:
        return "low"
    elif margin < 30:
        return "moderate"
    elif margin < 100:
        return "high"
    else:
        return "very_high"


def calculate_reserve_factor(
    allowable_value: float,
    applied_value: float
) -> float:
    """
    Calculate reserve factor (allowable / applied).

    The reserve factor is commonly used in aerospace and is
    simply the ratio of allowable to applied value.

    Args:
        allowable_value: Allowable (limit) value
        applied_value: Applied (actual) value

    Returns:
        Reserve factor (RF >= 1 is acceptable)

    Examples:
        >>> rf = calculate_reserve_factor(allowable_value=150, applied_value=100)
        >>> print(f"Reserve Factor: {rf:.2f}")
    """
    if applied_value <= 0:
        raise ValueError("Applied value must be positive")

    return allowable_value / applied_value


def calculate_damage_safety_margin(
    current_damage: float,
    critical_damage: float = 1.0
) -> float:
    """
    Calculate safety margin in terms of damage.

    Args:
        current_damage: Current damage level (0-1)
        critical_damage: Critical damage level (default: 1.0)

    Returns:
        Safety margin as fraction (0 = at limit, >0 = safe)

    Examples:
        >>> margin = calculate_damage_safety_margin(current_damage=0.5)
        >>> print(f"Damage margin: {margin:.2%}")
    """
    if critical_damage <= 0:
        raise ValueError("Critical damage must be positive")

    if current_damage < 0:
        current_damage = 0

    margin = 1 - (current_damage / critical_damage)
    return max(0, margin)

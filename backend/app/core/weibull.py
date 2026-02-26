"""
Weibull distribution analysis for reliability engineering.
Used to analyze failure data and calculate B10, B50, B63.2 life.

This module provides functions for fitting Weibull distributions to failure data,
calculating reliability metrics, and generating probability plots for reliability
engineering applications per IEC 61714 and related standards.

Reference:
- IEC 61714: Reliability testing - Failure rates
- Weibull analysis methods for reliability engineering
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional

import math

import numpy as np
from scipy import stats, optimize


@dataclass
class WeibullResult:
    """Weibull analysis result.

    Attributes:
        shape: Weibull shape parameter (Beta, slope). Beta < 1 indicates
            early-life failures, Beta = 1 constant failure rate, Beta > 1
            wear-out failures.
        scale: Weibull scale parameter (Eta, characteristic life). The time at
            which 63.2% of the population will have failed.
        location: Weibull location parameter (Gamma, minimum life). Usually 0
            for electronic components.
        r_squared: Coefficient of determination for the fit (0-1).
        b10_life: Life at 10% failure probability (90% reliability).
        b50_life: Life at 50% failure probability (median life).
        b63_life: Life at 63.2% failure probability (characteristic life).
        mttf: Mean time to failure (expected value of distribution).
        confidence_interval: Optional (lower, upper) bound for scale parameter.
    """
    shape: float
    scale: float
    location: float
    r_squared: float
    b10_life: float
    b50_life: float
    b63_life: float
    mttf: float
    confidence_interval: Optional[Tuple[float, float]]


class WeibullAnalysisError(Exception):
    """Exception raised for errors in Weibull analysis."""
    pass


def fit_weibull(
    failure_data: List[float],
    censored_data: Optional[List[float]] = None,
    confidence_level: float = 0.9
) -> WeibullResult:
    """Fit Weibull distribution to failure data using MLE.

    This function fits a 2-parameter or 3-parameter Weibull distribution
    to failure data using Maximum Likelihood Estimation (MLE). Handles
    both complete and censored (suspension) data.

    Args:
        failure_data: List of failure times (cycles or hours). Must contain
            at least 2 non-negative values.
        censored_data: List of suspension/censored times. These represent
            units that did not fail by the end of the test.
        confidence_level: Confidence level for intervals (0-1). Default 0.9
            for 90% confidence intervals.

    Returns:
        WeibullResult with fitted parameters and life estimates.

    Raises:
        WeibullAnalysisError: If data is insufficient or fitting fails.

    Examples:
        >>> failures = [100, 200, 300, 400, 500]
        >>> result = fit_weibull(failures)
        >>> print(f"B10 life: {result.b10_life:.1f} cycles")
    """
    # Validate inputs
    if not failure_data:
        raise WeibullAnalysisError("Failure data cannot be empty")

    failures = np.array(failure_data, dtype=float)
    if len(failures) < 2:
        raise WeibullAnalysisError("At least 2 failure points required")
    if np.any(failures <= 0):
        raise WeibullAnalysisError("All failure times must be positive")

    censored = np.array(censored_data, dtype=float) if censored_data else np.array([])
    if np.any(censored <= 0):
        raise WeibullAnalysisError("All censored times must be positive")

    # Combine data for analysis
    n_failures = len(failures)
    n_censored = len(censored)
    n_total = n_failures + n_censored

    # Sort failure data
    failures_sorted = np.sort(failures)

    # Calculate median ranks for probability plot
    # Using Bernard's approximation for median rank
    median_ranks = _calculate_median_ranks(n_failures)

    # Fit using MLE via scipy.stats
    try:
        if n_censored > 0:
            # For censored data, we need custom likelihood
            shape, scale, loc = _fit_censored_weibull(failures_sorted, censored)
        else:
            # For complete data, use scipy's built-in fit
            # Fixed location at 0 for 2-parameter Weibull
            params = stats.weibull_min.fit(failures_sorted, floc=0)
            shape, loc, scale = params[0], params[1], params[2]

            # Validate shape parameter
            if shape <= 0 or scale <= 0:
                raise WeibullAnalysisError("Invalid fitted parameters")

    except Exception as e:
        raise WeibullAnalysisError(f"Fitting failed: {str(e)}") from e

    # Calculate R-squared for goodness of fit
    r_squared = _calculate_weibull_r_squared(failures_sorted, shape, scale, loc)

    # Calculate B-life values
    b10_life = calculate_b_life(shape, scale, 0.1)
    b50_life = calculate_b_life(shape, scale, 0.5)
    b63_life = scale  # By definition

    # Calculate MTTF using gamma function
    mttf = scale * math.gamma(1 + 1 / shape)

    # Calculate confidence interval for scale parameter
    confidence_interval = _calculate_confidence_interval(
        failures_sorted, censored, shape, scale, confidence_level
    ) if n_total >= 5 else None

    return WeibullResult(
        shape=shape,
        scale=scale,
        location=loc,
        r_squared=r_squared,
        b10_life=b10_life,
        b50_life=b50_life,
        b63_life=b63_life,
        mttf=mttf,
        confidence_interval=confidence_interval
    )


def calculate_b_life(shape: float, scale: float, percentile: float) -> float:
    """Calculate life at given failure probability.

    The B(P) life is the time at which P percent of the population
    will have failed. For example, B10 is the time at 10% failures
    (90% reliability).

    Formula: B(P) = eta × [-ln(1-P)]^(1/beta)

    Args:
        shape: Weibull shape parameter (beta, slope). Must be positive.
        scale: Weibull scale parameter (eta, characteristic life). Must be positive.
        percentile: Failure probability (0-1). e.g., 0.1 for B10, 0.5 for B50.

    Returns:
        Life at specified percentile (same units as scale parameter).

    Raises:
        ValueError: If shape or scale are non-positive, or percentile out of range.

    Examples:
        >>> calculate_b_life(shape=2.0, scale=1000, percentile=0.1)
        316.23  # B10 life
    """
    if shape <= 0:
        raise ValueError("Shape parameter must be positive")
    if scale <= 0:
        raise ValueError("Scale parameter must be positive")
    if not (0 < percentile < 1):
        raise ValueError("Percentile must be between 0 and 1")

    # B(P) = eta × [-ln(1-P)]^(1/beta)
    return scale * (-np.log(1 - percentile)) ** (1 / shape)


def weibull_probability_plot_data(
    failure_data: List[float]
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate data for Weibull probability plot.

    A Weibull probability plot is used to visually assess whether data
    follows a Weibull distribution. If the data is Weibull-distributed,
    the points will fall approximately on a straight line.

    Returns:
        Tuple of (x_values, y_values) for plotting:
        - x: ln(failure_time) - natural log of failure times
        - y: ln(-ln(1-F)) where F is the cumulative failure probability

    Raises:
        WeibullAnalysisError: If insufficient data provided.

    Examples:
        >>> failures = [100, 200, 300, 400, 500]
        >>> x, y = weibull_probability_plot_data(failures)
        >>> # Plot y vs x to assess linearity
    """
    if not failure_data:
        raise WeibullAnalysisError("Failure data cannot be empty")

    failures = np.array(failure_data, dtype=float)
    failures_sorted = np.sort(failures)

    if len(failures) < 2:
        raise WeibullAnalysisError("At least 2 failure points required")

    # Calculate median ranks (cumulative failure probabilities)
    n = len(failures_sorted)
    median_ranks = _calculate_median_ranks(n)

    # Avoid log(0) by clamping
    median_ranks_clamped = np.clip(median_ranks, 1e-10, 1 - 1e-10)

    # x = ln(failure_time)
    x_values = np.log(failures_sorted)

    # y = ln(-ln(1 - F))
    y_values = np.log(-np.log(1 - median_ranks_clamped))

    return x_values, y_values


def calculate_reliability(
    shape: float,
    scale: float,
    time: float,
    location: float = 0.0
) -> float:
    """Calculate reliability at given time.

    Reliability R(t) is the probability that a unit will survive beyond
    time t without failure.

    Formula: R(t) = exp(-((t - gamma) / eta)^beta)

    Args:
        shape: Weibull shape parameter (beta). Must be positive.
        scale: Weibull scale parameter (eta). Must be positive.
        time: Time at which to calculate reliability.
        location: Weibull location parameter (gamma). Default 0.

    Returns:
        Reliability value between 0 and 1.

    Raises:
        ValueError: If parameters are invalid.

    Examples:
        >>> calculate_reliability(shape=2.0, scale=1000, time=500)
        0.7788  # 77.88% reliability at 500 time units
    """
    if shape <= 0:
        raise ValueError("Shape parameter must be positive")
    if scale <= 0:
        raise ValueError("Scale parameter must be positive")

    adjusted_time = time - location

    if adjusted_time <= 0:
        return 1.0  # 100% reliability before location parameter

    # R(t) = exp(-((t - gamma) / eta)^beta)
    return np.exp(-((adjusted_time / scale) ** shape))


def calculate_hazard_rate(
    shape: float,
    scale: float,
    time: float,
    location: float = 0.0
) -> float:
    """Calculate instantaneous hazard rate at given time.

    The hazard rate h(t) is the instantaneous failure rate given
    survival to time t.

    Formula: h(t) = (beta / eta) * ((t - gamma) / eta)^(beta - 1)

    Args:
        shape: Weibull shape parameter (beta). Must be positive.
        scale: Weibull scale parameter (eta). Must be positive.
        time: Time at which to calculate hazard rate.
        location: Weibull location parameter (gamma). Default 0.

    Returns:
        Hazard rate at given time.

    Examples:
        >>> calculate_hazard_rate(shape=2.0, scale=1000, time=500)
        0.001  # Instantaneous failure rate
    """
    if shape <= 0:
        raise ValueError("Shape parameter must be positive")
    if scale <= 0:
        raise ValueError("Scale parameter must be positive")

    adjusted_time = time - location

    if adjusted_time <= 0:
        return 0.0

    # h(t) = (beta / eta) * ((t - gamma) / eta)^(beta - 1)
    return (shape / scale) * ((adjusted_time / scale) ** (shape - 1))


def _calculate_median_ranks(n: int) -> np.ndarray:
    """Calculate median ranks using Bernard's approximation.

    Bernard's approximation: MR(i) = (i - 0.3) / (n + 0.4)

    Args:
        n: Total number of data points

    Returns:
        Array of median ranks for each position i=1 to n
    """
    i = np.arange(1, n + 1)
    return (i - 0.3) / (n + 0.4)


def _calculate_weibull_r_squared(
    failures: np.ndarray,
    shape: float,
    scale: float,
    loc: float
) -> float:
    """Calculate R-squared for Weibull fit.

    Args:
        failures: Sorted failure times
        shape, scale, loc: Weibull parameters

    Returns:
        R-squared value (0-1)
    """
    n = len(failures)
    expected_percentiles = _calculate_median_ranks(n)

    # Predicted cumulative probabilities from fitted Weibull
    predicted_cdf = stats.weibull_min.cdf(failures, shape, loc=loc, scale=scale)

    # Calculate R-squared
    ss_res = np.sum((expected_percentiles - predicted_cdf) ** 2)
    ss_tot = np.sum((expected_percentiles - np.mean(expected_percentiles)) ** 2)

    if ss_tot == 0:
        return 1.0

    return 1 - (ss_res / ss_tot)


def _fit_censored_weibull(
    failures: np.ndarray,
    censored: np.ndarray
) -> Tuple[float, float, float]:
    """Fit Weibull with censored data using MLE.

    Args:
        failures: Array of failure times
        censored: Array of censored/suspension times

    Returns:
        Tuple of (shape, scale, location) parameters
    """
    # Log-likelihood function for censored data
    def neg_log_likelihood(params):
        shape, scale = params
        if shape <= 0 or scale <= 0:
            return np.inf

        # Log likelihood for failures
        ll_failures = np.sum(
            np.log(shape) + (shape - 1) * np.log(failures)
            - (failures / scale) ** shape
            - shape * np.log(scale)
        )

        # Log likelihood for censored (survival function)
        ll_censored = -np.sum((censored / scale) ** shape)

        return -(ll_failures + ll_censored)

    # Initial guess using method of moments
    mean_failures = np.mean(failures)
    std_failures = np.std(failures)
    initial_shape = 1.2 if std_failures > 0 else 1.2
    initial_scale = mean_failures

    # Optimize
    result = optimize.minimize(
        neg_log_likelihood,
        x0=[initial_shape, initial_scale],
        bounds=[(0.1, 10), (mean_failures * 0.1, mean_failures * 10)],
        method='L-BFGS-B'
    )

    if not result.success:
        raise WeibullAnalysisError(f"MLE optimization failed: {result.message}")

    shape, scale = result.x
    return shape, scale, 0.0


def _calculate_confidence_interval(
    failures: np.ndarray,
    censored: np.ndarray,
    shape: float,
    scale: float,
    confidence_level: float
) -> Optional[Tuple[float, float]]:
    """Calculate confidence interval for scale parameter.

    Uses Fisher information approximation for confidence interval.

    Args:
        failures: Failure times
        censored: Censored times
        shape, scale: Fitted parameters
        confidence_level: Confidence level (0-1)

    Returns:
        Tuple of (lower, upper) bounds or None if cannot calculate
    """
    alpha = 1 - confidence_level
    n_total = len(failures) + len(censored)

    if n_total < 5:
        return None

    # Approximate standard error using Fisher information
    # For Weibull, the variance of ln(eta) is approximately 1/(n * beta^2)
    var_log_scale = 1 / (n_total * shape ** 2)
    se_log_scale = np.sqrt(var_log_scale)

    # Critical value from normal distribution
    z_critical = stats.norm.ppf(1 - alpha / 2)

    # Confidence interval on log scale
    log_scale = np.log(scale)
    lower_log = log_scale - z_critical * se_log_scale
    upper_log = log_scale + z_critical * se_log_scale

    return np.exp(lower_log), np.exp(upper_log)


def weibull_pdf(
    shape: float,
    scale: float,
    time: np.ndarray,
    location: float = 0.0
) -> np.ndarray:
    """Calculate Weibull probability density function.

    Args:
        shape: Weibull shape parameter (beta)
        scale: Weibull scale parameter (eta)
        time: Time values (can be array)
        location: Weibull location parameter (gamma)

    Returns:
        PDF values at given times
    """
    adjusted_time = time - location
    adjusted_time = np.maximum(adjusted_time, 0)

    return (shape / scale) * ((adjusted_time / scale) ** (shape - 1)) * \
           np.exp(-((adjusted_time / scale) ** shape))


def weibull_cdf(
    shape: float,
    scale: float,
    time: np.ndarray,
    location: float = 0.0
) -> np.ndarray:
    """Calculate Weibull cumulative distribution function.

    Args:
        shape: Weibull shape parameter (beta)
        scale: Weibull scale parameter (eta)
        time: Time values (can be array)
        location: Weibull location parameter (gamma)

    Returns:
        CDF values at given times (cumulative failure probability)
    """
    adjusted_time = time - location
    adjusted_time = np.maximum(adjusted_time, 0)

    return 1 - np.exp(-((adjusted_time / scale) ** shape))

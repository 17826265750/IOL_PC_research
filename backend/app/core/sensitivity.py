"""
Sensitivity analysis for lifetime model parameters.

This module provides tools for analyzing how changes in input parameters
affect model outputs. Sensitivity analysis is crucial for:

1. Understanding which parameters most influence lifetime predictions
2. Identifying parameters that require accurate measurement
3. Quantifying uncertainty propagation through the model
4. Supporting robust design decisions

Key methods:
- Single-parameter sensitivity (elasticity analysis)
- Tornado analysis for comparative visualization
- Two-parameter sensitivity for interaction effects
- Global sensitivity indices (Sobol)

References:
- Saltelli, A. et al. "Global Sensitivity Analysis"
- CIPS 2008 IGBT lifetime model
"""

from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Callable

import numpy as np


@dataclass
class SensitivityResult:
    """Result of single parameter sensitivity analysis.

    Attributes:
        parameter_name: Name of the varied parameter.
        base_value: Base value of the parameter.
        elasticity: Percentage change in output divided by percentage
            change in input. |E| > 1 indicates elastic (sensitive),
            |E| < 1 indicates inelastic (less sensitive).
        sensitivity_index: Normalized sensitivity coefficient.
            Higher values indicate greater sensitivity.
        min_output: Model output at minimum parameter value.
        max_output: Model output at maximum parameter value.
        output_at_base: Model output at base parameter value.
        percent_change: Percent change in output from min to max.
    """
    parameter_name: str
    base_value: float
    elasticity: float
    sensitivity_index: float
    min_output: float
    max_output: float
    output_at_base: float
    percent_change: float


@dataclass
class TornadoData:
    """Data for tornado diagram visualization.

    Tornado diagrams show the impact of each parameter on the output,
    sorted by impact magnitude. Each bar shows the range of output
    values as the parameter varies from its minimum to maximum.

    Attributes:
        parameter: Parameter name.
        low_value: Output value at parameter minimum.
        high_value: Output value at parameter maximum.
        base_output: Output value at base parameter value.
        range_width: Absolute difference between high and low.
        percent_change: Percent change from base.
    """
    parameter: str
    low_value: float
    high_value: float
    base_output: float
    range_width: float
    percent_change: float


@dataclass
class SobolResult:
    """Result of Sobol global sensitivity analysis.

    Attributes:
        first_order: First-order sensitivity indices (main effects).
        total_order: Total-order sensitivity indices (including interactions).
        parameters: List of parameter names.
        confidence_intervals: Confidence intervals for indices (if computed).
    """
    first_order: Dict[str, float]
    total_order: Dict[str, float]
    parameters: List[str]
    confidence_intervals: Optional[Dict[str, Tuple[float, float]]] = None


class SensitivityAnalysisError(Exception):
    """Exception raised for errors in sensitivity analysis."""
    pass


def single_parameter_sensitivity(
    model_func: Callable,
    base_params: Dict[str, float],
    parameter_name: str,
    variation_range: Tuple[float, float],
    steps: int = 10
) -> SensitivityResult:
    """Perform single parameter sensitivity analysis.

    Varies one parameter while holding others constant. Calculates
    elasticity, which measures the percentage change in output per
    percentage change in input.

    Args:
        model_func: Model function with signature f(**params) -> output.
            Accepts all parameters as keyword arguments.
        base_params: Dictionary of all parameter base values.
        parameter_name: Name of parameter to vary.
        variation_range: (min_value, max_value) for the parameter.
        steps: Number of evaluation points within the range.

    Returns:
        SensitivityResult with elasticity and sensitivity metrics.

    Raises:
        SensitivityAnalysisError: If parameter not found or model fails.

    Examples:
        >>> def lifetime(**params):
        ...     return 1e6 * params['dTj']**-2 * np.exp(-1000/params['T'])
        >>> base = {'dTj': 80, 'T': 400}
        >>> result = single_parameter_sensitivity(
        ...     lifetime, base, 'dTj', (40, 120)
        ... )
        >>> print(f"Elasticity: {result.elasticity:.2f}")
    """
    # Validate inputs
    if parameter_name not in base_params:
        raise SensitivityAnalysisError(
            f"Parameter '{parameter_name}' not in base_params"
        )

    if steps < 2:
        raise SensitivityAnalysisError("At least 2 steps required")

    min_val, max_val = variation_range
    if min_val >= max_val:
        raise SensitivityAnalysisError("Invalid variation range")

    base_value = base_params[parameter_name]

    # Generate parameter values
    param_values = np.linspace(min_val, max_val, steps)

    # Evaluate model at each point
    outputs = []
    for val in param_values:
        params = base_params.copy()
        params[parameter_name] = val
        try:
            output = model_func(**params)
            outputs.append(float(output))
        except Exception as e:
            raise SensitivityAnalysisError(
                f"Model evaluation failed: {str(e)}"
            ) from e

    outputs = np.array(outputs)

    # Calculate metrics
    min_output = outputs.min()
    max_output = outputs.max()
    output_at_base = model_func(**base_params)

    # Find closest to base for elasticity calculation
    base_idx = np.argmin(np.abs(param_values - base_value))

    # Calculate elasticity using finite difference
    # Elasticity E = (%Δoutput) / (%Δinput)
    if base_idx > 0 and base_idx < len(param_values) - 1:
        # Use central difference if possible
        delta_x = param_values[base_idx + 1] - param_values[base_idx - 1]
        delta_y = outputs[base_idx + 1] - outputs[base_idx - 1]

        if delta_x != 0 and base_value != 0 and output_at_base != 0:
            elasticity = (delta_y / output_at_base) / (delta_x / base_value)
        else:
            elasticity = 0.0
    elif len(param_values) >= 2:
        # Use forward/backward difference
        if base_idx == 0:
            delta_x = param_values[1] - param_values[0]
            delta_y = outputs[1] - outputs[0]
        else:
            delta_x = param_values[-1] - param_values[-2]
            delta_y = outputs[-1] - outputs[-2]

        if delta_x != 0 and base_value != 0 and output_at_base != 0:
            elasticity = (delta_y / output_at_base) / (delta_x / base_value)
        else:
            elasticity = 0.0
    else:
        elasticity = 0.0

    # Sensitivity index (normalized by parameter range)
    param_range = max_val - min_val
    if param_range > 0:
        sensitivity_index = (max_output - min_output) / param_range
    else:
        sensitivity_index = 0.0

    # Percent change from base
    if output_at_base != 0:
        percent_change = ((max_output - min_output) / output_at_base) * 100
    else:
        percent_change = 0.0

    return SensitivityResult(
        parameter_name=parameter_name,
        base_value=base_value,
        elasticity=elasticity,
        sensitivity_index=sensitivity_index,
        min_output=min_output,
        max_output=max_output,
        output_at_base=output_at_base,
        percent_change=percent_change
    )


def tornado_analysis(
    model_func: Callable,
    base_params: Dict[str, float],
    parameter_ranges: Dict[str, Tuple[float, float]],
    sort_by: str = 'impact'
) -> List[TornadoData]:
    """Generate tornado diagram data.

    Evaluates model at min/max values for each parameter while holding
    others at base values. Results are sorted by impact magnitude.

    Args:
        model_func: Model function f(**params) -> output.
        base_params: Dictionary of all parameter base values.
        parameter_ranges: Dictionary mapping parameter names to
            (min_value, max_value) tuples.
        sort_by: How to sort results. Options:
            - 'impact': Sort by absolute output range (default)
            - 'name': Sort alphabetically by parameter name
            - 'percent': Sort by percent change from base

    Returns:
        List of TornadoData sorted by specified criterion.

    Examples:
        >>> ranges = {'dTj': (40, 120), 'T': (350, 450), 't_on': (0.1, 10)}
        >>> tornado = tornado_analysis(lifetime_model, base, ranges)
        >>> # Plot as horizontal bar chart with base as center line
    """
    if not parameter_ranges:
        return []

    # Calculate sensitivity for each parameter
    results = []

    for param_name, (min_val, max_val) in parameter_ranges.items():
        if param_name not in base_params:
            continue

        if min_val >= max_val:
            continue

        # Calculate base output
        base_output = model_func(**base_params)

        # Calculate output at min
        params_min = base_params.copy()
        params_min[param_name] = min_val
        try:
            output_min = model_func(**params_min)
        except Exception:
            output_min = np.nan

        # Calculate output at max
        params_max = base_params.copy()
        params_max[param_name] = max_val
        try:
            output_max = model_func(**params_max)
        except Exception:
            output_max = np.nan

        # Determine which is low/high for visualization
        low_value = min(output_min, output_max)
        high_value = max(output_min, output_max)

        range_width = high_value - low_value

        # Percent change from base
        if base_output != 0 and not np.isnan(low_value) and not np.isnan(high_value):
            # Use the direction of change from base
            percent_change = ((high_value - base_output) / base_output) * 100
        else:
            percent_change = 0.0

        results.append(TornadoData(
            parameter=param_name,
            low_value=low_value,
            high_value=high_value,
            base_output=base_output,
            range_width=range_width,
            percent_change=percent_change
        ))

    # Sort results
    if sort_by == 'impact':
        results.sort(key=lambda x: x.range_width, reverse=True)
    elif sort_by == 'name':
        results.sort(key=lambda x: x.parameter)
    elif sort_by == 'percent':
        results.sort(key=lambda x: abs(x.percent_change), reverse=True)

    return results


def two_parameter_sensitivity(
    model_func: Callable,
    base_params: Dict[str, float],
    param1_name: str,
    param1_range: Tuple[float, float],
    param2_name: str,
    param2_range: Tuple[float, float],
    steps: int = 20
) -> np.ndarray:
    """Generate 2D sensitivity heatmap data.

    Evaluates model over a grid of two parameter values to identify
    interaction effects. Useful for heatmap visualization.

    Args:
        model_func: Model function f(**params) -> output.
        base_params: Dictionary of all parameter base values.
        param1_name: Name of first parameter (x-axis).
        param1_range: (min, max) for first parameter.
        param2_name: Name of second parameter (y-axis).
        param2_range: (min, max) for second parameter.
        steps: Number of grid points per dimension (total = steps²).

    Returns:
        2D array of output values. Array indices correspond to:
            - First index: param2 values (y-axis, rows)
            - Second index: param1 values (x-axis, columns)

    Examples:
        >>> heatmap = two_parameter_sensitivity(
        ...     lifetime, base, 'dTj', (40, 120), 'T', (350, 450), steps=20
        ... )
        >>> plt.imshow(heatmap, extent=[40, 120, 350, 450], aspect='auto')
        >>> plt.colorbar(label='Lifetime (cycles)')
    """
    # Validate
    if param1_name not in base_params:
        raise SensitivityAnalysisError(f"Parameter '{param1_name}' not found")
    if param2_name not in base_params:
        raise SensitivityAnalysisError(f"Parameter '{param2_name}' not found")

    if steps < 2:
        raise SensitivityAnalysisError("At least 2 steps required per dimension")

    min1, max1 = param1_range
    min2, max2 = param2_range

    if min1 >= max1 or min2 >= max2:
        raise SensitivityAnalysisError("Invalid parameter ranges")

    # Generate parameter grids
    param1_values = np.linspace(min1, max1, steps)
    param2_values = np.linspace(min2, max2, steps)

    # Evaluate model on grid
    outputs = np.zeros((steps, steps))

    for i, val2 in enumerate(param2_values):
        for j, val1 in enumerate(param1_values):
            params = base_params.copy()
            params[param1_name] = val1
            params[param2_name] = val2

            try:
                outputs[i, j] = model_func(**params)
            except Exception:
                outputs[i, j] = np.nan

    return outputs


def calculate_elasticity(
    base_value: float,
    base_output: float,
    perturbed_value: float,
    perturbed_output: float
) -> float:
    """Calculate elasticity coefficient.

    Elasticity measures the responsiveness of output to changes in input.
    An elasticity of -2 means a 1% increase in input causes a 2% decrease
    in output.

    E = (%Δoutput) / (%Δinput)

    Args:
        base_value: Original parameter value.
        base_output: Original model output.
        perturbed_value: New parameter value.
        perturbed_output: New model output.

    Returns:
        Elasticity coefficient. Can be positive or negative.
        Values > 1 indicate elastic (sensitive).
        Values < 1 indicate inelastic (less sensitive).

    Examples:
        >>> calculate_elasticity(100, 1000, 110, 800)
        -2.0  # 10% increase in input causes 20% decrease in output
    """
    if base_value == 0 or base_output == 0:
        return 0.0

    percent_change_input = (perturbed_value - base_value) / base_value
    percent_change_output = (perturbed_output - base_output) / base_output

    if percent_change_input == 0:
        return 0.0

    return percent_change_output / percent_change_input


def sobol_sensitivity(
    model_func: Callable,
    param_ranges: Dict[str, Tuple[float, float]],
    n_samples: int = 1000,
    calc_second_order: bool = False
) -> SobolResult:
    """Perform Sobol global sensitivity analysis.

    Sobol analysis decomposes output variance into contributions from
    each parameter and their interactions. Uses Saltelli's algorithm
    for efficient computation.

    First-order indices (S_i): Main effect contribution of each parameter.
    Total-order indices (S_Ti): Total contribution including interactions.

    Args:
        model_func: Model function f(**params) -> output.
        param_ranges: Dictionary mapping parameter names to (min, max).
        n_samples: Number of samples for Monte Carlo integration.
            More samples = more accurate but slower.
        calc_second_order: Whether to calculate second-order interaction
            indices (computationally expensive).

    Returns:
        SobolResult with first-order and total-order indices.

    Note:
        This is a simplified implementation. For production use,
        consider using the SALib library which provides optimized
        Sobol sequence generation.

    Examples:
        >>> result = sobol_sensitivity(lifetime_model, ranges, n_samples=1000)
        >>> print(f"Most sensitive: {max(result.first_order, key=result.first_order.get)}")
    """
    if not param_ranges:
        raise SensitivityAnalysisError("No parameter ranges provided")

    if n_samples < 100:
        raise SensitivityAnalysisError("At least 100 samples recommended")

    param_names = list(param_ranges.keys())
    n_params = len(param_names)

    # Generate sample matrices using pseudo-Sobol sequence
    # Using Saltelli's scheme: N*(2D+2) model evaluations
    np.random.seed(42)  # For reproducibility

    # Matrix A: base samples
    A = _sample_latin_hypercube(param_ranges, n_samples)

    # Matrix B: independent samples
    B = _sample_latin_hypercube(param_ranges, n_samples)

    # Matrices C_i: column i from B, others from A
    C_matrices = []
    for i in range(n_params):
        C_i = A.copy()
        C_i[:, i] = B[:, i]
        C_matrices.append(C_i)

    # Evaluate model
    f_A = np.array([model_func(**dict(zip(param_names, row))) for row in A])
    f_B = np.array([model_func(**dict(zip(param_names, row))) for row in B])
    f_C = []
    for C_i in C_matrices:
        f_C_i = np.array([model_func(**dict(zip(param_names, row))) for row in C_i])
        f_C.append(f_C_i)

    # Calculate variance
    f_all = np.concatenate([f_A, f_B] + f_C)
    variance = np.var(f_all)

    if variance == 0:
        # No variance in output
        return SobolResult(
            first_order={name: 0.0 for name in param_names},
            total_order={name: 0.0 for name in param_names},
            parameters=param_names
        )

    # Calculate first-order indices (S_i)
    first_order = {}
    for i, name in enumerate(param_names):
        numerator = np.mean(f_A * (f_C[i] - f_B))
        first_order[name] = numerator / variance

    # Calculate total-order indices (S_Ti)
    total_order = {}
    for i, name in enumerate(param_names):
        numerator = np.mean((f_B - f_C[i]) ** 2)
        total_order[name] = numerator / (2 * variance)

    # Ensure indices are in valid range [0, 1]
    for name in param_names:
        first_order[name] = max(0, min(1, first_order[name]))
        total_order[name] = max(0, min(1, total_order[name]))

    return SobolResult(
        first_order=first_order,
        total_order=total_order,
        parameters=param_names
    )


def _sample_latin_hypercube(
    param_ranges: Dict[str, Tuple[float, float]],
    n_samples: int
) -> np.ndarray:
    """Generate Latin Hypercube samples.

    Args:
        param_ranges: Dictionary of parameter ranges.
        n_samples: Number of samples to generate.

    Returns:
        Array of shape (n_samples, n_params) with samples.
    """
    param_names = list(param_ranges.keys())
    n_params = len(param_names)

    samples = np.zeros((n_samples, n_params))

    for i, name in enumerate(param_names):
        min_val, max_val = param_ranges[name]

        # Generate stratified samples
        perm = np.random.permutation(n_samples)
        samples[:, i] = (perm + np.random.random(n_samples)) / n_samples

        # Scale to parameter range
        samples[:, i] = min_val + samples[:, i] * (max_val - min_val)

    return samples


def morris_sensitivity(
    model_func: Callable,
    param_ranges: Dict[str, Tuple[float, float]],
    n_trajectories: int = 10,
    delta: float = 0.5
) -> Dict[str, Dict[str, float]]:
    """Perform Morris elementary effects screening.

    Morris method is a qualitative screening technique that identifies
    which parameters could be considered to have negligible effects.
    Computationally cheaper than Sobol for high-dimensional problems.

    For each parameter, calculates:
    - mu*: Mean of absolute elementary effects (overall influence)
    - mu: Mean of elementary effects (direction of influence)
    - sigma: Standard deviation (nonlinear effects/interactions)

    Args:
        model_func: Model function f(**params) -> output.
        param_ranges: Dictionary mapping parameter names to (min, max).
        n_trajectories: Number of random trajectories.
        delta: Step size as fraction of parameter range (0-1).

    Returns:
        Dictionary with mu_star, mu, and sigma for each parameter.

    Examples:
        >>> result = morris_sensitivity(lifetime_model, ranges)
        >>> for param, metrics in sorted(
        ...     result.items(), key=lambda x: x[1]['mu_star'], reverse=True
        ... ):
        ...     print(f"{param}: mu*={metrics['mu_star']:.2f}")
    """
    param_names = list(param_ranges.keys())
    n_params = len(param_names)

    if n_trajectories < 4:
        raise SensitivityAnalysisError("At least 4 trajectories recommended")

    # Elementary effects storage
    elementary_effects = {name: [] for name in param_names}

    for _ in range(n_trajectories):
        # Random starting point
        base = {}
        for name, (min_val, max_val) in param_ranges.items():
            base[name] = min_val + np.random.random() * (max_val - min_val)

        # Random trajectory order
        order = list(param_names)
        np.random.shuffle(order)

        # Follow trajectory
        for i, param_name in enumerate(order):
            # Calculate elementary effect for this parameter
            min_val, max_val = param_ranges[param_name]
            step = delta * (max_val - min_val)

            # Choose direction
            direction = 1 if np.random.random() < 0.5 else -1

            # New value
            new_value = base[param_name] + direction * step

            # Clamp to range
            new_value = max(min_val, min(max_val, new_value))

            # Evaluate
            params_plus = base.copy()
            params_plus[param_name] = new_value

            try:
                f_base = model_func(**base)
                f_plus = model_func(**params_plus)

                ee = (f_plus - f_base) / (direction * step)
                elementary_effects[param_name].append(ee)

            except Exception:
                continue

            # Update base for next step
            base[param_name] = new_value

    # Calculate statistics
    results = {}
    for name in param_names:
        ees = np.array(elementary_effects[name])

        if len(ees) == 0:
            results[name] = {'mu_star': 0, 'mu': 0, 'sigma': 0}
            continue

        mu_star = np.mean(np.abs(ees))
        mu = np.mean(ees)
        sigma = np.std(ees, ddof=1)

        results[name] = {
            'mu_star': mu_star,
            'mu': mu,
            'sigma': sigma
        }

    return results


def monte_carlo_sensitivity(
    model_func: Callable,
    param_distributions: Dict[str, Tuple[float, float]],
    n_samples: int = 10000
) -> Dict[str, float]:
    """Perform Monte Carlo uncertainty propagation.

    Evaluates model with random parameter samples from specified
    distributions to estimate output uncertainty. Calculates
    correlation coefficients between parameters and output.

    Args:
        model_func: Model function f(**params) -> output.
        param_distributions: Dictionary mapping parameter names to
            (mean, std) for normal distribution or (min, max) for uniform.
            Uses normal if mean > 0, otherwise uniform.
        n_samples: Number of Monte Carlo samples.

    Returns:
        Dictionary with statistics:
        - 'mean': Mean output
        - 'std': Standard deviation of output
        - 'min': Minimum output
        - 'max': Maximum output
        - 'q05': 5th percentile
        - 'q95': 95th percentile
        - Correlation coefficients for each parameter

    Examples:
        >>> dists = {'dTj': (80, 10), 'T': (400, 20)}
        >>> stats = monte_carlo_sensitivity(lifetime_model, dists)
        >>> print(f"Mean lifetime: {stats['mean']:.0f} ± {stats['std']:.0f}")
    """
    if n_samples < 100:
        raise SensitivityAnalysisError("At least 100 samples recommended")

    param_names = list(param_distributions.keys())
    n_params = len(param_names)

    # Generate samples
    samples = np.zeros((n_samples, n_params))
    for i, name in enumerate(param_names):
        val1, val2 = param_distributions[name]
        if val1 > 0:  # Assume normal (mean, std)
            samples[:, i] = np.random.normal(val1, val2, n_samples)
        else:  # Assume uniform (min, max)
            samples[:, i] = np.random.uniform(val1, val2, n_samples)

    # Evaluate model
    outputs = np.zeros(n_samples)
    for i in range(n_samples):
        params = dict(zip(param_names, samples[i, :]))
        try:
            outputs[i] = model_func(**params)
        except Exception:
            outputs[i] = np.nan

    # Remove NaN values
    valid_mask = ~np.isnan(outputs)
    outputs_valid = outputs[valid_mask]
    samples_valid = samples[valid_mask, :]

    if len(outputs_valid) == 0:
        return {
            'mean': np.nan,
            'std': np.nan,
            'min': np.nan,
            'max': np.nan,
            'q05': np.nan,
            'q95': np.nan
        }

    # Calculate statistics
    result = {
        'mean': np.mean(outputs_valid),
        'std': np.std(outputs_valid, ddof=1),
        'min': np.min(outputs_valid),
        'max': np.max(outputs_valid),
        'q05': np.percentile(outputs_valid, 5),
        'q95': np.percentile(outputs_valid, 95)
    }

    # Calculate correlation coefficients
    for i, name in enumerate(param_names):
        if len(samples_valid) > 1:
            corr = np.corrcoef(samples_valid[:, i], outputs_valid)[0, 1]
            result[name] = corr if not np.isnan(corr) else 0.0
        else:
            result[name] = 0.0

    return result

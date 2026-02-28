"""
Non-linear least squares fitting for lifetime model parameters.

This module provides functions for fitting non-linear lifetime models
to experimental data using various optimization methods. Supports
parameter estimation with uncertainty quantification for reliability
engineering applications.

Key features:
- Levenberg-Marquardt and trust region reflective algorithms
- Parameter bounds and constraints
- Statistical metrics (R², RMSE, confidence intervals)
- Specialized fitting for CIPS 2008 IGBT lifetime model

References:
- CIPS 2008: IGBT Lifetime Model
- SciPy optimization documentation
"""

from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Callable

import numpy as np
from scipy.optimize import curve_fit, least_squares
from scipy import stats


@dataclass
class FittingResult:
    """Result of model parameter fitting.

    Attributes:
        parameters: Dictionary of fitted parameter names to values.
        std_errors: Dictionary of parameter standard errors.
        r_squared: Coefficient of determination (0-1). Higher is better.
        rmse: Root mean square error (same units as y_data).
        residuals: Array of residuals (y_actual - y_predicted).
        covariance: Covariance matrix of fitted parameters.
        confidence_intervals: Dictionary of (lower, upper) bounds for each
            parameter at 95% confidence.
        fixed_params: Dictionary of β parameters that were fixed (not fitted).
        fixed_data_values: Dictionary of data values for fixed parameters.
        auto_fixed_info: List of strings describing auto-fixed parameters.
    """
    parameters: Dict[str, float]
    std_errors: Dict[str, float]
    r_squared: float
    rmse: float
    residuals: np.ndarray
    covariance: np.ndarray
    confidence_intervals: Dict[str, Tuple[float, float]]
    fixed_params: Optional[Dict[str, float]] = None
    fixed_data_values: Optional[Dict[str, float]] = None
    auto_fixed_info: Optional[List[str]] = None


class FittingError(Exception):
    """Exception raised for errors in model fitting."""
    pass


def fit_lifetime_model(
    model_func: Callable,
    x_data: np.ndarray,
    y_data: np.ndarray,
    initial_params: Dict[str, float],
    param_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
    method: str = 'lm'
) -> FittingResult:
    """Fit lifetime model parameters using non-linear least squares.

    This function performs non-linear regression to fit model parameters
    to experimental data. Supports multiple optimization algorithms and
    parameter constraints.

    Args:
        model_func: Model function with signature f(x, **params) -> y.
            The function should accept x data followed by parameters
            as separate arguments (scipy curve_fit convention).
        x_data: Independent variable data (e.g., ΔTj, temperature).
        y_data: Dependent variable data (e.g., cycles to failure).
        initial_params: Dictionary of parameter names to initial guesses.
        param_bounds: Optional dictionary of (lower, upper) bounds for
            each parameter. Unbounded parameters default to (-inf, inf).
        method: Optimization method. Options:
            - 'lm': Levenberg-Marquardt (default, no bounds)
            - 'trf': Trust Region Reflective (supports bounds)
            - 'dogbox': Dogleg algorithm with bounds

    Returns:
        FittingResult with optimized parameters and statistics.

    Raises:
        FittingError: If fitting fails due to invalid data, poor initial
            guesses, or optimization failure.

    Examples:
        >>> def arrhenius_model(T, A, Ea):
        ...     return A * np.exp(-Ea / (8.314 * T))
        >>> x = np.array([300, 350, 400, 450])
        >>> y = np.array([10000, 1000, 100, 10])
        >>> init = {'A': 1e10, 'Ea': 100000}
        >>> result = fit_lifetime_model(arrhenius_model, x, y, init)
    """
    # Validate inputs
    x_data = np.asarray(x_data, dtype=float)
    y_data = np.asarray(y_data, dtype=float)

    if len(x_data) != len(y_data):
        raise FittingError("x_data and y_data must have same length")

    if len(x_data) < 2:
        raise FittingError("At least 2 data points required for fitting")

    if len(initial_params) == 0:
        raise FittingError("At least one parameter must be specified")

    # Extract parameter names, initial values, and bounds
    param_names = list(initial_params.keys())
    p0 = [initial_params[name] for name in param_names]

    # Set up bounds
    if param_bounds:
        lower_bounds = []
        upper_bounds = []
        for name in param_names:
            if name in param_bounds:
                lower_bounds.append(param_bounds[name][0])
                upper_bounds.append(param_bounds[name][1])
            else:
                lower_bounds.append(-np.inf)
                upper_bounds.append(np.inf)

        # Method selection based on bounds
        if method == 'lm' and any(b != -np.inf for b in lower_bounds):
            method = 'trf'  # LM doesn't support bounds
    else:
        lower_bounds = [-np.inf] * len(param_names)
        upper_bounds = [np.inf] * len(param_names)

    # Wrap model function for scipy
    def scipy_model(x, *params):
        param_dict = dict(zip(param_names, params))
        return model_func(x, **param_dict)

    try:
        # Perform fitting
        popt, pcov, infodict, mesg, ier = curve_fit(
            scipy_model,
            x_data,
            y_data,
            p0=p0,
            bounds=(lower_bounds, upper_bounds) if param_bounds else (-np.inf, np.inf),
            method=method,
            full_output=True
        )

        if ier not in [1, 2, 3, 4]:
            raise FittingError(f"Optimization failed: {mesg}")

    except RuntimeError as e:
        raise FittingError(f"Fitting optimization failed: {str(e)}") from e

    # Calculate predictions
    y_pred = scipy_model(x_data, *popt)
    residuals = y_data - y_pred

    # Calculate statistics
    r_squared = calculate_r_squared(y_data, y_pred)
    rmse = calculate_rmse(y_data, y_pred)

    # Calculate parameter errors
    try:
        perr = np.sqrt(np.diag(pcov))
    except (np.linalg.LinAlgError, ValueError):
        # Covariance matrix may be singular or poorly conditioned
        perr = np.full(len(popt), np.nan)

    # Build result dictionaries
    parameters = dict(zip(param_names, popt))
    std_errors = dict(zip(param_names, perr))

    # Calculate confidence intervals (95%)
    confidence_intervals = {}
    for name, val, err in zip(param_names, popt, perr):
        if np.isnan(err):
            confidence_intervals[name] = (val, val)
        else:
            z_critical = stats.norm.ppf(0.975)  # 95% CI
            ci = (val - z_critical * err, val + z_critical * err)
            confidence_intervals[name] = ci

    return FittingResult(
        parameters=parameters,
        std_errors=std_errors,
        r_squared=r_squared,
        rmse=rmse,
        residuals=residuals,
        covariance=pcov,
        confidence_intervals=confidence_intervals
    )


def calculate_r_squared(
    y_actual: np.ndarray,
    y_predicted: np.ndarray
) -> float:
    """Calculate coefficient of determination (R²).

    R² measures the proportion of variance in the dependent variable
    that is predictable from the independent variable(s).

    R² = 1 - SS_res / SS_tot

    where:
        SS_res = sum((y_actual - y_predicted)²)
        SS_tot = sum((y_actual - mean(y_actual))²)

    Args:
        y_actual: Actual observed values.
        y_predicted: Model predicted values.

    Returns:
        R² value between 0 and 1. Values close to 1 indicate better fit.
        Can be negative for very poor fits (worse than horizontal line).

    Examples:
        >>> y_true = np.array([1, 2, 3, 4, 5])
        >>> y_pred = np.array([1.1, 1.9, 3.1, 4.0, 5.1])
        >>> calculate_r_squared(y_true, y_pred)
        0.985
    """
    y_actual = np.asarray(y_actual)
    y_predicted = np.asarray(y_predicted)

    ss_res = np.sum((y_actual - y_predicted) ** 2)
    ss_tot = np.sum((y_actual - np.mean(y_actual)) ** 2)

    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0

    return 1 - (ss_res / ss_tot)


def calculate_rmse(
    y_actual: np.ndarray,
    y_predicted: np.ndarray
) -> float:
    """Calculate root mean square error.

    RMSE measures the average magnitude of prediction errors.
    Lower values indicate better fit. RMSE is in the same units
    as the input data.

    RMSE = sqrt(mean((y_actual - y_predicted)²))

    Args:
        y_actual: Actual observed values.
        y_predicted: Model predicted values.

    Returns:
        RMSE value (same units as input data).

    Examples:
        >>> y_true = np.array([10, 20, 30])
        >>> y_pred = np.array([12, 18, 32])
        >>> calculate_rmse(y_true, y_pred)
        2.0
    """
    y_actual = np.asarray(y_actual)
    y_predicted = np.asarray(y_predicted)

    return np.sqrt(np.mean((y_actual - y_predicted) ** 2))


def calculate_mae(
    y_actual: np.ndarray,
    y_predicted: np.ndarray
) -> float:
    """Calculate mean absolute error.

    MAE measures the average absolute prediction error.
    Less sensitive to outliers than RMSE.

    Args:
        y_actual: Actual observed values.
        y_predicted: Model predicted values.

    Returns:
        MAE value (same units as input data).
    """
    y_actual = np.asarray(y_actual)
    y_predicted = np.asarray(y_predicted)

    return np.mean(np.abs(y_actual - y_predicted))


def calculate_mape(
    y_actual: np.ndarray,
    y_predicted: np.ndarray,
    epsilon: float = 1e-10
) -> float:
    """Calculate mean absolute percentage error.

    MAPE expresses prediction error as a percentage.
    Note: MAPE is undefined when actual values are zero.

    Args:
        y_actual: Actual observed values.
        y_predicted: Model predicted values.
        epsilon: Small value to avoid division by zero.

    Returns:
        MAPE as a percentage (0-100).
    """
    y_actual = np.asarray(y_actual)
    y_predicted = np.asarray(y_predicted)

    # Avoid division by zero
    y_actual_safe = np.where(np.abs(y_actual) < epsilon, epsilon, y_actual)

    return np.mean(np.abs((y_actual - y_predicted) / y_actual_safe)) * 100


def fit_cips2008_model(
    experiment_data: List[Dict],
    fixed_params: Optional[Dict[str, float]] = None,
    couple_vd_to_k: bool = False
) -> FittingResult:
    """Specialized fitting for CIPS 2008 IGBT lifetime model parameters.

    The CIPS 2008 model for IGBT lifetime is:

    Nf = K * (ΔTj)^β1 * exp(β2 / (Tj_max + 273.15)) *
         (t_on)^β3 * (I / I_nom)^β4 * (V / V_nom)^β5 * D^β6

    where:
        Nf: Cycles to failure
        K: Calibration constant
        ΔTj: Junction temperature swing
        Tj_max: Maximum junction temperature (C)
        t_on: On-time duration
        I: Current (relative or absolute)
        V: Voltage (relative or absolute)
        D: Duty cycle (0-1)

    Args:
        experiment_data: List of dictionaries, each containing:
            - 'dTj': Temperature swing (K or C)
            - 'Tj_max': Max junction temperature (C)
            - 't_on': On-time (s or similar)
            - 'I': Current (A or relative)
            - 'V': Voltage (V or relative)
            - 'D': Duty cycle (0-1)
            - 'Nf': Cycles to failure (observed)
        fixed_params: Optional dictionary of parameters to fix at
            specific values (not fitted). For example: {'β3': -0.5}
        couple_vd_to_k: If True, β5/β6 effects are coupled into K_eff
            (requires V and D to be constant across all experiment points).

    Returns:
        FittingResult with fitted K and β parameters.

    Raises:
        FittingError: If data is invalid or fitting fails.

    Examples:
        >>> data = [
        ...     {'dTj': 80, 'Tj_max': 125, 't_on': 1, 'I': 1, 'V': 1, 'D': 0.5, 'Nf': 1e6},
        ...     {'dTj': 100, 'Tj_max': 150, 't_on': 1, 'I': 1, 'V': 1, 'D': 0.5, 'Nf': 1e5},
        ... ]
        >>> result = fit_cips2008_model(data)
    """
    if not experiment_data:
        raise FittingError("Experiment data cannot be empty")

    # Validate and extract data
    n_points = len(experiment_data)

    # Extract arrays
    try:
        dTj = np.array([d['dTj'] for d in experiment_data])
        Tj_max = np.array([d['Tj_max'] for d in experiment_data])
        t_on = np.array([d['t_on'] for d in experiment_data])
        I = np.array([d['I'] for d in experiment_data])
        V = np.array([d['V'] for d in experiment_data])
        D = np.array([d['D'] for d in experiment_data])
        Nf_observed = np.array([d['Nf'] for d in experiment_data])
    except KeyError as e:
        raise FittingError(f"Missing required field in experiment data: {e}")

    if np.any(Nf_observed <= 0):
        raise FittingError("Nf values must be strictly positive for log-space fitting")

    if np.any(dTj <= 0) or np.any(t_on <= 0) or np.any(I <= 0) or np.any(V <= 0) or np.any(D <= 0):
        raise FittingError("All input variables (dTj, t_on, I, V, D) must be strictly positive")

    # Default β parameters (typical values from CIPS 2008 paper)
    default_betas = {
        'β1': -4.423,  # ΔTj exponent
        'β2': 1285.0,  # Tj_max coefficient
        'β3': -0.462,  # ton exponent
        'β4': -0.716,  # I exponent
        'β5': -0.761,  # V exponent
        'β6': -0.5     # D exponent
    }

    # Determine which parameters are fixed (user-specified)
    fixed_betas = {}  # β parameters that are fixed (not fitted)
    fixed_data_values = {}  # Optional metadata values for fixed parameters

    # Process user-specified fixed parameters
    if fixed_params:
        for name, beta_value in fixed_params.items():
            if name in default_betas:
                fixed_betas[name] = beta_value

    auto_fixed_info = []

    # Optional model simplification for same product class:
    # couple V/D related terms into K_eff.
    coupled_betas = set()
    if couple_vd_to_k:
        if not np.allclose(V, V[0]):
            raise FittingError("V must be constant when couple_vd_to_k=True")
        if not np.allclose(D, D[0]):
            raise FittingError("D must be constant when couple_vd_to_k=True")

        coupled_betas.update({'β5', 'β6'})
        if 'β5' not in fixed_betas:
            fixed_betas['β5'] = default_betas['β5']
        if 'β6' not in fixed_betas:
            fixed_betas['β6'] = default_betas['β6']

        fixed_data_values['V'] = float(V[0])
        fixed_data_values['D'] = float(D[0])
        auto_fixed_info.append(f"V/D耦合到K_eff (V={V[0]}, D={D[0]})")

    # Store constant values for explicitly fixed non-coupled parameters (metadata only)
    if 'β3' in fixed_betas and np.allclose(t_on, t_on[0]):
        fixed_data_values['ton'] = float(t_on[0])
    if 'β4' in fixed_betas and np.allclose(I, I[0]):
        fixed_data_values['I'] = float(I[0])

    # Parameters to fit (only those not fixed/coupled)
    params_to_fit = [p for p in ['β1', 'β2', 'β3', 'β4', 'β5', 'β6'] if p not in fixed_betas and p not in coupled_betas]
    k_name = 'ln_K_eff' if couple_vd_to_k else 'ln_K'
    n_params = len(params_to_fit) + 1  # +1 for K/K_eff

    min_points = max(2, n_params)
    if n_points < min_points:
        raise FittingError(
            f"At least {min_points} data points required for fitting {n_params} parameters"
        )

    def cips2008_predict_ln(X, **params):
        """Prediction function in log-space for CIPS 2008 model."""
        beta_values = {}
        for beta_name in ['β1', 'β2', 'β3', 'β4', 'β5', 'β6']:
            if beta_name in params:
                beta_values[beta_name] = params[beta_name]
            elif beta_name in fixed_betas:
                beta_values[beta_name] = fixed_betas[beta_name]
            else:
                beta_values[beta_name] = default_betas[beta_name]

        Tj_max_kelvin = Tj_max + 273.15

        ln_nf = (
            params[k_name]
            + beta_values['β1'] * np.log(dTj)
            + beta_values['β2'] / Tj_max_kelvin
            + beta_values['β3'] * np.log(t_on)
            + beta_values['β4'] * np.log(I)
        )

        if not couple_vd_to_k:
            ln_nf = ln_nf + beta_values['β5'] * np.log(V) + beta_values['β6'] * np.log(D)

        return ln_nf

    # Initial guesses for parameters to fit
    initial_params = {k_name: float(np.log(max(np.median(Nf_observed), 1.0)))}
    for p in params_to_fit:
        initial_params[p] = default_betas[p]

    # Parameter bounds
    param_bounds = {
        'ln_K': (-50, 100),
        'ln_K_eff': (-50, 100),
        'β1': (-10, 0),
        'β2': (0, 20000),  # Relax upper bound for Arrhenius coefficient
        'β3': (-2, 1),
        'β4': (-5, 1),
        'β5': (-5, 1),
        'β6': (-2, 1)
    }
    fitted_bounds = {name: param_bounds[name] for name in initial_params.keys()}

    try:
        result = fit_lifetime_model(
            cips2008_predict_ln,
            np.arange(len(Nf_observed)),
            np.log(Nf_observed),
            initial_params,
            fitted_bounds
        )
    except FittingError as e:
        raise FittingError(f"CIPS 2008 model fitting failed: {str(e)}") from e

    # Build complete beta values (fitted + fixed)
    final_beta_values = {}
    for p in ['β1', 'β2', 'β3', 'β4', 'β5', 'β6']:
        if p in result.parameters:
            final_beta_values[p] = float(result.parameters[p])
        elif p in fixed_betas:
            final_beta_values[p] = float(fixed_betas[p])
        else:
            final_beta_values[p] = float(default_betas[p])

    ln_k_value = float(result.parameters[k_name])
    k_value = float(np.exp(ln_k_value))

    # Couple fixed-constant stress terms among β3~β6 into returned K.
    def _compute_coupled_factor(beta_keys: List[str]) -> Optional[float]:
        factor_log = 0.0
        data_map = {
            'β3': t_on,
            'β4': I,
            'β5': V,
            'β6': D,
        }

        used_any = False
        for beta_key in beta_keys:
            if beta_key not in fixed_betas:
                continue

            values = data_map[beta_key]
            if not np.allclose(values, values[0]):
                return None

            factor_log += final_beta_values[beta_key] * np.log(float(values[0]))
            used_any = True

        if not used_any:
            return None

        return float(np.exp(np.clip(factor_log, -700, 700)))

    vd_factor = _compute_coupled_factor(['β5', 'β6'])
    if vd_factor is None or vd_factor <= 0:
        vd_factor = 1.0

    full_coupled_factor = _compute_coupled_factor(['β3', 'β4', 'β5', 'β6'])
    if full_coupled_factor is None or full_coupled_factor <= 0:
        full_coupled_factor = 1.0

    if couple_vd_to_k:
        # fitted k_value is V/D-coupled, recover base K then couple all fixed terms
        k_base = float(k_value / vd_factor)
        returned_k = float(k_base * full_coupled_factor)
    else:
        # fitted k_value is base K, couple all fixed terms directly
        returned_k = float(k_value * full_coupled_factor)

    final_params = {'K': returned_k}
    for p in params_to_fit:
        if p in result.parameters:
            final_params[p] = float(result.parameters[p])

    # Build confidence intervals
    final_ci = {}
    if k_name in result.confidence_intervals:
        ln_k_ci = result.confidence_intervals[k_name]
        k_ci = (
            float(np.exp(np.clip(ln_k_ci[0], -700, 700))),
            float(np.exp(np.clip(ln_k_ci[1], -700, 700)))
        )
        scale_factor = float(returned_k / k_value) if k_value != 0 else 1.0
        final_ci['K'] = (float(k_ci[0] * scale_factor), float(k_ci[1] * scale_factor))

    for p in params_to_fit:
        if p in result.confidence_intervals:
            ci = result.confidence_intervals[p]
            final_ci[p] = (float(ci[0]), float(ci[1]))

    # Build std_errors
    final_std_errors = {}
    if k_name in result.std_errors and result.std_errors[k_name] is not None:
        sigma_ln_k = float(result.std_errors[k_name])
        final_std_errors['K'] = float(returned_k * sigma_ln_k)
    else:
        final_std_errors['K'] = None

    for p in params_to_fit:
        if p in result.std_errors and result.std_errors[p] is not None:
            final_std_errors[p] = float(result.std_errors[p])

    # Calculate predictions in linear space for metrics
    ln_Nf_pred = cips2008_predict_ln(None, **result.parameters)
    Nf_pred = np.exp(ln_Nf_pred)

    r_squared_linear = float(calculate_r_squared(Nf_observed, Nf_pred))
    rmse_linear = float(calculate_rmse(Nf_observed, Nf_pred))
    residuals_linear = Nf_observed - Nf_pred

    # Create updated result
    from dataclasses import replace
    result = replace(
        result,
        parameters=final_params,
        std_errors=final_std_errors,
        r_squared=r_squared_linear,
        rmse=rmse_linear,
        residuals=residuals_linear,
        confidence_intervals=final_ci
    )

    # Add metadata for fixed params (for display purposes)
    result.fixed_params = {k: float(v) for k, v in fixed_betas.items()}
    result.fixed_data_values = {k: float(v) for k, v in fixed_data_values.items()}
    result.auto_fixed_info = auto_fixed_info

    return result


# ============================================================
# General model fitting (all supported models)
# ============================================================

_kB_eV = 8.617e-5   # Boltzmann constant in eV/K
_R_gas = 8.314       # Gas constant in J/(mol·K)
_eV_per_J_mol = 96485.0  # 1 eV/atom ≈ 96485 J/mol


def fit_general_model(
    model_type: str,
    experiment_data: List[Dict],
    fixed_params: Optional[Dict[str, float]] = None,
) -> FittingResult:
    """Unified fitting dispatcher for all supported lifetime models.

    Args:
        model_type: One of 'coffin_manson', 'coffin_manson_arrhenius',
                    'norris_landzberg', 'lesit', 'cips2008'.
        experiment_data: List of dicts with experiment data.
        fixed_params: Optional fixed parameter values.
    """
    dispatch = {
        'cips2008': lambda: fit_cips2008_model(experiment_data, fixed_params),
        'coffin_manson': lambda: _fit_coffin_manson(experiment_data),
        'coffin_manson_arrhenius': lambda: _fit_coffin_manson_arrhenius(experiment_data),
        'norris_landzberg': lambda: _fit_norris_landzberg(experiment_data, fixed_params),
        'lesit': lambda: _fit_lesit(experiment_data),
    }
    handler = dispatch.get(model_type)
    if handler is None:
        raise FittingError(f"Unknown model type: {model_type}")
    return handler()


def _build_linear_result(
    log_result: FittingResult,
    Nf_obs: np.ndarray,
    Nf_pred: np.ndarray,
    exp_map: Dict[str, str],
    param_order: List[str],
) -> FittingResult:
    """Convert log-space FittingResult to linear-space.

    Args:
        log_result: Result from log-space fitting.
        Nf_obs: Observed Nf (linear).
        Nf_pred: Predicted Nf (linear).
        exp_map: {log_name: linear_name} for params needing exp().
        param_order: Final parameter name list.
    """
    p, ci, se = {}, {}, {}
    inv_map = {v: k for k, v in exp_map.items()}
    for name in param_order:
        if name in inv_map:
            log_name = inv_map[name]
            raw = log_result.parameters[log_name]
            p[name] = float(np.exp(raw))
            if log_name in log_result.confidence_intervals:
                c = log_result.confidence_intervals[log_name]
                ci[name] = (float(np.exp(c[0])), float(np.exp(c[1])))
            if log_name in log_result.std_errors and log_result.std_errors[log_name] is not None:
                se[name] = float(p[name] * log_result.std_errors[log_name])
        elif name in log_result.parameters:
            p[name] = float(log_result.parameters[name])
            if name in log_result.confidence_intervals:
                ci[name] = tuple(float(v) for v in log_result.confidence_intervals[name])
            if name in log_result.std_errors and log_result.std_errors[name] is not None:
                se[name] = float(log_result.std_errors[name])
    return FittingResult(
        parameters=p, std_errors=se,
        r_squared=float(calculate_r_squared(Nf_obs, Nf_pred)),
        rmse=float(calculate_rmse(Nf_obs, Nf_pred)),
        residuals=Nf_obs - Nf_pred,
        covariance=log_result.covariance,
        confidence_intervals=ci,
    )


def _fit_coffin_manson(experiment_data: List[Dict]) -> FittingResult:
    """Fit Coffin-Manson: Nf = A * (ΔTj)^alpha  (alpha < 0)."""
    if not experiment_data or len(experiment_data) < 2:
        raise FittingError("Coffin-Manson fitting requires at least 2 data points")
    try:
        dTj = np.array([d['dTj'] for d in experiment_data], dtype=float)
        Nf = np.array([d['Nf'] for d in experiment_data], dtype=float)
    except KeyError as e:
        raise FittingError(f"Missing field: {e}")
    if np.any(dTj <= 0) or np.any(Nf <= 0):
        raise FittingError("dTj and Nf must be strictly positive")

    ln_dTj = np.log(dTj)
    ln_Nf = np.log(Nf)

    def model(X, ln_A, alpha):
        return ln_A + alpha * ln_dTj

    log_result = fit_lifetime_model(
        model, np.arange(len(Nf)), ln_Nf,
        {'ln_A': float(np.mean(ln_Nf)), 'alpha': -5.0},
    )
    A = float(np.exp(log_result.parameters['ln_A']))
    alpha = float(log_result.parameters['alpha'])
    Nf_pred = A * (dTj ** alpha)
    return _build_linear_result(log_result, Nf, Nf_pred, {'ln_A': 'A'}, ['A', 'alpha'])


def _fit_coffin_manson_arrhenius(experiment_data: List[Dict]) -> FittingResult:
    """Fit CMA: Nf = A * (ΔTj)^alpha * exp(Ea / (kB * Tj_mean_K))."""
    if not experiment_data or len(experiment_data) < 3:
        raise FittingError("CMA fitting requires at least 3 data points")
    try:
        dTj = np.array([d['dTj'] for d in experiment_data], dtype=float)
        Tj_max = np.array([d['Tj_max'] for d in experiment_data], dtype=float)
        Nf = np.array([d['Nf'] for d in experiment_data], dtype=float)
    except KeyError as e:
        raise FittingError(f"Missing field: {e}")
    if np.any(dTj <= 0) or np.any(Nf <= 0):
        raise FittingError("dTj and Nf must be strictly positive")

    Tj_mean_K = Tj_max - dTj / 2.0 + 273.15
    if np.any(Tj_mean_K <= 0):
        raise FittingError("Derived Tj_mean in Kelvin must be positive")

    ln_dTj = np.log(dTj)
    ln_Nf = np.log(Nf)
    inv_Tj = 1.0 / Tj_mean_K

    def model(X, ln_A, alpha, Ea_kB):
        return ln_A + alpha * ln_dTj + Ea_kB * inv_Tj

    log_result = fit_lifetime_model(
        model, np.arange(len(Nf)), ln_Nf,
        {'ln_A': float(np.mean(ln_Nf)), 'alpha': -5.0, 'Ea_kB': 9000.0},
        {'ln_A': (-200, 200), 'alpha': (-15, 0), 'Ea_kB': (0, 50000)},
    )
    A = float(np.exp(log_result.parameters['ln_A']))
    alpha = float(log_result.parameters['alpha'])
    Ea_kB = float(log_result.parameters['Ea_kB'])
    Ea = Ea_kB * _kB_eV  # convert back to eV

    Nf_pred = A * (dTj ** alpha) * np.exp(Ea / (_kB_eV * Tj_mean_K))

    # Build result with unit conversion for Ea
    p = {'A': A, 'alpha': alpha, 'Ea': float(Ea)}
    ci, se = {}, {}
    if 'ln_A' in log_result.confidence_intervals:
        c = log_result.confidence_intervals['ln_A']
        ci['A'] = (float(np.exp(c[0])), float(np.exp(c[1])))
    if 'ln_A' in log_result.std_errors and log_result.std_errors['ln_A'] is not None:
        se['A'] = float(A * log_result.std_errors['ln_A'])
    if 'alpha' in log_result.confidence_intervals:
        ci['alpha'] = tuple(float(v) for v in log_result.confidence_intervals['alpha'])
    if 'alpha' in log_result.std_errors and log_result.std_errors['alpha'] is not None:
        se['alpha'] = float(log_result.std_errors['alpha'])
    if 'Ea_kB' in log_result.confidence_intervals:
        c = log_result.confidence_intervals['Ea_kB']
        ci['Ea'] = (float(c[0] * _kB_eV), float(c[1] * _kB_eV))
    if 'Ea_kB' in log_result.std_errors and log_result.std_errors['Ea_kB'] is not None:
        se['Ea'] = float(log_result.std_errors['Ea_kB'] * _kB_eV)

    return FittingResult(
        parameters=p, std_errors=se,
        r_squared=float(calculate_r_squared(Nf, Nf_pred)),
        rmse=float(calculate_rmse(Nf, Nf_pred)),
        residuals=Nf - Nf_pred, covariance=log_result.covariance,
        confidence_intervals=ci,
    )


def _fit_norris_landzberg(
    experiment_data: List[Dict],
    fixed_params: Optional[Dict[str, float]] = None,
) -> FittingResult:
    """Fit N-L: Nf = A * (ΔTj)^alpha * f^beta * exp(Ea / (kB * Tj_max_K))."""
    if not experiment_data or len(experiment_data) < 3:
        raise FittingError("Norris-Landzberg fitting requires at least 3 data points")
    try:
        dTj = np.array([d['dTj'] for d in experiment_data], dtype=float)
        Tj_max = np.array([d['Tj_max'] for d in experiment_data], dtype=float)
        t_on = np.array([d['t_on'] for d in experiment_data], dtype=float)
        Nf = np.array([d['Nf'] for d in experiment_data], dtype=float)
    except KeyError as e:
        raise FittingError(f"Missing field: {e}")
    if np.any(dTj <= 0) or np.any(Nf <= 0) or np.any(t_on <= 0):
        raise FittingError("dTj, t_on, and Nf must be strictly positive")

    f = 1.0 / (2.0 * t_on)
    Tj_max_K = Tj_max + 273.15
    ln_dTj = np.log(dTj)
    ln_f = np.log(f)
    ln_Nf = np.log(Nf)

    # If frequency is constant, fix beta
    fix_beta = False
    fixed_beta_val = -0.33
    if fixed_params and 'beta' in fixed_params:
        fix_beta = True
        fixed_beta_val = fixed_params['beta']
    elif np.allclose(t_on, t_on[0]):
        fix_beta = True

    auto_info = []
    fixed_info = {}

    if fix_beta:
        def model(X, ln_A, alpha, Ea_kB):
            return ln_A + alpha * ln_dTj + fixed_beta_val * ln_f + Ea_kB / Tj_max_K
        initial = {'ln_A': float(np.mean(ln_Nf)), 'alpha': -5.0, 'Ea_kB': 9000.0}
        bounds = {'ln_A': (-200, 200), 'alpha': (-15, 0), 'Ea_kB': (0, 50000)}
        fixed_info['beta'] = fixed_beta_val
        if not (fixed_params and 'beta' in fixed_params):
            auto_info.append(f"频率f为常值，β自动固定为 {fixed_beta_val}")
    else:
        def model(X, ln_A, alpha, beta, Ea_kB):
            return ln_A + alpha * ln_dTj + beta * ln_f + Ea_kB / Tj_max_K
        initial = {'ln_A': float(np.mean(ln_Nf)), 'alpha': -5.0, 'beta': -0.33, 'Ea_kB': 9000.0}
        bounds = {'ln_A': (-200, 200), 'alpha': (-15, 0), 'beta': (-5, 5), 'Ea_kB': (0, 50000)}

    log_result = fit_lifetime_model(
        model, np.arange(len(Nf)), ln_Nf, initial, bounds,
    )

    A = float(np.exp(log_result.parameters['ln_A']))
    alpha = float(log_result.parameters['alpha'])
    beta = fixed_beta_val if fix_beta else float(log_result.parameters['beta'])
    Ea_kB = float(log_result.parameters['Ea_kB'])
    Ea = Ea_kB * _kB_eV

    Nf_pred = A * (dTj ** alpha) * (f ** beta) * np.exp(Ea / (_kB_eV * Tj_max_K))

    p = {'A': A, 'alpha': alpha, 'beta': float(beta), 'Ea': float(Ea)}
    ci, se = {}, {}
    if 'ln_A' in log_result.confidence_intervals:
        c = log_result.confidence_intervals['ln_A']
        ci['A'] = (float(np.exp(c[0])), float(np.exp(c[1])))
    if 'ln_A' in log_result.std_errors and log_result.std_errors['ln_A'] is not None:
        se['A'] = float(A * log_result.std_errors['ln_A'])
    for pname in ['alpha'] + ([] if fix_beta else ['beta']):
        if pname in log_result.confidence_intervals:
            ci[pname] = tuple(float(v) for v in log_result.confidence_intervals[pname])
        if pname in log_result.std_errors and log_result.std_errors[pname] is not None:
            se[pname] = float(log_result.std_errors[pname])
    if 'Ea_kB' in log_result.confidence_intervals:
        c = log_result.confidence_intervals['Ea_kB']
        ci['Ea'] = (float(c[0] * _kB_eV), float(c[1] * _kB_eV))
    if 'Ea_kB' in log_result.std_errors and log_result.std_errors['Ea_kB'] is not None:
        se['Ea'] = float(log_result.std_errors['Ea_kB'] * _kB_eV)

    out = FittingResult(
        parameters=p, std_errors=se,
        r_squared=float(calculate_r_squared(Nf, Nf_pred)),
        rmse=float(calculate_rmse(Nf, Nf_pred)),
        residuals=Nf - Nf_pred, covariance=log_result.covariance,
        confidence_intervals=ci,
    )
    out.fixed_params = fixed_info
    out.auto_fixed_info = auto_info
    if np.allclose(t_on, t_on[0]):
        out.fixed_data_values = {'ton': float(t_on[0])}
    return out


def _fit_lesit(experiment_data: List[Dict]) -> FittingResult:
    """Fit LESIT: Nf = A * (ΔTj)^alpha * exp(Q / (R * Tj_min_K))."""
    if not experiment_data or len(experiment_data) < 3:
        raise FittingError("LESIT fitting requires at least 3 data points")
    try:
        dTj = np.array([d['dTj'] for d in experiment_data], dtype=float)
        Tj_max = np.array([d['Tj_max'] for d in experiment_data], dtype=float)
        Nf = np.array([d['Nf'] for d in experiment_data], dtype=float)
    except KeyError as e:
        raise FittingError(f"Missing field: {e}")
    if np.any(dTj <= 0) or np.any(Nf <= 0):
        raise FittingError("dTj and Nf must be strictly positive")

    Tj_min_K = Tj_max - dTj + 273.15
    if np.any(Tj_min_K <= 0):
        raise FittingError("Derived Tj_min in Kelvin must be positive")

    ln_dTj = np.log(dTj)
    ln_Nf = np.log(Nf)
    inv_Tj = 1.0 / Tj_min_K

    def model(X, ln_A, alpha, Q_R):
        return ln_A + alpha * ln_dTj + Q_R * inv_Tj

    log_result = fit_lifetime_model(
        model, np.arange(len(Nf)), ln_Nf,
        {'ln_A': float(np.mean(ln_Nf)), 'alpha': -5.0, 'Q_R': 9000.0},
        {'ln_A': (-200, 200), 'alpha': (-15, 0), 'Q_R': (0, 50000)},
    )
    A = float(np.exp(log_result.parameters['ln_A']))
    alpha = float(log_result.parameters['alpha'])
    Q_R = float(log_result.parameters['Q_R'])
    Q_eV = Q_R * _R_gas / _eV_per_J_mol  # Q/R * R / (eV→J/mol) = Q in eV

    Nf_pred = A * (dTj ** alpha) * np.exp(Q_R / Tj_min_K)

    p = {'A': A, 'alpha': alpha, 'Q': float(Q_eV)}
    ci, se = {}, {}
    if 'ln_A' in log_result.confidence_intervals:
        c = log_result.confidence_intervals['ln_A']
        ci['A'] = (float(np.exp(c[0])), float(np.exp(c[1])))
    if 'ln_A' in log_result.std_errors and log_result.std_errors['ln_A'] is not None:
        se['A'] = float(A * log_result.std_errors['ln_A'])
    if 'alpha' in log_result.confidence_intervals:
        ci['alpha'] = tuple(float(v) for v in log_result.confidence_intervals['alpha'])
    if 'alpha' in log_result.std_errors and log_result.std_errors['alpha'] is not None:
        se['alpha'] = float(log_result.std_errors['alpha'])
    q_factor = _R_gas / _eV_per_J_mol
    if 'Q_R' in log_result.confidence_intervals:
        c = log_result.confidence_intervals['Q_R']
        ci['Q'] = (float(c[0] * q_factor), float(c[1] * q_factor))
    if 'Q_R' in log_result.std_errors and log_result.std_errors['Q_R'] is not None:
        se['Q'] = float(log_result.std_errors['Q_R'] * q_factor)

    return FittingResult(
        parameters=p, std_errors=se,
        r_squared=float(calculate_r_squared(Nf, Nf_pred)),
        rmse=float(calculate_rmse(Nf, Nf_pred)),
        residuals=Nf - Nf_pred, covariance=log_result.covariance,
        confidence_intervals=ci,
    )


def weighted_fit(
    model_func: Callable,
    x_data: np.ndarray,
    y_data: np.ndarray,
    y_errors: np.ndarray,
    initial_params: Dict[str, float],
    param_bounds: Optional[Dict[str, Tuple[float, float]]] = None
) -> FittingResult:
    """Perform weighted least squares fitting.

    Weighted fitting is useful when measurement uncertainties vary
    across data points. Each point is weighted by 1/σ² where σ is
    the measurement error.

    Args:
        model_func: Model function f(x, **params) -> y.
        x_data: Independent variable data.
        y_data: Dependent variable data.
        y_errors: Measurement errors (1-sigma) for each y value.
        initial_params: Dictionary of initial parameter guesses.
        param_bounds: Optional parameter bounds.

    Returns:
        FittingResult with weighted fit statistics.
    """
    # Validate
    if len(y_data) != len(y_errors):
        raise FittingError("y_data and y_errors must have same length")

    if np.any(y_errors <= 0):
        raise FittingError("All errors must be positive")

    # Calculate weights
    weights = 1.0 / (y_errors ** 2)

    param_names = list(initial_params.keys())
    p0 = [initial_params[name] for name in param_names]

    # Set up bounds
    if param_bounds:
        lower = [param_bounds.get(name, (-np.inf, np.inf))[0] for name in param_names]
        upper = [param_bounds.get(name, (-np.inf, np.inf))[1] for name in param_names]
    else:
        lower = [-np.inf] * len(param_names)
        upper = [np.inf] * len(param_names)

    def scipy_model(x, *params):
        param_dict = dict(zip(param_names, params))
        return model_func(x, **param_dict)

    try:
        popt, pcov = curve_fit(
            scipy_model,
            x_data,
            y_data,
            p0=p0,
            sigma=y_errors,
            absolute_sigma=True,
            bounds=(lower, upper),
            method='trf'
        )
    except RuntimeError as e:
        raise FittingError(f"Weighted fitting failed: {str(e)}") from e

    y_pred = scipy_model(x_data, *popt)
    residuals = y_data - y_pred

    # Calculate weighted R-squared
    weighted_sum_sq = np.sum(weights * residuals ** 2)
    weighted_total = np.sum(weights * (y_data - np.average(y_data, weights=weights)) ** 2)
    r_squared = 1 - (weighted_sum_sq / weighted_total) if weighted_total > 0 else 0

    rmse = calculate_rmse(y_data, y_pred)

    try:
        perr = np.sqrt(np.diag(pcov))
    except (np.linalg.LinAlgError, ValueError):
        perr = np.full(len(popt), np.nan)

    parameters = dict(zip(param_names, popt))
    std_errors = dict(zip(param_names, perr))

    confidence_intervals = {}
    for name, val, err in zip(param_names, popt, perr):
        if np.isnan(err):
            confidence_intervals[name] = (val, val)
        else:
            z_critical = stats.norm.ppf(0.975)
            ci = (val - z_critical * err, val + z_critical * err)
            confidence_intervals[name] = ci

    return FittingResult(
        parameters=parameters,
        std_errors=std_errors,
        r_squared=r_squared,
        rmse=rmse,
        residuals=residuals,
        covariance=pcov,
        confidence_intervals=confidence_intervals
    )


def robust_fit(
    model_func: Callable,
    x_data: np.ndarray,
    y_data: np.ndarray,
    initial_params: Dict[str, float],
    loss: str = 'soft_l1'
) -> FittingResult:
    """Perform robust fitting using least_squares with outlier rejection.

    Robust fitting is useful when data contains outliers that would
    unduly influence standard least squares. Uses 'soft_l1' loss
    by default which is less sensitive to outliers.

    Args:
        model_func: Model function f(x, **params) -> y.
        x_data: Independent variable data.
        y_data: Dependent variable data.
        initial_params: Dictionary of initial parameter guesses.
        loss: Loss function. Options: 'linear', 'soft_l1', 'huber',
            'cauchy', 'arctan'.

    Returns:
        FittingResult from robust optimization.
    """
    x_data = np.asarray(x_data)
    y_data = np.asarray(y_data)

    param_names = list(initial_params.keys())
    p0 = [initial_params[name] for name in param_names]

    def residuals(params):
        param_dict = dict(zip(param_names, params))
        y_pred = model_func(x_data, **param_dict)
        return y_pred - y_data

    try:
        result = least_squares(
            residuals,
            p0,
            loss=loss,
            method='trf'
        )

        if not result.success:
            raise FittingError(f"Robust fitting failed: {result.message}")

        popt = result.x
        # Approximate covariance from Jacobian
        if result.jac is not None:
            try:
                # Covariance = (J^T J)^-1 * variance
                # Assume unit variance for residuals
                J = result.jac
                pcov = np.linalg.inv(J.T @ J) * (result.cost / len(y_data))
            except np.linalg.LinAlgError:
                pcov = np.eye(len(popt)) * np.inf
        else:
            pcov = np.eye(len(popt)) * np.inf

    except Exception as e:
        raise FittingError(f"Robust fitting failed: {str(e)}") from e

    # Calculate predictions
    param_dict = dict(zip(param_names, popt))
    y_pred = model_func(x_data, **param_dict)
    residuals_final = y_data - y_pred

    r_squared = calculate_r_squared(y_data, y_pred)
    rmse = calculate_rmse(y_data, y_pred)

    try:
        perr = np.sqrt(np.diag(pcov))
    except (np.linalg.LinAlgError, ValueError):
        perr = np.full(len(popt), np.nan)

    parameters = dict(zip(param_names, popt))
    std_errors = dict(zip(param_names, perr))

    confidence_intervals = {}
    for name, val, err in zip(param_names, popt, perr):
        if np.isnan(err):
            confidence_intervals[name] = (val, val)
        else:
            z_critical = stats.norm.ppf(0.975)
            ci = (val - z_critical * err, val + z_critical * err)
            confidence_intervals[name] = ci

    return FittingResult(
        parameters=parameters,
        std_errors=std_errors,
        r_squared=r_squared,
        rmse=rmse,
        residuals=residuals_final,
        covariance=pcov,
        confidence_intervals=confidence_intervals
    )

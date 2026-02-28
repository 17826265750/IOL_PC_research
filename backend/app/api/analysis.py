"""
Analysis endpoints for advanced reliability analysis.

功率模块寿命分析软件 - 可靠性分析API
Author: GSH

Provides endpoints for:
- Weibull reliability analysis
- Model parameter fitting
- Acceleration factor calculation
- Probability plotting
"""
from fastapi import APIRouter, HTTPException, status, Request
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import logging

from app.core.weibull import (
    fit_weibull,
    calculate_b_life,
    calculate_reliability,
    calculate_hazard_rate,
    weibull_probability_plot_data,
    weibull_pdf,
    weibull_cdf,
    WeibullResult
)
from app.core.fitting import (
    fit_lifetime_model,
    fit_cips2008_model,
    fit_general_model,
    FittingResult,
    FittingError
)
from app.core.sensitivity import (
    single_parameter_sensitivity,
    tornado_analysis,
    two_parameter_sensitivity,
    SensitivityResult,
    TornadoData
)
from app.schemas.analysis import (
    WeibullAnalysisRequest,
    WeibullAnalysisResponse,
    AccelerationFactorRequest,
    AccelerationFactorResponse,
    SensitivityAnalysisRequest,
    SensitivityAnalysisResponse,
    AnalysisRequest,
    AnalysisResponse
)


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/analysis", tags=["analysis"])


# ==================== Weibull Analysis Endpoints ====================

@router.post("/weibull/fit")
async def perform_weibull_analysis(request: Request) -> Dict[str, Any]:
    """
    Perform Weibull reliability analysis on failure data.

    Fits Weibull distribution to failure times and calculates:
    - Shape parameter (beta) - indicates failure mode
    - Scale parameter (eta) - characteristic life
    - B10, B50, B63.2 life values
    - MTTF (Mean Time To Failure)

    Methods:
    - mle: Maximum Likelihood Estimation (default)
    - rrx: Rank Regression on X
    - rry: Rank Regression on Y
    """
    try:
        body = await request.json()
        failure_times = body.get("failure_times", [])
        censored_times = body.get("censored_times", [])
        confidence_level = body.get("confidence_level", 0.9)
        method = body.get("method", "mle")

        if len(failure_times) < 2:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="At least 2 failure times required"
            )

        # Fit Weibull distribution
        result = fit_weibull(
            failure_data=failure_times,
            censored_data=censored_times if censored_times else None,
            confidence_level=confidence_level,
            method=method
        )

        # Return comprehensive result matching frontend expectations
        return {
            "shape": result.shape,
            "scale": result.scale,
            "location": result.location,
            "r_squared": result.r_squared,
            "mttf": result.mttf,
            "b10": result.b10_life,
            "b50": result.b50_life,
            "b63_2": result.b63_life,
            "confidence_level": 0.9,
            "shape_ci": {
                "lower": result.confidence_interval[0] if result.confidence_interval else None,
                "upper": result.confidence_interval[1] if result.confidence_interval else None
            } if result.confidence_interval else None,
            "scale_ci": {
                "lower": result.confidence_interval[0] if result.confidence_interval else None,
                "upper": result.confidence_interval[1] if result.confidence_interval else None
            } if result.confidence_interval else None
        }

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error in Weibull analysis: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Weibull analysis failed: {str(e)}"
        )


@router.post("/weibull/b-life")
async def calculate_b_life_endpoint(request: Request) -> Dict[str, Any]:
    """
    Calculate life at given failure probability.

    The B(P) life is the time at which P percent of the population
    will have failed. For example, B10 is the time at 10% failures.
    """
    try:
        body = await request.json()
        shape = body.get("shape")
        scale = body.get("scale")
        percentiles = body.get("percentiles", [])

        if shape is None or scale is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="shape and scale are required"
            )

        if shape <= 0 or scale <= 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Shape and scale must be positive"
            )

        b_lifes = {}
        for p in percentiles:
            if not (0 < p < 100):
                continue
            # Convert percentage to fraction
            fraction = p / 100
            b_life = calculate_b_life(shape, scale, fraction)
            b_lifes[str(p)] = float(b_life)

        return {
            "b_lifes": b_lifes,
            "shape": shape,
            "scale": scale
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error calculating B-life: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"B-life calculation failed: {str(e)}"
        )


@router.post("/weibull/reliability")
async def calculate_reliability_endpoint(request: Request) -> Dict[str, Any]:
    """
    Calculate reliability at given time points.

    Reliability R(t) is the probability that a unit will survive
    beyond time t without failure.
    """
    try:
        body = await request.json()
        shape = body.get("shape")
        scale = body.get("scale")
        times = body.get("times", [])
        location = body.get("location", 0.0)

        if shape is None or scale is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="shape and scale are required"
            )

        if not times:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="times array is required"
            )

        reliabilities = [float(calculate_reliability(shape, scale, t, location)) for t in times]

        return {
            "times": times,
            "reliabilities": reliabilities,
            "shape": shape,
            "scale": scale,
            "location": location
        }

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error calculating reliability: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Reliability calculation failed: {str(e)}"
        )


@router.post("/weibull/hazard-rate")
async def calculate_hazard_endpoint(request: Request) -> Dict[str, Any]:
    """
    Calculate instantaneous hazard rate at given time points.

    The hazard rate h(t) is the instantaneous failure rate given
    survival to time t.
    """
    try:
        body = await request.json()
        shape = body.get("shape")
        scale = body.get("scale")
        times = body.get("times", [])
        location = body.get("location", 0.0)

        if shape is None or scale is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="shape and scale are required"
            )

        if not times:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="times array is required"
            )

        hazard_rates = [float(calculate_hazard_rate(shape, scale, t, location)) for t in times]

        return {
            "times": times,
            "hazard_rates": hazard_rates,
            "shape": shape,
            "scale": scale,
            "location": location
        }

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error calculating hazard rate: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Hazard rate calculation failed: {str(e)}"
        )


@router.post("/weibull/probability-plot")
async def generate_weibull_plot_data(request: Request) -> Dict[str, Any]:
    """
    Generate data for Weibull probability plot.

    Returns x and y values for plotting. If data is Weibull-distributed,
    the points will fall approximately on a straight line.
    """
    try:
        body = await request.json()
        failure_times = body.get("failure_times", [])
        censored_times = body.get("censored_times", [])

        if len(failure_times) < 2:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="At least 2 failure times required"
            )

        x_values, y_values = weibull_probability_plot_data(failure_times)

        # Build points array
        points = []
        for i, t in enumerate(failure_times):
            points.append({
                "time": t,
                "ln_time": float(x_values[i]),
                "median_rank": 1 - np.exp(-np.exp(y_values[i])),  # Convert back from Weibull scale
                "weibull_y": float(y_values[i])
            })

        # Calculate fitted line (linear regression on ln-ln scale)
        # y = β * ln(t) - β * ln(η)
        # Fit line through the data
        x_arr = np.array(x_values)
        y_arr = np.array(y_values)
        slope = np.polyfit(x_arr, y_arr, 1)[0]
        intercept = np.polyfit(x_arr, y_arr, 1)[1]

        fitted_line_x = np.linspace(float(x_arr.min()), float(x_arr.max()), 50)
        fitted_line_y = slope * fitted_line_x + intercept

        return {
            "points": points,
            "fitted_line": {
                "x": fitted_line_x.tolist(),
                "y": fitted_line_y.tolist()
            },
            "failure_times": failure_times,
            "n": len(failure_times)
        }

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error generating plot data: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Plot data generation failed: {str(e)}"
        )


@router.post("/weibull/curve")
async def generate_weibull_curve_endpoint(request: Request) -> Dict[str, Any]:
    """
    Generate Weibull PDF, CDF, reliability, and hazard curve data.

    Returns curve values for the specified time range.
    """
    try:
        body = await request.json()
        shape = body.get("shape")
        scale = body.get("scale")
        t_min = body.get("t_min", 0)
        t_max = body.get("t_max", 1000)
        num_points = body.get("num_points", 100)
        location = body.get("location", 0.0)

        if shape is None or scale is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="shape and scale are required"
            )

        if shape <= 0 or scale <= 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Shape and scale must be positive"
            )

        times = np.linspace(t_min, t_max, num_points)

        pdf_values = weibull_pdf(shape, scale, times, location)
        cdf_values = weibull_cdf(shape, scale, times, location)
        reliability_values = 1 - cdf_values
        hazard_values = np.array([calculate_hazard_rate(shape, scale, t, location) for t in times])

        return {
            "times": times.tolist(),
            "pdf": pdf_values.tolist(),
            "cdf": cdf_values.tolist(),
            "reliability": reliability_values.tolist(),
            "hazard_rate": hazard_values.tolist(),
            "shape": shape,
            "scale": scale,
            "location": location
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating curve: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Curve generation failed: {str(e)}"
        )


# ==================== Fitting Endpoints ====================

@router.post("/fitting/fit-model")
async def fit_model_to_data(
    x_data: List[float],
    y_data: List[float],
    model_type: str = "arrhenius",
    initial_params: Optional[Dict[str, float]] = None,
    param_bounds: Optional[Dict[str, Tuple[float, float]]] = None
) -> Dict[str, Any]:
    """
    Fit lifetime model parameters to experimental data.

    Performs non-linear least squares fitting to estimate model parameters.
    """
    try:
        if len(x_data) != len(y_data):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="x_data and y_data must have same length"
            )

        if len(x_data) < 2:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="At least 2 data points required"
            )

        x_arr = np.array(x_data)
        y_arr = np.array(y_data)

        # Define model functions
        def arrhenius_model(x, A, Ea):
            """Arrhenius model: y = A * exp(-Ea / x)"""
            k_b = 8.617e-5  # Boltzmann constant in eV/K
            return A * np.exp(-Ea / (k_b * x))

        def power_model(x, A, n):
            """Power law model: y = A * x^n"""
            return A * (x ** n)

        def exponential_model(x, A, b):
            """Exponential model: y = A * exp(b * x)"""
            return A * np.exp(b * x)

        # Select model
        if model_type == "arrhenius":
            model_func = arrhenius_model
            if initial_params is None:
                initial_params = {"A": 1e10, "Ea": 1.0}
        elif model_type == "power":
            model_func = power_model
            if initial_params is None:
                initial_params = {"A": 1.0, "n": -2.0}
        elif model_type == "exponential":
            model_func = exponential_model
            if initial_params is None:
                initial_params = {"A": 1.0, "b": -0.01}
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unknown model type: {model_type}"
            )

        # Fit model
        result = fit_lifetime_model(
            model_func,
            x_arr,
            y_arr,
            initial_params,
            param_bounds
        )

        # Calculate predicted values
        def prediction_func(x_val):
            params = result.parameters
            return model_func(x_val, **params)

        y_pred = np.array([prediction_func(x) for x in x_data])

        return {
            "parameters": result.parameters,
            "std_errors": result.std_errors,
            "r_squared": float(result.r_squared),
            "rmse": float(result.rmse),
            "confidence_intervals": {
                k: (float(v[0]), float(v[1])) for k, v in result.confidence_intervals.items()
            },
            "predicted": y_pred.tolist(),
            "residuals": result.residuals.tolist()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fitting model: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Model fitting failed: {str(e)}"
        )


@router.post("/fitting/cips2008")
async def fit_cips_model(request: Request) -> Dict[str, Any]:
    """
    Fit CIPS 2008 model parameters to experimental data.

    Default behavior fits all CIPS 2008 parameters.
    For same-product datasets, V/D effects can be optionally coupled into K_eff.

    Request body:
    - experiment_data: List of experiment data points
    - fixed_params: Optional dict of fixed β values (e.g., {'β3': -0.462})
    - couple_vd_to_k: Optional bool, couple β5/β6 terms into K_eff

    Returns:
    - parameters: Dict with K (original), K_eff (for prediction), and all β values
    - fixed_params: Dict of β parameters that were fixed
    - fixed_data_values: Dict of data values for fixed parameters (ton, I, V, D)
    - auto_fixed_info: List of auto-detected fixed parameters
    """
    try:
        # 获取原始JSON
        body = await request.json()

        # 支持两种格式：直接列表或包含experiment_data的对象
        if isinstance(body, list):
            experiment_data = body
            fixed_params = None
            couple_vd_to_k = False
        else:
            experiment_data = body.get("experiment_data", [])
            fixed_params = body.get("fixed_params")
            couple_vd_to_k = bool(
                body.get("couple_vd_to_k", body.get("couple_vd_into_k", False))
            )

        result = fit_cips2008_model(
            experiment_data,
            fixed_params=fixed_params,
            couple_vd_to_k=couple_vd_to_k,
        )

        response = {
            "parameters": result.parameters,
            "std_errors": result.std_errors,
            "r_squared": float(result.r_squared),
            "rmse": float(result.rmse),
            "confidence_intervals": {
                k: (float(v[0]) if v[0] is not None else None,
                    float(v[1]) if v[1] is not None else None)
                for k, v in result.confidence_intervals.items()
            },
            "fixed_params": getattr(result, 'fixed_params', {}),
            "fixed_data_values": getattr(result, 'fixed_data_values', {}),
            "auto_fixed_info": getattr(result, 'auto_fixed_info', []),
            "couple_vd_to_k": couple_vd_to_k,
        }

        return response

    except Exception as e:
        logger.error(f"Error fitting CIPS 2008 model: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"CIPS 2008 fitting failed: {str(e)}"
        )


@router.post("/fitting/model")
async def fit_model_unified(request: Request) -> Dict[str, Any]:
    """Unified model fitting endpoint for all supported lifetime models.

    Request body:
    - model_type: 'coffin_manson' | 'coffin_manson_arrhenius' | 'norris_landzberg' | 'lesit' | 'cips2008'
    - experiment_data: List of experiment data points
    - fixed_params: Optional dict of fixed parameter values
    """
    try:
        body = await request.json()
        model_type = body.get('model_type', 'cips2008')
        experiment_data = body.get('experiment_data', [])
        fixed_params = body.get('fixed_params')

        result = fit_general_model(model_type, experiment_data, fixed_params)

        response: Dict[str, Any] = {
            "parameters": result.parameters,
            "std_errors": result.std_errors,
            "r_squared": float(result.r_squared),
            "rmse": float(result.rmse),
            "confidence_intervals": {
                k: (
                    float(v[0]) if v[0] is not None else None,
                    float(v[1]) if v[1] is not None else None,
                )
                for k, v in result.confidence_intervals.items()
            },
            "fixed_params": getattr(result, 'fixed_params', None) or {},
            "fixed_data_values": getattr(result, 'fixed_data_values', None) or {},
            "auto_fixed_info": getattr(result, 'auto_fixed_info', None) or [],
        }
        return response

    except FittingError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error(f"Error fitting model: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Model fitting failed: {str(e)}",
        )


# ==================== Acceleration Factor Endpoints ====================

@router.post("/acceleration-factor", response_model=AccelerationFactorResponse)
async def calculate_acceleration_factor(request: AccelerationFactorRequest):
    """
    Calculate acceleration factor for accelerated testing.

    The acceleration factor (AF) relates test time to use time:
    AF = exp[(Ea/k) * (1/T_use - 1/T_test)]

    where:
    - Ea is activation energy (eV)
    - k is Boltzmann constant (8.617e-5 eV/K)
    - T are temperatures in Kelvin
    """
    try:
        k_b = 8.617e-5  # Boltzmann constant in eV/K

        T_test = request.test_temperature + 273.15  # Convert to Kelvin
        T_use = request.use_temperature + 273.15

        if T_test <= 0 or T_use <= 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Temperatures must be above absolute zero"
            )

        # Calculate acceleration factor
        # AF = exp[(Ea/k) * (1/T_use - 1/T_test)]
        exponent = (request.activation_energy / k_b) * (1/T_use - 1/T_test)
        af = np.exp(exponent)

        return AccelerationFactorResponse(
            acceleration_factor=float(af),
            test_time_hours=None,
            equivalent_use_hours=None
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error calculating acceleration factor: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Acceleration factor calculation failed: {str(e)}"
        )


@router.post("/acceleration-factor/convert")
async def convert_test_time(
    test_time_hours: float,
    test_temperature: float,
    use_temperature: float,
    activation_energy: float = 0.7
) -> Dict[str, float]:
    """
    Convert test time to equivalent use time using acceleration factor.

    Returns the equivalent operating hours at use conditions.
    """
    try:
        k_b = 8.617e-5

        T_test = test_temperature + 273.15
        T_use = use_temperature + 273.15

        exponent = (activation_energy / k_b) * (1/T_use - 1/T_test)
        af = np.exp(exponent)

        equivalent_hours = test_time_hours * af

        return {
            "test_time_hours": test_time_hours,
            "test_temperature": test_temperature,
            "use_temperature": use_temperature,
            "activation_energy": activation_energy,
            "acceleration_factor": float(af),
            "equivalent_use_hours": float(equivalent_hours)
        }

    except Exception as e:
        logger.error(f"Error converting test time: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Time conversion failed: {str(e)}"
        )


# ==================== Generic Analysis Endpoint ====================

@router.post("/analyze")
async def perform_analysis(request: AnalysisRequest) -> AnalysisResponse:
    """
    Generic analysis endpoint that routes to specific analysis types.

    Supported analysis types:
    - weibull_fit: Weibull distribution fitting
    - acceleration_factor: Temperature acceleration calculation
    - sensitivity: Parameter sensitivity analysis
    """
    try:
        if request.analysis_type == "weibull_fit":
            failure_times = request.input_data.get("failure_times", [])
            result = fit_weibull(failure_times)

            return AnalysisResponse(
                analysis_type=request.analysis_type,
                results={
                    "shape": result.shape,
                    "scale": result.scale,
                    "b10_life": result.b10_life,
                    "b50_life": result.b50_life,
                    "characteristic_life": result.b63_life,
                    "mtbf": result.mttf
                }
            )

        elif request.analysis_type == "acceleration_factor":
            test_temp = request.input_data.get("test_temperature", 150)
            use_temp = request.input_data.get("use_temperature", 85)
            ea = request.input_data.get("activation_energy", 0.7)

            k_b = 8.617e-5
            T_test = test_temp + 273.15
            T_use = use_temp + 273.15
            exponent = (ea / k_b) * (1/T_use - 1/T_test)
            af = np.exp(exponent)

            return AnalysisResponse(
                analysis_type=request.analysis_type,
                results={
                    "acceleration_factor": float(af),
                    "test_temperature": test_temp,
                    "use_temperature": use_temp,
                    "activation_energy": ea
                }
            )

        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unknown analysis type: {request.analysis_type}"
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in analysis: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Analysis failed: {str(e)}"
        )

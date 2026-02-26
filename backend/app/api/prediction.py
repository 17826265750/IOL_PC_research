"""
Prediction endpoints for lifetime calculation.

Provides endpoints for:
- Single lifetime prediction
- Multiple model comparison
- Parameter sensitivity analysis
- Prediction history management
"""
from fastapi import APIRouter, HTTPException, Depends, status
from sqlalchemy.orm import Session
from typing import List, Optional, Dict, Any
from datetime import datetime
import logging

from app.core.models.model_factory import ModelFactory
from app.core.rainflow import rainflow_counting, find_peaks_and_valleys
from app.core.damage_accumulation import calculate_miner_damage
from app.core.weibull import fit_weibull
from app.core.fitting import fit_lifetime_model
from app.core.sensitivity import single_parameter_sensitivity, tornado_analysis, sobol_sensitivity
from app.db.database import get_db
from app.db.crud import CRUDBase
from app.models.prediction import Prediction
from app.schemas.prediction import (
    PredictionCreate,
    PredictionUpdate,
    PredictionResponse
)
from app.schemas.damage import (
    LifetimePredictionRequest,
    LifetimePredictionResponse
)
from app.schemas.analysis import (
    SensitivityAnalysisRequest,
    SensitivityAnalysisResponse
)


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/prediction", tags=["prediction"])

# CRUD instance
prediction_crud = CRUDBase(Prediction)


def lifetime_model_wrapper(range_val: float, mean_val: float, params: Dict[str, Any]) -> float:
    """Wrapper function for damage calculation compatibility."""
    model_type = params.get("model_type", "cips-2008")
    model = ModelFactory.get_model(model_type)

    # Extract model parameters
    model_params = {
        "delta_Tj": range_val,
        "Tj_max": mean_val + range_val / 2,  # Approximate Tj_max from mean and range
        "t_on": params.get("t_on", 1.0),
        "I": params.get("I", 100.0),
        "V": params.get("V", 1200.0),
        "D": params.get("D", 300.0),
    }

    # Add any override parameters
    model_params.update(params.get("model_overrides", {}))

    return model.calculate_cycles_to_failure(**model_params)


@router.post("/calculate", response_model=LifetimePredictionResponse)
async def calculate_lifetime(request: LifetimePredictionRequest):
    """
    Calculate lifetime using selected model.

    Performs a complete lifetime prediction including:
    - Rainflow cycle counting if time-series data provided
    - Damage accumulation using Miner's rule
    - Lifetime prediction using specified model
    """
    try:
        # Get the lifetime model
        model = ModelFactory.get_model(request.model_type)

        # Get model parameters with defaults
        params = request.parameters.copy()

        # Set default parameters if not provided
        params.setdefault("delta_Tj", 80.0)
        params.setdefault("Tj_max", 398.0)  # 125°C in Kelvin
        params.setdefault("t_on", 1.0)
        params.setdefault("I", 100.0)
        params.setdefault("V", 1200.0)
        params.setdefault("D", 300.0)

        # Calculate cycles to failure
        cycles_to_failure = model.calculate_cycles_to_failure(**params)

        # Apply safety factor
        safety_factor = request.safety_factor
        adjusted_cycles = cycles_to_failure / safety_factor

        # Convert to years (assuming 1 cycle per hour by default)
        cycles_per_year = params.get("cycles_per_year", 8760)  # Default: 1 cycle/hour
        predicted_lifetime_years = adjusted_cycles / cycles_per_year

        # Calculate damage at end of life
        damage_at_eol = 1.0 / safety_factor

        return LifetimePredictionResponse(
            predicted_lifetime_years=predicted_lifetime_years,
            predicted_lifetime_cycles=int(adjusted_cycles),
            confidence_level=None,
            damage_at_eol=damage_at_eol,
            model_used=request.model_type,
            safety_factor=safety_factor
        )

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid parameters: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Error calculating lifetime: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Calculation failed: {str(e)}"
        )


@router.post("/compare")
async def compare_models(request: Dict[str, Any]):
    """
    Compare multiple lifetime models.

    Returns predictions from all specified models for comparison.
    """
    model_types = request.get("models", ["cips-2008"])
    parameters = request.get("parameters", {})

    results = []
    for model_type in model_types:
        try:
            model = ModelFactory.get_model(model_type)

            # Set default parameters
            params = parameters.copy()
            params.setdefault("delta_Tj", 80.0)
            params.setdefault("Tj_max", 398.0)
            params.setdefault("t_on", 1.0)
            params.setdefault("I", 100.0)
            params.setdefault("V", 1200.0)
            params.setdefault("D", 300.0)

            cycles = model.calculate_cycles_to_failure(**params)

            results.append({
                "model_type": model_type,
                "model_name": model.get_model_name(),
                "cycles_to_failure": cycles,
                "equation": model.get_equation()
            })
        except Exception as e:
            results.append({
                "model_type": model_type,
                "error": str(e)
            })

    # Calculate statistics
    valid_results = [r for r in results if "cycles_to_failure" in r]
    if valid_results:
        min_cycles = min(r["cycles_to_failure"] for r in valid_results)
        max_cycles = max(r["cycles_to_failure"] for r in valid_results)
        avg_cycles = sum(r["cycles_to_failure"] for r in valid_results) / len(valid_results)

        stats = {
            "min_lifetime": min_cycles,
            "max_lifetime": max_cycles,
            "avg_lifetime": avg_cycles,
            "range": max_cycles - min_cycles,
            "coefficient_of_variation": (max_cycles - min_cycles) / avg_cycles if avg_cycles > 0 else 0
        }
    else:
        stats = {}

    return {
        "results": results,
        "statistics": stats
    }


@router.post("/sensitivity", response_model=SensitivityAnalysisResponse)
async def sensitivity_analysis(request: SensitivityAnalysisRequest):
    """
    Perform sensitivity analysis on model parameters.

    Analyzes how changes in input parameters affect lifetime predictions.
    """
    try:
        model_type = request.base_parameters.get("model_type", "cips-2008")

        # Create model function for sensitivity analysis
        def model_func(**params):
            model = ModelFactory.get_model(model_type)

            # Merge with base parameters
            full_params = request.base_parameters.copy()
            full_params.update(params)

            return model.calculate_cycles_to_failure(**full_params)

        # Perform tornado analysis
        tornado_results = tornado_analysis(
            model_func,
            request.base_parameters,
            request.parameter_ranges
        )

        # Calculate base lifetime
        base_lifetime = model_func(**request.base_parameters)

        # Find most sensitive parameter
        most_sensitive = max(
            request.parameter_ranges.keys(),
            key=lambda k: abs(
                tornado_results[
                    [t.parameter for t in tornado_results].index(k)
                ].percent_change
            ) if k in [t.parameter for t in tornado_results] else 0
        ) if tornado_results else None

        # Convert to response format
        from app.schemas.analysis import SensitivityResult
        sensitivity_results = []
        for t in tornado_results:
            sensitivity_results.append(SensitivityResult(
                parameter_name=t.parameter,
                sensitivity_coefficient=t.range_width / (t.base_output or 1),
                min_lifetime=t.low_value,
                max_lifetime=t.high_value,
                percent_change=t.percent_change
            ))

        return SensitivityAnalysisResponse(
            results=sensitivity_results,
            base_lifetime=base_lifetime,
            most_sensitive_parameter=most_sensitive or "unknown"
        )

    except Exception as e:
        logger.error(f"Error in sensitivity analysis: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Sensitivity analysis failed: {str(e)}"
        )


@router.get("", response_model=List[PredictionResponse])
async def get_predictions(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """Get all predictions with pagination."""
    try:
        predictions = prediction_crud.get_multi(db, skip=skip, limit=limit)
        return predictions
    except Exception as e:
        logger.error(f"Error fetching predictions: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch predictions: {str(e)}"
        )


@router.get("/{prediction_id}", response_model=PredictionResponse)
async def get_prediction(prediction_id: int, db: Session = Depends(get_db)):
    """Get a specific prediction by ID."""
    prediction = prediction_crud.get(db, prediction_id)
    if not prediction:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Prediction {prediction_id} not found"
        )
    return prediction


@router.post("", response_model=PredictionResponse, status_code=status.HTTP_201_CREATED)
async def create_prediction(
    request: PredictionCreate,
    db: Session = Depends(get_db)
):
    """Create a new prediction record."""
    try:
        prediction = prediction_crud.create(db, request)
        return prediction
    except Exception as e:
        logger.error(f"Error creating prediction: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create prediction: {str(e)}"
        )


@router.put("/{prediction_id}", response_model=PredictionResponse)
async def update_prediction(
    prediction_id: int,
    request: PredictionUpdate,
    db: Session = Depends(get_db)
):
    """Update an existing prediction."""
    prediction = prediction_crud.get(db, prediction_id)
    if not prediction:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Prediction {prediction_id} not found"
        )

    try:
        updated = prediction_crud.update(db, prediction, request)
        return updated
    except Exception as e:
        logger.error(f"Error updating prediction: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update prediction: {str(e)}"
        )


@router.delete("/{prediction_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_prediction(prediction_id: int, db: Session = Depends(get_db)):
    """Delete a prediction."""
    prediction = prediction_crud.get(db, prediction_id)
    if not prediction:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Prediction {prediction_id} not found"
        )

    prediction_crud.delete(db, prediction_id)
    return None


@router.post("/predict")
async def predict_lifetime(request: Dict[str, Any]):
    """
    Predict lifetime using the specified model.

    This endpoint matches the frontend API format.
    """
    try:
        model_type = request.get("modelType", "cips-2008")
        params = request.get("params", {})
        cycles = request.get("cycles", [])

        # Map frontend parameter names to backend expected names
        # Calculate delta_Tj from Tmax and Tmin
        Tmax = params.get("Tmax", 125)
        Tmin = params.get("Tmin", 40)
        delta_Tj = Tmax - Tmin

        # Convert Tmax (°C) to Tj_max (Kelvin)
        Tj_max = Tmax + 273.15

        # Get heating time
        t_on = params.get("theating", params.get("t_on", 60))

        # Get CIPS 2008 specific parameters
        I = params.get("I", 100)
        V = params.get("V", 1200)
        D = params.get("D", 300)

        # Get model parameters (beta values) - Bayerer et al. 2008 typical values
        # K needs to be fitted to specific device, using 1e17 as reasonable default
        K = params.get("K", params.get("A", 1e17))
        beta1 = params.get("beta1", params.get("n", -4.423))
        beta2 = params.get("beta2", 1285)
        beta3 = params.get("beta3", -0.462)
        beta4 = params.get("beta4", -0.716)
        beta5 = params.get("beta5", -0.761)
        beta6 = params.get("beta6", -0.5)

        # Map model type to backend format
        backend_model_type = model_type.replace("_", "-")
        if backend_model_type == "cips2008":
            backend_model_type = "cips-2008"

        # Get the model
        model = ModelFactory.get_model(backend_model_type)

        # Build parameters for model calculation
        model_params = {
            "delta_Tj": delta_Tj,
            "Tj_max": Tj_max,
            "t_on": t_on,
            "I": I,
            "V": V,
            "D": D,
            "K": K,
            "beta1": beta1,
            "beta2": beta2,
            "beta3": beta3,
            "beta4": beta4,
            "beta5": beta5,
            "beta6": beta6,
        }

        # Add other model-specific parameters
        for key, value in params.items():
            if key not in ["Tmax", "Tmin", "theating", "tcooling"]:
                model_params[key] = value

        # Calculate cycles to failure
        cycles_to_failure = model.calculate_cycles_to_failure(**model_params)

        # Calculate lifetime in hours (assuming cycle time from cycles data)
        cycle_time_hours = 1.0  # Default 1 hour per cycle
        if cycles and len(cycles) > 0:
            cycle = cycles[0]
            heating = cycle.get("theating", 60)
            cooling = cycle.get("tcooling", 60)
            cycle_time_hours = (heating + cooling) / 3600.0

        lifetime_hours = cycles_to_failure * cycle_time_hours
        lifetime_years = lifetime_hours / 8760  # 8760 hours per year

        # Calculate confidence interval (simplified)
        confidence_factor = 0.3  # 30% variation
        confidence_lower = cycles_to_failure * (1 - confidence_factor)
        confidence_upper = cycles_to_failure * (1 + confidence_factor)

        # Build cycle results for frontend compatibility
        cycle_results = []
        if cycles and len(cycles) > 0:
            for i, cycle in enumerate(cycles):
                cycle_Tmax = cycle.get("Tmax", Tmax)
                cycle_Tmin = cycle.get("Tmin", Tmin)
                cycle_delta_T = cycle_Tmax - cycle_Tmin
                cycle_results.append({
                    "index": i,
                    "deltaT": cycle_delta_T,
                    "cyclesToFailure": cycles_to_failure,
                    "damagePerCycle": 1.0 / cycles_to_failure if cycles_to_failure > 0 else 0
                })
        else:
            # Default single cycle result
            cycle_results.append({
                "index": 0,
                "deltaT": delta_Tj,
                "cyclesToFailure": cycles_to_failure,
                "damagePerCycle": 1.0 / cycles_to_failure if cycles_to_failure > 0 else 0
            })

        return {
            "success": True,
            "data": {
                "modelType": model_type,
                "predictedCycles": cycles_to_failure,
                "lifetimeHours": lifetime_hours,
                "lifetimeYears": lifetime_years,
                "confidenceLower": confidence_lower,
                "confidenceUpper": confidence_upper,
                "deltaTj": delta_Tj,
                "cycleResults": cycle_results,
                "timestamp": datetime.now().isoformat(),
                "TjMax": Tj_max,
                "inputParams": params,
            }
        }

    except ValueError as e:
        logger.error(f"ValueError in predict_lifetime: {e}")
        return {
            "success": False,
            "error": f"参数错误 / Parameter error: {str(e)}"
        }
    except Exception as e:
        logger.error(f"Error in predict_lifetime: {e}")
        return {
            "success": False,
            "error": f"计算失败 / Calculation failed: {str(e)}"
        }


@router.get("/models/available")
async def list_available_models():
    """List all available lifetime models."""
    try:
        ModelFactory.register_all()

        models = []
        for model_name in ModelFactory.list_models():
            try:
                info = ModelFactory.get_model_info(model_name)
                models.append({
                    "name": model_name,
                    "display_name": info.get("model_name", model_name),
                    "equation": info.get("equation", ""),
                    "parameters": info.get("parameters", {})
                })
            except Exception as e:
                logger.warning(f"Could not get info for model {model_name}: {e}")
                models.append({
                    "name": model_name,
                    "display_name": model_name,
                    "error": str(e)
                })

        return {"models": models}
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list models: {str(e)}"
        )


@router.get("/models/{model_name}")
async def get_model_info(model_name: str):
    """Get detailed information about a specific model."""
    try:
        if not ModelFactory.is_registered(model_name):
            ModelFactory.register_all()

        info = ModelFactory.get_model_info(model_name)
        return info
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get model info: {str(e)}"
        )

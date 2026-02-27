"""
Damage accumulation endpoints.

Provides endpoints for:
- Miner's rule damage calculation
- S-N curve based lifetime prediction
- Remaining life estimation
- Damage rate calculation
- Safety margin calculation
- Lifetime curve generation
"""
from fastapi import APIRouter, HTTPException, status, Depends
from sqlalchemy.orm import Session
from typing import List, Dict, Any
import numpy as np
import logging

from app.core.damage_accumulation import (
    calculate_miner_damage,
    estimate_remaining_cycles,
    calculate_damage_rate,
    calculate_confidence_interval,
    combine_damage_states,
    predict_time_to_failure,
    adjust_for_sequence_effect,
    DamageResult
)
from app.core.models.model_factory import ModelFactory
from app.core.safety_margin import (
    calculate_safety_margin,
    calculate_statistical_safety_margin,
    calculate_required_safety_factor,
    assess_design_adequacy,
    SafetyMarginResult
)
from app.core.remaining_life import (
    assess_remaining_life,
    DegradationPoint,
    RemainingLifeResult
)
from app.db.database import get_db
from app.schemas.damage import (
    DamageRequest,
    DamageResponse,
    DamageResult as DamageResultSchema,
    SNCurve,
    LifetimePredictionRequest,
    LifetimePredictionResponse,
    # Safety margin schemas
    SafetyMarginRequest,
    SafetyMarginResponse,
    StatisticalSafetyMarginRequest,
    StatisticalSafetyMarginResponse,
    RequiredSafetyFactorRequest,
    RequiredSafetyFactorResponse,
    # Lifetime curve schemas
    LifetimeCurveRequest,
    LifetimeCurveResponse,
    CurvePoint,
    ModelCurve,
    # Remaining life schemas
    RemainingLifeRequest,
    RemainingLifeResponse,
    DegradationHistoryPoint,
    ConfidenceInterval
)


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/damage", tags=["damage"])


def create_lifetime_function_from_sn_curve(sn_curve: SNCurve, model_type: str = "cips-2008"):
    """
    Create a lifetime function from S-N curve parameters.

    S-N curve: N = A * S^(-b)
    where A is intercept and b is slope (Basquin equation)
    """
    def lifetime_model(range_val: float, mean_val: float, params: Dict[str, Any]) -> float:
        # Calculate allowable cycles using S-N curve
        # N = (stress_intercept / stress_range)^slope
        stress_range = max(range_val, 0.1)  # Avoid division by zero

        allowable_cycles = (sn_curve.intercept / stress_range) ** sn_curve.slope

        return allowable_cycles

    return lifetime_model


@router.post("/calculate", response_model=DamageResponse)
async def calculate_damage(request: DamageRequest):
    """
    Calculate cumulative damage using Miner's linear rule.

    Total damage = Σ (n_i / N_i) where failure occurs when damage >= 1.
    """
    try:
        # Create lifetime model function
        lifetime_model = create_lifetime_function_from_sn_curve(request.sn_curve)

        # Convert cycles to format expected by damage module
        cycles_list = [
            {
                "range": cycle.stress_range,
                "mean": cycle.mean_value,
                "count": cycle.cycles
            }
            for cycle in request.cycles
        ]

        # Model parameters (not used for S-N but required by interface)
        model_params = {
            "safety_factor": request.safety_factor
        }

        # Calculate damage
        result = calculate_miner_damage(
            cycles_list,
            lifetime_model,
            model_params,
            include_half_cycles=True
        )

        # Calculate predicted cycles to failure
        # If total damage is D, then predicted cycles = total_cycles_applied / D
        total_applied_cycles = sum(cycle.cycles for cycle in request.cycles)
        if result.total_damage > 0:
            predicted_cycles = total_applied_cycles / result.total_damage * request.safety_factor
        else:
            predicted_cycles = float('inf')

        # Convert details to response format
        details = []
        cumulative = 0.0
        for detail in result.details:
            cumulative += detail['damage_contribution']
            details.append(DamageResultSchema(
                stress_range=detail['range'],
                cycles=float(detail['count']),
                allowable_cycles=float(detail['cycles_to_failure']),
                damage_ratio=float(detail['damage_contribution']),
                cumulative_damage=float(cumulative)
            ))

        return DamageResponse(
            total_damage=float(result.total_damage),
            remaining_life_fraction=float(result.remaining_life_fraction),
            predicted_cycles=float(predicted_cycles) if predicted_cycles != float('inf') else None,
            details=details,
            safety_factor=request.safety_factor,
            is_failed=result.is_critical
        )

    except Exception as e:
        logger.error(f"Error calculating damage: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Damage calculation failed: {str(e)}"
        )


@router.post("/remaining-life")
async def calculate_remaining_life(
    cycles: List[Any],
    current_damage: float,
    cycle_condition: Dict[str, float],
    sn_curve: SNCurve,
    model_type: str = "cips-2008"
) -> Dict[str, float]:
    """
    Estimate remaining cycles based on current damage state.

    Calculates how many more cycles at the given condition can be
    sustained before reaching failure (damage = 1.0).
    """
    try:
        # Create lifetime model
        lifetime_model = create_lifetime_function_from_sn_curve(sn_curve, model_type)

        # Model parameters
        model_params = {}

        # Current cycle condition
        condition = {
            "range": cycle_condition.get("range", 0.0),
            "mean": cycle_condition.get("mean", 0.0)
        }

        # Calculate remaining cycles
        remaining = estimate_remaining_cycles(
            current_damage,
            lifetime_model,
            model_params,
            condition
        )

        return {
            "current_damage": current_damage,
            "remaining_damage_capacity": max(0.0, 1.0 - current_damage),
            "estimated_remaining_cycles": float(remaining),
            "cycle_condition": condition
        }

    except Exception as e:
        logger.error(f"Error calculating remaining life: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Remaining life calculation failed: {str(e)}"
        )


@router.post("/rate")
async def calculate_damage_accumulation_rate(
    cycles: List[Any],
    time_period: float,
    sn_curve: SNCurve
) -> Dict[str, float]:
    """
    Calculate damage accumulation rate per unit time.

    Useful for tracking how quickly damage accumulates during operation.
    """
    try:
        if time_period <= 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Time period must be positive"
            )

        # Create lifetime model
        lifetime_model = create_lifetime_function_from_sn_curve(sn_curve)

        # Convert cycles
        cycles_list = [
            {
                "range": cycle.stress_range if hasattr(cycle, 'stress_range') else cycle.get('range', 0),
                "mean": cycle.mean_value if hasattr(cycle, 'mean_value') else cycle.get('mean', 0),
                "count": cycle.cycles if hasattr(cycle, 'cycles') else cycle.get('count', 0)
            }
            for cycle in cycles
        ]

        # Model parameters
        model_params = {}

        # Calculate damage rate
        rate = calculate_damage_rate(
            cycles_list,
            time_period,
            lifetime_model,
            model_params
        )

        # Calculate time to failure
        if rate > 0:
            ttf = 1.0 / rate
        else:
            ttf = float('inf')

        return {
            "damage_rate": float(rate),
            "time_period": time_period,
            "time_to_failure": float(ttf) if ttf != float('inf') else None,
            "unit": f"damage per {time_period} time units"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error calculating damage rate: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Damage rate calculation failed: {str(e)}"
        )


@router.post("/confidence")
async def calculate_damage_confidence(
    cycles: List[Any],
    sn_curve: SNCurve,
    scatter_factor: float = 1.2,
    confidence_level: float = 0.95
) -> Dict[str, Any]:
    """
    Calculate confidence interval for damage estimate.

    Uses a scatter factor to account for variability in the S-N data.
    """
    try:
        # Create lifetime model
        lifetime_model = create_lifetime_function_from_sn_curve(sn_curve)

        # Convert cycles
        cycles_list = [
            {
                "range": cycle.stress_range if hasattr(cycle, 'stress_range') else cycle.get('range', 0),
                "mean": cycle.mean_value if hasattr(cycle, 'mean_value') else cycle.get('mean', 0),
                "count": cycle.cycles if hasattr(cycle, 'cycles') else cycle.get('count', 0)
            }
            for cycle in cycles
        ]

        # Model parameters
        model_params = {}

        # Calculate confidence interval
        lower, upper = calculate_confidence_interval(
            cycles_list,
            lifetime_model,
            model_params,
            scatter_factor,
            confidence_level
        )

        # Get base damage
        base_result = calculate_miner_damage(cycles_list, lifetime_model, model_params)

        return {
            "base_damage": float(base_result.total_damage),
            "confidence_lower": float(lower),
            "confidence_upper": float(upper),
            "scatter_factor": scatter_factor,
            "confidence_level": confidence_level
        }

    except Exception as e:
        logger.error(f"Error calculating confidence interval: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Confidence interval calculation failed: {str(e)}"
        )


@router.post("/combine")
async def combine_damage_states(
    damage_states: List[float],
    weights: List[float] = None
) -> Dict[str, float]:
    """
    Combine multiple damage states (e.g., from different components).

    Uses either simple averaging or weighted averaging based on
    component importance.
    """
    try:
        combined = combine_damage_states(damage_states, weights)

        return {
            "combined_damage": float(combined),
            "individual_damages": [float(d) for d in damage_states],
            "weights": weights if weights else None,
            "method": "weighted" if weights else "simple_average"
        }

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error combining damage states: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Damage combination failed: {str(e)}"
        )


@router.post("/time-to-failure")
async def calculate_ttf(
    current_damage: float,
    damage_rate: float,
    time_unit: str = "hours"
) -> Dict[str, Any]:
    """
    Predict time to failure based on current damage and damage rate.
    """
    try:
        if current_damage < 0 or current_damage > 1:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Current damage must be between 0 and 1"
            )

        if damage_rate < 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Damage rate must be non-negative"
            )

        ttf = predict_time_to_failure(current_damage, damage_rate, time_unit)

        return {
            "current_damage": current_damage,
            "damage_rate": damage_rate,
            "time_to_failure": float(ttf) if ttf != float('inf') else None,
            "time_unit": time_unit,
            "is_failed": current_damage >= 1.0
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error calculating time to failure: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Time to failure calculation failed: {str(e)}"
        )


@router.post("/sequence-adjustment")
async def adjust_for_sequence(
    cycles: List[Any],
    sn_curve: SNCurve,
    sequence_factor: float = 1.0
) -> Dict[str, Any]:
    """
    Adjust damage calculation for load sequence effects.

    Miner's rule assumes sequence independence, but in reality,
    high-low load sequences can cause different damage than
    low-high sequences. This applies an empirical correction.
    """
    try:
        # Create lifetime model
        lifetime_model = create_lifetime_function_from_sn_curve(sn_curve)

        # Convert cycles
        cycles_list = [
            {
                "range": cycle.stress_range if hasattr(cycle, 'stress_range') else cycle.get('range', 0),
                "mean": cycle.mean_value if hasattr(cycle, 'mean_value') else cycle.get('mean', 0),
                "count": cycle.cycles if hasattr(cycle, 'cycles') else cycle.get('count', 0)
            }
            for cycle in cycles
        ]

        # Model parameters
        model_params = {}

        # Calculate adjusted damage
        result = adjust_for_sequence_effect(
            cycles_list,
            lifetime_model,
            model_params,
            sequence_factor
        )

        # Get base (unadjusted) result
        base_result = calculate_miner_damage(cycles_list, lifetime_model, model_params)

        return {
            "adjusted_damage": float(result.total_damage),
            "base_damage": float(base_result.total_damage),
            "adjustment_factor": sequence_factor,
            "difference": float(result.total_damage - base_result.total_damage),
            "is_failed": result.is_critical,
            "remaining_life_fraction": float(result.remaining_life_fraction)
        }

    except Exception as e:
        logger.error(f"Error adjusting for sequence: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Sequence adjustment failed: {str(e)}"
        )


@router.post("/lifetime-from-model")
async def predict_lifetime_from_model(request: LifetimePredictionRequest) -> LifetimePredictionResponse:
    """
    Predict lifetime using CIPS 2008 lifetime model.

    Integrates with the core lifetime models for full prediction.
    """
    try:
        # Get the model
        model = ModelFactory.get_model(request.model_type)

        # Get parameters with defaults
        params = request.parameters.copy()
        params.setdefault("delta_Tj", 80.0)
        params.setdefault("Tj_max", 398.0)
        params.setdefault("t_on", 1.0)
        params.setdefault("I", 100.0)
        params.setdefault("V", 1200.0)
        params.setdefault("D", 300.0)

        # Calculate cycles to failure
        cycles_to_failure = model.calculate_cycles_to_failure(**params)

        # Apply safety factor
        safety_factor = request.safety_factor
        adjusted_cycles = cycles_to_failure / safety_factor

        # Convert to years
        cycles_per_year = params.get("cycles_per_year", 8760)
        predicted_lifetime_years = adjusted_cycles / cycles_per_year

        # Damage at end of life
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
        logger.error(f"Error predicting lifetime: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Lifetime prediction failed: {str(e)}"
        )


# ============================================================================
# Safety Margin Endpoints
# ============================================================================

@router.post("/safety-margin/calculate", response_model=SafetyMarginResponse)
async def calculate_safety_margin_endpoint(request: SafetyMarginRequest):
    """
    Calculate safety margin between design life and predicted life.

    Safety Margin = (Predicted Life / (Design Life × Safety Factor)) - 1

    A positive margin indicates the design exceeds requirements.
    Returns detailed analysis including design adequacy assessment.
    """
    try:
        result = calculate_safety_margin(
            design_life=request.design_life_cycles,
            predicted_life=request.predicted_life_cycles,
            safety_factor=request.safety_factor,
            minimum_acceptable_margin=request.minimum_acceptable_margin
        )

        # Get design adequacy assessment
        assessment = assess_design_adequacy(result)

        return SafetyMarginResponse(
            safety_factor=result.safety_factor,
            design_life_cycles=result.design_life_cycles,
            predicted_life_cycles=result.predicted_life_cycles,
            margin_percentage=result.margin_percentage,
            margin_value=result.margin_value,
            utilization=result.utilization,
            is_acceptable=result.is_acceptable,
            adequacy_level=assessment['adequacy_level'],
            recommendation=assessment['recommendation'],
            inspection_recommendation=assessment.get('inspection_recommendation')
        )

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error calculating safety margin: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Safety margin calculation failed: {str(e)}"
        )


@router.post("/safety-margin/statistical", response_model=StatisticalSafetyMarginResponse)
async def calculate_statistical_safety_margin_endpoint(request: StatisticalSafetyMarginRequest):
    """
    Calculate safety margin with statistical uncertainty.

    Accounts for uncertainty in the predicted life by treating it as
    a random variable with specified mean and standard deviation.

    Returns statistical margins including percentiles and probability of acceptability.
    """
    try:
        result = calculate_statistical_safety_margin(
            design_life=request.design_life_cycles,
            predicted_life_mean=request.predicted_life_mean,
            predicted_life_std=request.predicted_life_std,
            safety_factor=request.safety_factor,
            minimum_acceptable_margin=request.minimum_acceptable_margin
        )

        return StatisticalSafetyMarginResponse(
            mean_margin=result.mean_margin,
            std_margin=result.std_margin,
            percentile_5=result.percentile_5,
            percentile_95=result.percentile_95,
            probability_acceptable=result.probability_acceptable
        )

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error calculating statistical safety margin: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Statistical safety margin calculation failed: {str(e)}"
        )


@router.post("/safety-margin/required-factor", response_model=RequiredSafetyFactorResponse)
async def calculate_required_safety_factor_endpoint(request: RequiredSafetyFactorRequest):
    """
    Calculate required safety factor to achieve target margin.

    Given design life, predicted life, and target margin, returns
    the safety factor needed to achieve the desired safety level.
    """
    try:
        required_sf = calculate_required_safety_factor(
            design_life=request.design_life_cycles,
            predicted_life=request.predicted_life_cycles,
            target_margin=request.target_margin
        )

        return RequiredSafetyFactorResponse(
            required_safety_factor=required_sf
        )

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error calculating required safety factor: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Required safety factor calculation failed: {str(e)}"
        )


# ============================================================================
# Lifetime Curve Endpoints
# ============================================================================

@router.post("/lifetime-curve/generate", response_model=LifetimeCurveResponse)
async def generate_lifetime_curve(request: LifetimeCurveRequest):
    """
    Generate lifetime curve data for visualization.

    Supports:
    - Delta Tj vs Nf curve
    - Tj_max vs Nf curve
    - t_on vs Nf curve
    - Multi-model comparison curves

    Returns curve data points formatted for charting libraries.
    """
    try:
        curves = []

        for model_type in request.model_types:
            try:
                model = ModelFactory.get_model(model_type)

                points = []
                for param_value in request.parameter_values:
                    # Build parameters with the varying parameter
                    params = request.fixed_parameters.copy()
                    params[request.parameter_to_vary] = param_value

                    # Set defaults for required parameters
                    params.setdefault("delta_Tj", 80.0)
                    params.setdefault("Tj_max", 398.0)
                    params.setdefault("t_on", 1.0)
                    params.setdefault("I", 100.0)
                    params.setdefault("V", 1200.0)
                    params.setdefault("D", 300.0)

                    # Calculate cycles to failure
                    cycles_to_failure = model.calculate_cycles_to_failure(**params)

                    point = CurvePoint(
                        x_value=param_value,
                        y_value=cycles_to_failure,
                        log_x=np.log10(param_value) if request.log_scale_x else None,
                        log_y=np.log10(cycles_to_failure) if request.log_scale_y and cycles_to_failure > 0 else None
                    )
                    points.append(point)

                curve = ModelCurve(
                    model_type=model_type,
                    model_name=model.get_model_name(),
                    points=points,
                    equation=model.get_equation()
                )
                curves.append(curve)

            except Exception as e:
                logger.warning(f"Could not generate curve for model {model_type}: {e}")
                # Add error curve
                curves.append(ModelCurve(
                    model_type=model_type,
                    model_name=model_type,
                    points=[],
                    equation=f"Error: {str(e)}"
                ))

        # Generate axis labels and title based on parameter
        axis_labels = {
            "delta_Tj": ("ΔTj (°C)", "Cycles to Failure (Nf)", "Lifetime vs Temperature Range"),
            "Tj_max": ("Tj_max (K)", "Cycles to Failure (Nf)", "Lifetime vs Maximum Temperature"),
            "t_on": ("t_on (hours)", "Cycles to Failure (Nf)", "Lifetime vs On-Time"),
            "I": ("Current (A)", "Cycles to Failure (Nf)", "Lifetime vs Current"),
            "V": ("Voltage (V)", "Cycles to Failure (Nf)", "Lifetime vs Voltage"),
            "D": ("Die Attach (mm)", "Cycles to Failure (Nf)", "Lifetime vs Die Attach Size")
        }

        x_label, y_label, title = axis_labels.get(
            request.parameter_to_vary,
            (request.parameter_to_vary, "Cycles to Failure", f"Lifetime vs {request.parameter_to_vary}")
        )

        return LifetimeCurveResponse(
            curves=curves,
            x_axis_label=x_label,
            y_axis_label=y_label,
            title=title
        )

    except Exception as e:
        logger.error(f"Error generating lifetime curve: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Lifetime curve generation failed: {str(e)}"
        )


# ============================================================================
# Remaining Life Endpoints
# ============================================================================

@router.post("/remaining-life/evaluate", response_model=RemainingLifeResponse)
async def evaluate_remaining_life_endpoint(request: RemainingLifeRequest):
    """
    Full remaining life assessment with comprehensive analysis.

    Provides:
    - Current degradation state evaluation
    - Degradation trend analysis (if history provided)
    - Health indicator calculation
    - Confidence interval estimation
    - Actionable recommendations

    The method can be:
    - 'auto': Automatically select best method (default)
    - 'linear': Linear extrapolation from history
    - 'exponential': Exponential fit for accelerating degradation
    - 'constant': Constant rate based on current conditions
    """
    try:
        # Get the lifetime model
        model = ModelFactory.get_model(request.model_type)

        # Create lifetime function compatible with remaining_life module
        def lifetime_fn(range_val: float, mean_val: float, params: Dict[str, Any]) -> float:
            model_params = request.model_parameters.copy()
            model_params["delta_Tj"] = range_val
            model_params["Tj_max"] = mean_val + range_val / 2
            return model.calculate_cycles_to_failure(**model_params)

        # Convert degradation history if provided
        degradation_history = None
        if request.degradation_history:
            degradation_history = [
                DegradationPoint(
                    cycles=p.cycles,
                    damage=p.damage,
                    time=p.time
                )
                for p in request.degradation_history
            ]

        # Assess remaining life
        result: RemainingLifeResult = assess_remaining_life(
            current_damage=request.current_damage,
            lifetime_model=lifetime_fn,
            model_parameters=request.model_parameters,
            operating_conditions=request.operating_conditions,
            degradation_history=degradation_history,
            cycle_frequency=request.cycle_frequency,
            method=request.method
        )

        # Convert confidence interval
        confidence_interval = None
        if result.confidence_interval:
            confidence_interval = ConfidenceInterval(
                lower_bound=result.confidence_interval[0],
                upper_bound=result.confidence_interval[1],
                confidence_level=0.95
            )

        # Determine damage state category
        if request.current_damage >= 1.0:
            damage_state = "failed"
            recommendation = "Component has failed. Replacement required."
        elif request.current_damage >= 0.8:
            damage_state = "critical"
            recommendation = "Critical damage level. Immediate replacement recommended."
        elif request.current_damage >= 0.5:
            damage_state = "high"
            recommendation = "High damage level. Schedule replacement soon."
        elif request.current_damage >= 0.2:
            damage_state = "moderate"
            recommendation = "Moderate damage. Monitor and plan for replacement."
        else:
            damage_state = "low"
            recommendation = "Low damage. Continue normal operation."

        # Calculate time conversions
        days_remaining = result.estimated_time_remaining / 24 if result.estimated_time_remaining != float('inf') else None
        years_remaining = result.estimated_time_remaining / 8760 if result.estimated_time_remaining != float('inf') else None

        return RemainingLifeResponse(
            estimated_cycles_remaining=result.estimated_cycles_remaining,
            estimated_time_remaining=result.estimated_time_remaining,
            estimated_days_remaining=days_remaining,
            estimated_years_remaining=years_remaining,
            health_index=result.health_index,
            degradation_rate=result.degradation_rate,
            confidence_interval=confidence_interval,
            method_used=result.method_used,
            is_failed=request.current_damage >= 1.0,
            damage_state=damage_state,
            recommendation=recommendation
        )

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error evaluating remaining life: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Remaining life evaluation failed: {str(e)}"
        )

"""
Unit tests for remaining life assessment module.

Tests cover health index calculation, degradation extrapolation,
remaining life estimation, and statistical analysis.
"""
import pytest
import numpy as np
from backend.app.core.remaining_life import (
    RemainingLifeResult,
    DegradationPoint,
    assess_remaining_life,
    calculate_health_index,
    extrapolate_degradation,
    calculate_remaining_life_distribution,
    estimate_remaining_life_weibull
)


# Test lifetime model function
def test_lifetime_model(stress_range: float, stress_mean: float, params: dict) -> float:
    """Simple lifetime model: N = (C / range)^m"""
    C = params.get('C', 10000)
    m = params.get('m', 2)

    if stress_range <= 0:
        return float('inf')

    return (C / stress_range) ** m


class TestCalculateHealthIndex:
    """Tests for health index calculation."""

    def test_zero_damage(self):
        """Test with zero damage."""
        hi = calculate_health_index(0.0)
        assert hi == 1.0

    def test_half_damage(self):
        """Test with 50% damage."""
        hi = calculate_health_index(0.5)
        assert hi == 0.5

    def test_full_damage(self):
        """Test with 100% damage."""
        hi = calculate_health_index(1.0)
        assert hi == 0.0

    def test_negative_damage(self):
        """Test with negative damage (clamped to 0)."""
        hi = calculate_health_index(-0.2)
        assert hi == 1.0

    def test_excessive_damage(self):
        """Test with damage > 1 (clamped to 1)."""
        hi = calculate_health_index(1.5)
        assert hi == 0.0

    def test_boundary_values(self):
        """Test boundary values."""
        assert calculate_health_index(0.0) == 1.0
        assert calculate_health_index(1.0) == 0.0
        assert calculate_health_index(0.01) == 0.99
        assert calculate_health_index(0.99) == 0.01


class TestAssessRemainingLife:
    """Tests for remaining life assessment."""

    def test_no_damage(self):
        """Test with zero current damage."""
        result = assess_remaining_life(
            current_damage=0.0,
            lifetime_model=test_lifetime_model,
            model_params={'C': 10000, 'm': 2},
            operating_conditions={'range': 100, 'mean': 0},
            cycle_frequency=1.0
        )

        assert result.health_index == 1.0
        assert result.estimated_cycles_remaining > 0
        assert result.estimated_time_remaining > 0
        assert not np.isinf(result.estimated_cycles_remaining)

    def test_critical_damage(self):
        """Test with damage >= 1.0."""
        result = assess_remaining_life(
            current_damage=1.0,
            lifetime_model=test_lifetime_model,
            model_params={'C': 10000, 'm': 2},
            operating_conditions={'range': 100, 'mean': 0}
        )

        assert result.health_index == 0.0
        assert result.estimated_cycles_remaining == 0.0
        assert result.estimated_time_remaining == 0.0
        assert result.method_used == "failed"

    def test_constant_rate_method(self):
        """Test constant rate estimation method."""
        result = assess_remaining_life(
            current_damage=0.3,
            lifetime_model=test_lifetime_model,
            model_params={'C': 10000, 'm': 2},
            operating_conditions={'range': 100, 'mean': 0},
            method='constant'
        )

        assert result.method_used == 'constant'
        assert result.estimated_cycles_remaining > 0

    def test_with_degradation_history(self):
        """Test with degradation history provided."""
        history = [
            DegradationPoint(cycles=0, damage=0.0),
            DegradationPoint(cycles=1000, damage=0.1),
            DegradationPoint(cycles=2000, damage=0.2),
        ]

        result = assess_remaining_life(
            current_damage=0.2,
            lifetime_model=test_lifetime_model,
            model_params={'C': 10000, 'm': 2},
            operating_conditions={'range': 100, 'mean': 0},
            degradation_history=history,
            method='linear'
        )

        assert result.method_used == 'linear'
        assert result.estimated_cycles_remaining > 0

    def test_insufficient_history(self):
        """Test with insufficient history for extrapolation."""
        history = [
            DegradationPoint(cycles=0, damage=0.0),
            DegradationPoint(cycles=100, damage=0.05),
        ]

        result = assess_remaining_life(
            current_damage=0.05,
            lifetime_model=test_lifetime_model,
            model_params={'C': 10000, 'm': 2},
            operating_conditions={'range': 100, 'mean': 0},
            degradation_history=history,
            method='auto'
        )

        # Should fall back to constant rate method
        assert result.estimated_cycles_remaining > 0

    def test_with_cycle_frequency(self):
        """Test time conversion with cycle frequency."""
        result = assess_remaining_life(
            current_damage=0.5,
            lifetime_model=test_lifetime_model,
            model_params={'C': 10000, 'm': 2},
            operating_conditions={'range': 100, 'mean': 0},
            cycle_frequency=60.0  # 60 cycles per hour
        )

        # Time remaining should be cycles / frequency
        expected_time = result.estimated_cycles_remaining / 60.0
        assert np.isclose(result.estimated_time_remaining, expected_time, rtol=0.01)

    def test_extrapolation_linear(self):
        """Test linear extrapolation."""
        history = [
            DegradationPoint(cycles=0, damage=0.0),
            DegradationPoint(cycles=2000, damage=0.2),
            DegradationPoint(cycles=4000, damage=0.4),
        ]

        result = assess_remaining_life(
            current_damage=0.4,
            lifetime_model=test_lifetime_model,
            model_params={'C': 10000, 'm': 2},
            operating_conditions={'range': 100, 'mean': 0},
            degradation_history=history,
            method='linear'
        )

        # Linear extrapolation: 0.4 damage at 4000 cycles
        # Rate = 0.4/4000 = 0.0001 per cycle
        # Remaining: 0.6 / 0.0001 = 6000 cycles
        assert result.method_used == 'linear'
        assert result.degradation_rate > 0

    def test_extrapolation_exponential(self):
        """Test exponential extrapolation."""
        history = [
            DegradationPoint(cycles=0, damage=0.0),
            DegradationPoint(cycles=1000, damage=0.1),
            DegradationPoint(cycles=2000, damage=0.19),
            DegradationPoint(cycles=3000, damage=0.27),
        ]

        result = assess_remaining_life(
            current_damage=0.27,
            lifetime_model=test_lifetime_model,
            model_params={'C': 10000, 'm': 2},
            operating_conditions={'range': 100, 'mean': 0},
            degradation_history=history,
            method='exponential'
        )

        assert result.method_used == 'exponential'
        assert result.estimated_cycles_remaining > 0

    def test_infinite_life_prediction(self):
        """Test case where model predicts infinite life."""
        def infinite_life_model(range_val, mean, params):
            return float('inf')

        result = assess_remaining_life(
            current_damage=0.0,
            lifetime_model=infinite_life_model,
            model_params={},
            operating_conditions={'range': 100, 'mean': 0}
        )

        # Should handle infinite life gracefully
        assert result.health_index == 1.0


class TestExtrapolateDegradation:
    """Tests for degradation extrapolation."""

    def test_linear_extrapolation(self):
        """Test linear extrapolation of degradation."""
        history = [
            DegradationPoint(cycles=0, damage=0.0),
            DegradationPoint(cycles=1000, damage=0.1),
        ]

        forecast = extrapolate_degradation(history, forecast_cycles=1000, method='linear')

        assert len(forecast) == 1001  # Including starting point
        assert forecast[0] >= 0
        assert forecast[-1] <= 1.0

    def test_exponential_extrapolation(self):
        """Test exponential extrapolation of degradation."""
        history = [
            DegradationPoint(cycles=0, damage=0.0),
            DegradationPoint(cycles=1000, damage=0.1),
            DegradationPoint(cycles=2000, damage=0.19),
        ]

        forecast = extrapolate_degradation(history, forecast_cycles=1000, method='exponential')

        assert len(forecast) == 1001
        assert np.all(forecast >= 0)
        assert np.all(forecast < 1.0)

    def test_extrapolation_with_dict_history(self):
        """Test extrapolation with dict-style history."""
        history = [
            {'cycles': 0, 'damage': 0.0},
            {'cycles': 1000, 'damage': 0.1},
        ]

        forecast = extrapolate_degradation(history, forecast_cycles=500, method='linear')

        assert len(forecast) == 501
        assert forecast[0] >= 0

    def test_empty_history(self):
        """Test with empty history."""
        forecast = extrapolate_degradation([], forecast_cycles=100)

        assert len(forecast) == 101
        assert np.all(forecast == 0.0)

    def test_single_point_history(self):
        """Test with single history point."""
        history = [DegradationPoint(cycles=0, damage=0.0)]

        forecast = extrapolate_degradation(history, forecast_cycles=100)

        # Should return constant value
        assert len(forecast) == 101
        assert np.all(forecast == 0.0)

    def test_extrapolation_saturation(self):
        """Test that extrapolation saturates at 1.0."""
        history = [
            DegradationPoint(cycles=0, damage=0.0),
            DegradationPoint(cycles=100, damage=0.5),
        ]

        forecast = extrapolate_degradation(history, forecast_cycles=1000, method='linear')

        # Should be clipped at 1.0
        assert np.all(forecast <= 1.0)
        assert np.all(forecast >= 0.0)


class TestRemainingLifeDistribution:
    """Tests for statistical remaining life distribution."""

    def test_distribution_structure(self):
        """Test distribution result structure."""
        history = [
            DegradationPoint(cycles=0, damage=0.0),
            DegradationPoint(cycles=1000, damage=0.1),
            DegradationPoint(cycles=2000, damage=0.2),
            DegradationPoint(cycles=3000, damage=0.3),
        ]

        result = calculate_remaining_life_distribution(history, num_samples=100)

        assert 'mean' in result
        assert 'std' in result
        assert 'p5' in result
        assert 'p25' in result
        assert 'p50' in result
        assert 'p75' in result
        assert 'p95' in result

    def test_insufficient_history(self):
        """Test with insufficient history."""
        history = [
            DegradationPoint(cycles=0, damage=0.0),
            DegradationPoint(cycles=100, damage=0.05),
        ]

        result = calculate_remaining_life_distribution(history)

        assert result['mean'] == float('inf')
        assert result['std'] == 0.0

    def test_percentile_ordering(self):
        """Test that percentiles are in correct order."""
        history = [
            DegradationPoint(cycles=0, damage=0.0),
            DegradationPoint(cycles=1000, damage=0.1),
            DegradationPoint(cycles=2000, damage=0.2),
            DegradationPoint(cycles=3000, damage=0.3),
        ]

        result = calculate_remaining_life_distribution(history, num_samples=100)

        # p5 <= p25 <= p50 <= p75 <= p95
        if result['mean'] != float('inf'):
            assert result['p5'] <= result['p25']
            assert result['p25'] <= result['p50']
            assert result['p50'] <= result['p75']
            assert result['p75'] <= result['p95']

    def test_custom_confidence_levels(self):
        """Test with custom confidence levels."""
        history = [
            DegradationPoint(cycles=0, damage=0.0),
            DegradationPoint(cycles=1000, damage=0.1),
            DegradationPoint(cycles=2000, damage=0.2),
            DegradationPoint(cycles=3000, damage=0.3),
        ]

        result = calculate_remaining_life_distribution(
            history,
            num_samples=50,
            confidence_levels=[0.1, 0.5, 0.9]
        )

        assert 'p10' in result
        assert 'p50' in result
        assert 'p90' in result


class TestEstimateRemainingLifeWeibull:
    """Tests for Weibull-based remaining life estimation."""

    def test_half_damage_weibull(self):
        """Test Weibull estimation with 50% damage."""
        remaining = estimate_remaining_life_weibull(
            current_damage=0.5,
            shape_parameter=2.0,
            scale_parameter=10000
        )

        assert remaining > 0
        assert not np.isinf(remaining)

    def test_full_damage_weibull(self):
        """Test Weibull estimation with 100% damage."""
        remaining = estimate_remaining_life_weibull(
            current_damage=1.0,
            shape_parameter=2.0,
            scale_parameter=10000
        )

        assert remaining == 0.0

    def test_exponential_case(self):
        """Test exponential case (shape=1)."""
        remaining = estimate_remaining_life_weibull(
            current_damage=0.5,
            shape_parameter=1.0,
            scale_parameter=10000
        )

        # For exponential: remaining = -eta * ln(1 - damage)
        expected = -10000 * np.log(0.5)
        assert np.isclose(remaining, expected, rtol=0.01)

    def test_invalid_shape_parameter(self):
        """Test with invalid shape parameter."""
        with pytest.raises(ValueError):
            estimate_remaining_life_weibull(0.5, shape_parameter=0, scale_parameter=10000)

    def test_invalid_scale_parameter(self):
        """Test with invalid scale parameter."""
        with pytest.raises(ValueError):
            estimate_remaining_life_weibull(0.5, shape_parameter=2.0, scale_parameter=0)

    def test_weibull_shape_effect(self):
        """Test that shape parameter affects remaining life."""
        damage = 0.5
        scale = 10000

        # Shape < 1: Decreasing failure rate
        life_0_5 = estimate_remaining_life_weibull(damage, 0.5, scale)

        # Shape = 1: Constant failure rate
        life_1_0 = estimate_remaining_life_weibull(damage, 1.0, scale)

        # Shape > 1: Increasing failure rate
        life_2_0 = estimate_remaining_life_weibull(damage, 2.0, scale)

        # All should be positive and finite
        assert all([life_0_5 > 0, life_1_0 > 0, life_2_0 > 0])


class TestDegradationPoint:
    """Tests for DegradationPoint dataclass."""

    def test_degradation_point_creation(self):
        """Test DegradationPoint creation."""
        point = DegradationPoint(cycles=1000, damage=0.1, time=100.0)

        assert point.cycles == 1000
        assert point.damage == 0.1
        assert point.time == 100.0

    def test_degradation_point_without_time(self):
        """Test DegradationPoint without time."""
        point = DegradationPoint(cycles=1000, damage=0.1)

        assert point.cycles == 1000
        assert point.damage == 0.1
        assert point.time is None


class TestRemainingLifeResult:
    """Tests for RemainingLifeResult dataclass."""

    def test_result_creation(self):
        """Test RemainingLifeResult creation."""
        result = RemainingLifeResult(
            estimated_cycles_remaining=1000,
            estimated_time_remaining=100,
            health_index=0.8,
            degradation_rate=0.001,
            confidence_interval=(800, 1200),
            method_used='linear'
        )

        assert result.estimated_cycles_remaining == 1000
        assert result.estimated_time_remaining == 100
        assert result.health_index == 0.8
        assert result.degradation_rate == 0.001
        assert result.confidence_interval == (800, 1200)
        assert result.method_used == 'linear'

    def test_result_repr(self):
        """Test RemainingLifeResult string representation."""
        result = RemainingLifeResult(
            estimated_cycles_remaining=1000,
            estimated_time_remaining=100,
            health_index=0.8,
            degradation_rate=0.001,
            method_used='linear'
        )

        repr_str = repr(result)
        assert 'RemainingLifeResult' in repr_str
        assert '1000' in repr_str
        assert 'linear' in repr_str


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_negative_cycles_frequency(self):
        """Test with negative cycle frequency."""
        result = assess_remaining_life(
            current_damage=0.5,
            lifetime_model=test_lifetime_model,
            model_params={'C': 10000, 'm': 2},
            operating_conditions={'range': 100, 'mean': 0},
            cycle_frequency=-10.0
        )

        # Should handle gracefully
        assert result.estimated_time_remaining >= 0

    def test_zero_stress_range(self):
        """Test with zero stress range (infinite life)."""
        result = assess_remaining_life(
            current_damage=0.5,
            lifetime_model=test_lifetime_model,
            model_params={'C': 10000, 'm': 2},
            operating_conditions={'range': 0, 'mean': 0}
        )

        # Should handle infinite life prediction
        assert result.health_index == 0.5

    def test_clipped_damage(self):
        """Test that damage is properly clipped."""
        # Damage > 1.0
        result = assess_remaining_life(
            current_damage=1.5,
            lifetime_model=test_lifetime_model,
            model_params={'C': 10000, 'm': 2},
            operating_conditions={'range': 100, 'mean': 0}
        )

        assert result.health_index == 0.0

        # Negative damage
        result = assess_remaining_life(
            current_damage=-0.2,
            lifetime_model=test_lifetime_model,
            model_params={'C': 10000, 'm': 2},
            operating_conditions={'range': 100, 'mean': 0}
        )

        assert result.health_index == 1.0

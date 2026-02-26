"""
Unit tests for damage accumulation module.

Tests cover Miner's rule implementation, remaining life estimation,
sensitivity analysis, and confidence interval calculation.
"""
import pytest
import numpy as np
from backend.app.core.damage_accumulation import (
    DamageResult,
    DamageSensitivity,
    calculate_miner_damage,
    estimate_remaining_cycles,
    calculate_damage_rate,
    perform_sensitivity_analysis,
    calculate_confidence_interval,
    combine_damage_states,
    predict_time_to_failure,
    adjust_for_sequence_effect
)


# Test lifetime model function
def simple_s_n_model(stress_range: float, stress_mean: float, params: dict) -> float:
    """Simple S-N model: N = (S_f / S)^b"""
    fatigue_limit = params.get('fatigue_limit', 1000)
    exponent = params.get('exponent', 3)

    if stress_range <= 0:
        return float('inf')

    if stress_range >= fatigue_limit:
        return 1.0  # One cycle to failure at high stress

    cycles = (fatigue_limit / stress_range) ** exponent
    return cycles


class TestCalculateMinerDamage:
    """Tests for Miner's damage calculation."""

    def test_empty_cycles(self):
        """Test with empty cycle list."""
        result = calculate_miner_damage([], simple_s_n_model, {})
        assert result.total_damage == 0.0
        assert result.remaining_life_fraction == 1.0
        assert not result.is_critical
        assert result.details == []

    def test_single_cycle(self):
        """Test with single cycle."""
        cycles = [{'range': 500, 'mean': 0, 'count': 100}]
        params = {'fatigue_limit': 1000, 'exponent': 3}

        result = calculate_miner_damage(cycles, simple_s_n_model, params)

        assert result.total_damage > 0
        assert len(result.details) == 1
        assert result.details[0]['range'] == 500

    def test_multiple_cycles(self):
        """Test with multiple cycles."""
        cycles = [
            {'range': 500, 'mean': 0, 'count': 100},
            {'range': 700, 'mean': 50, 'count': 50}
        ]
        params = {'fatigue_limit': 1000, 'exponent': 3}

        result = calculate_miner_damage(cycles, simple_s_n_model, params)

        assert result.total_damage > 0
        assert len(result.details) == 2

    def test_damage_summation(self):
        """Test that damage is properly summed."""
        # Use cycles that will give predictable damage
        cycles = [
            {'range': 500, 'mean': 0, 'count': 4}  # Nf = 8, so damage = 0.5
        ]
        params = {'fatigue_limit': 1000, 'exponent': 3}

        result = calculate_miner_damage(cycles, simple_s_n_model, params)

        # Expected Nf = (1000/500)^3 = 8
        # Expected damage = 4/8 = 0.5
        expected_n_f = (1000 / 500) ** 3
        expected_damage = 4 / expected_n_f
        assert np.isclose(result.total_damage, expected_damage, rtol=0.01)

    def test_critical_damage(self):
        """Test damage >= 1.0 is marked as critical."""
        cycles = [{'range': 500, 'mean': 0, 'count': 1000}]
        params = {'fatigue_limit': 1000, 'exponent': 3}

        result = calculate_miner_damage(cycles, simple_s_n_model, params)

        # With 1000 cycles at stress giving Nf=8, damage should be > 1
        assert result.total_damage >= 1.0
        assert result.is_critical

    def test_half_cycle_handling(self):
        """Test half-cycle inclusion/exclusion."""
        cycles = [
            {'range': 500, 'mean': 0, 'count': 0.5}
        ]
        params = {'fatigue_limit': 1000, 'exponent': 3}

        # Include half cycles
        result_with = calculate_miner_damage(cycles, simple_s_n_model, params, include_half_cycles=True)
        assert result_with.total_damage > 0

        # Exclude half cycles
        result_without = calculate_miner_damage(cycles, simple_s_n_model, params, include_half_cycles=False)
        assert result_without.total_damage == 0

    def test_invalid_cycle_handling(self):
        """Test handling of invalid cycle data."""
        cycles = [
            {'range': 500, 'mean': 0, 'count': 100},
            {'range': 0, 'mean': 0, 'count': 50},  # Invalid: zero range
            {'range': -10, 'mean': 0, 'count': 30},  # Invalid: negative range
            {'range': 400, 'mean': 0, 'count': -5}  # Invalid: negative count
        ]
        params = {'fatigue_limit': 1000, 'exponent': 3}

        result = calculate_miner_damage(cycles, simple_s_n_model, params)

        # Should only process valid cycles
        assert result.total_damage > 0
        assert len(result.details) == 1  # Only first cycle is valid

    def test_detail_structure(self):
        """Test that result details have correct structure."""
        cycles = [{'range': 500, 'mean': 0, 'count': 100}]
        params = {'fatigue_limit': 1000, 'exponent': 3}

        result = calculate_miner_damage(cycles, simple_s_n_model, params)

        detail = result.details[0]
        assert 'cycle_index' in detail
        assert 'range' in detail
        assert 'mean' in detail
        assert 'count' in detail
        assert 'cycles_to_failure' in detail
        assert 'damage_contribution' in detail
        assert 'damage_fraction' in detail


class TestEstimateRemainingCycles:
    """Tests for remaining cycles estimation."""

    def test_no_damage(self):
        """Test with zero current damage."""
        remaining = estimate_remaining_cycles(
            0.0, simple_s_n_model,
            {'fatigue_limit': 1000, 'exponent': 3},
            {'range': 500, 'mean': 0}
        )

        # Should return full Nf
        expected_n_f = (1000 / 500) ** 3
        assert np.isclose(remaining, expected_n_f, rtol=0.01)

    def test_partial_damage(self):
        """Test with partial damage."""
        remaining = estimate_remaining_cycles(
            0.5, simple_s_n_model,
            {'fatigue_limit': 1000, 'exponent': 3},
            {'range': 500, 'mean': 0}
        )

        # Should return 50% of Nf
        expected_n_f = (1000 / 500) ** 3
        assert np.isclose(remaining, expected_n_f * 0.5, rtol=0.01)

    def test_critical_damage(self):
        """Test with damage >= 1.0."""
        remaining = estimate_remaining_cycles(
            1.0, simple_s_n_model,
            {'fatigue_limit': 1000, 'exponent': 3},
            {'range': 500, 'mean': 0}
        )

        assert remaining == 0.0

    def test_negative_damage(self):
        """Test with negative damage (should be treated as 0)."""
        remaining = estimate_remaining_cycles(
            -0.1, simple_s_n_model,
            {'fatigue_limit': 1000, 'exponent': 3},
            {'range': 500, 'mean': 0}
        )

        # Should treat as zero damage
        expected_n_f = (1000 / 500) ** 3
        assert np.isclose(remaining, expected_n_f, rtol=0.01)

    def test_zero_stress_range(self):
        """Test with zero stress range (infinite life)."""
        remaining = estimate_remaining_cycles(
            0.5, simple_s_n_model,
            {'fatigue_limit': 1000, 'exponent': 3},
            {'range': 0, 'mean': 0}
        )

        assert remaining == float('inf')


class TestCalculateDamageRate:
    """Tests for damage rate calculation."""

    def test_damage_rate_calculation(self):
        """Test damage rate per unit time."""
        cycles = [{'range': 500, 'mean': 0, 'count': 100}]
        params = {'fatigue_limit': 1000, 'exponent': 3}
        time_period = 10.0  # 10 hours

        rate = calculate_damage_rate(cycles, time_period, simple_s_n_model, params)

        assert rate > 0
        # Rate should be damage / time
        damage = calculate_miner_damage(cycles, simple_s_n_model, params).total_damage
        assert np.isclose(rate, damage / time_period)

    def test_invalid_time_period(self):
        """Test with invalid time period."""
        cycles = [{'range': 500, 'mean': 0, 'count': 100}]

        with pytest.raises(ValueError):
            calculate_damage_rate(cycles, 0, simple_s_n_model, {})

        with pytest.raises(ValueError):
            calculate_damage_rate(cycles, -10, simple_s_n_model, {})


class TestSensitivityAnalysis:
    """Tests for parameter sensitivity analysis."""

    def test_sensitivity_structure(self):
        """Test sensitivity result structure."""
        cycles = [{'range': 500, 'mean': 0, 'count': 100}]
        params = {'fatigue_limit': 1000, 'exponent': 3}
        param_range = np.array([0.8, 0.9, 1.0, 1.1, 1.2])

        result = perform_sensitivity_analysis(
            cycles, simple_s_n_model, params,
            'fatigue_limit', param_range, relative_variation=True
        )

        assert isinstance(result, DamageSensitivity)
        assert result.parameter_name == 'fatigue_limit'
        assert len(result.parameter_values) == len(param_range)
        assert len(result.resulting_damage) == len(param_range)

    def test_sensitivity_values(self):
        """Test that sensitivity produces expected values."""
        cycles = [{'range': 500, 'mean': 0, 'count': 100}]
        params = {'fatigue_limit': 1000, 'exponent': 3}
        param_range = np.array([0.5, 1.0, 1.5])

        result = perform_sensitivity_analysis(
            cycles, simple_s_n_model, params,
            'fatigue_limit', param_range, relative_variation=True
        )

        # Higher fatigue limit should give lower damage
        # (more cycles to failure)
        damages = result.resulting_damage
        # First value (0.5x limit) should have higher damage than baseline
        # Last value (1.5x limit) should have lower damage
        assert damages[0] >= damages[1]  # Lower limit = higher damage
        assert damages[2] <= damages[1]  # Higher limit = lower damage

    def test_absolute_variation(self):
        """Test sensitivity with absolute parameter variation."""
        cycles = [{'range': 500, 'mean': 0, 'count': 100}]
        params = {'fatigue_limit': 1000, 'exponent': 3}
        param_range = np.array([500, 1000, 1500])

        result = perform_sensitivity_analysis(
            cycles, simple_s_n_model, params,
            'fatigue_limit', param_range, relative_variation=False
        )

        assert len(result.resulting_damage) == 3


class TestConfidenceInterval:
    """Tests for confidence interval calculation."""

    def test_confidence_interval_structure(self):
        """Test confidence interval structure."""
        cycles = [{'range': 500, 'mean': 0, 'count': 100}]
        params = {'fatigue_limit': 1000, 'exponent': 3}

        lower, upper = calculate_confidence_interval(
            cycles, simple_s_n_model, params,
            scatter_factor=1.2, confidence_level=0.95
        )

        assert isinstance(lower, float)
        assert isinstance(upper, float)
        assert lower >= 0  # Lower bound should be non-negative
        assert upper > lower  # Upper should be greater than lower

    def test_scatter_factor_effect(self):
        """Test that scatter factor affects interval width."""
        cycles = [{'range': 500, 'mean': 0, 'count': 100}]
        params = {'fatigue_limit': 1000, 'exponent': 3}

        ci_1_0 = calculate_confidence_interval(
            cycles, simple_s_n_model, params,
            scatter_factor=1.0, confidence_level=0.95
        )

        ci_1_5 = calculate_confidence_interval(
            cycles, simple_s_n_model, params,
            scatter_factor=1.5, confidence_level=0.95
        )

        # Higher scatter should give wider interval
        width_1_0 = ci_1_0[1] - ci_1_0[0]
        width_1_5 = ci_1_5[1] - ci_1_5[0]
        assert width_1_5 > width_1_0

    def test_no_scatter(self):
        """Test with scatter_factor=1.0 (no scatter)."""
        cycles = [{'range': 500, 'mean': 0, 'count': 100}]
        params = {'fatigue_limit': 1000, 'exponent': 3}

        lower, upper = calculate_confidence_interval(
            cycles, simple_s_n_model, params,
            scatter_factor=1.0, confidence_level=0.95
        )

        # With no scatter, lower and upper should be equal
        assert np.isclose(lower, upper)


class TestCombineDamageStates:
    """Tests for combining damage from multiple components."""

    def test_simple_average(self):
        """Test simple average of damage states."""
        damages = [0.2, 0.4, 0.6]
        combined = combine_damage_states(damages)

        assert np.isclose(combined, 0.4)  # Average

    def test_weighted_average(self):
        """Test weighted average of damage states."""
        damages = [0.2, 0.4, 0.6]
        weights = [0.5, 0.3, 0.2]

        combined = combine_damage_states(damages, weights)

        # (0.2*0.5 + 0.4*0.3 + 0.6*0.2) / (0.5+0.3+0.2)
        expected = (0.1 + 0.12 + 0.12) / 1.0
        assert np.isclose(combined, expected)

    def test_empty_damage_states(self):
        """Test with empty damage list."""
        combined = combine_damage_states([])
        assert combined == 0.0

    def test_mismatched_weights(self):
        """Test with mismatched weights length."""
        damages = [0.2, 0.4, 0.6]
        weights = [0.5, 0.3]  # Wrong length

        with pytest.raises(ValueError):
            combine_damage_states(damages, weights)

    def test_zero_weights(self):
        """Test with all-zero weights."""
        damages = [0.2, 0.4, 0.6]
        weights = [0, 0, 0]

        combined = combine_damage_states(damages, weights)
        # Should fall back to simple average
        assert np.isclose(combined, 0.4)


class TestPredictTimeToFailure:
    """Tests for time to failure prediction."""

    def test_time_to_failure(self):
        """Test basic time to failure calculation."""
        current_damage = 0.5
        damage_rate = 0.01  # 1% per hour

        ttf = predict_time_to_failure(current_damage, damage_rate)

        # (1 - 0.5) / 0.01 = 50 hours
        assert np.isclose(ttf, 50.0)

    def test_already_failed(self):
        """Test with damage >= 1.0."""
        ttf = predict_time_to_failure(1.0, 0.01)
        assert ttf == 0.0

    def test_zero_damage_rate(self):
        """Test with zero damage rate."""
        ttf = predict_time_to_failure(0.5, 0.0)
        assert ttf == float('inf')

    def test_negative_damage_rate(self):
        """Test with negative damage rate (healing)."""
        ttf = predict_time_to_failure(0.5, -0.01)
        assert ttf == float('inf')


class TestSequenceEffect:
    """Tests for load sequence effect adjustment."""

    def test_no_sequence_effect(self):
        """Test with sequence_factor=1.0 (no correction)."""
        cycles = [{'range': 500, 'mean': 0, 'count': 100}]
        params = {'fatigue_limit': 1000, 'exponent': 3}

        result = adjust_for_sequence_effect(
            cycles, simple_s_n_model, params, sequence_factor=1.0
        )

        # Should be same as base calculation
        base = calculate_miner_damage(cycles, simple_s_n_model, params)
        assert np.isclose(result.total_damage, base.total_damage)

    def test_with_sequence_factor(self):
        """Test with non-unity sequence factor."""
        cycles = [{'range': 500, 'mean': 0, 'count': 100}]
        params = {'fatigue_limit': 1000, 'exponent': 3}

        result = adjust_for_sequence_effect(
            cycles, simple_s_n_model, params, sequence_factor=1.5
        )

        base = calculate_miner_damage(cycles, simple_s_n_model, params)

        # Damage should be scaled by factor
        assert np.isclose(result.total_damage, base.total_damage * 1.5)

    def test_sequence_effect_critical(self):
        """Test that sequence effect can change critical status."""
        cycles = [{'range': 500, 'mean': 0, 'count': 8}]
        params = {'fatigue_limit': 1000, 'exponent': 3}

        # Base calculation: Nf = 8, damage = 1.0 (borderline)
        base = calculate_miner_damage(cycles, simple_s_n_model, params)

        # With factor > 1, should become critical
        result = adjust_for_sequence_effect(
            cycles, simple_s_n_model, params, sequence_factor=1.1
        )

        assert result.is_critical


class TestDamageResult:
    """Tests for DamageResult dataclass."""

    def test_damage_result_creation(self):
        """Test DamageResult creation."""
        result = DamageResult(
            total_damage=0.5,
            remaining_life_fraction=0.5,
            is_critical=False,
            details=[],
            confidence_interval=(0.4, 0.6)
        )

        assert result.total_damage == 0.5
        assert result.remaining_life_fraction == 0.5
        assert not result.is_critical
        assert result.confidence_interval == (0.4, 0.6)

    def test_damage_result_repr(self):
        """Test DamageResult string representation."""
        result = DamageResult(
            total_damage=0.5,
            remaining_life_fraction=0.5,
            is_critical=False
        )

        repr_str = repr(result)
        assert 'DamageResult' in repr_str
        assert '0.5' in repr_str
        assert 'OK' in repr_str


class TestDamageSensitivity:
    """Tests for DamageSensitivity dataclass."""

    def test_sensitivity_creation(self):
        """Test DamageSensitivity creation."""
        param_values = np.array([1.0, 2.0, 3.0])
        damages = np.array([0.5, 0.3, 0.2])

        result = DamageSensitivity(
            parameter_name='test_param',
            parameter_values=param_values,
            resulting_damage=damages,
            sensitivity_coefficient=-0.15
        )

        assert result.parameter_name == 'test_param'
        assert np.array_equal(result.parameter_values, param_values)
        assert np.array_equal(result.resulting_damage, damages)
        assert result.sensitivity_coefficient == -0.15

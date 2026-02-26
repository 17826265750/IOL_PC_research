"""
Unit tests for Damage Accumulation (Miner's Rule).

Tests include:
- Miner's rule calculations with known damage values
- Empty and edge case handling
- Remaining life estimation
- Damage rate calculation
- Sensitivity analysis
- Confidence intervals
"""

import pytest
import numpy as np

from app.core.damage_accumulation import (
    calculate_miner_damage,
    estimate_remaining_cycles,
    calculate_damage_rate,
    perform_sensitivity_analysis,
    calculate_confidence_interval,
    combine_damage_states,
    predict_time_to_failure,
    adjust_for_sequence_effect,
    DamageResult,
    DamageSensitivity
)


class TestCalculateMinerDamage:
    """Test Miner's rule damage calculation."""

    def test_empty_cycles(self):
        """Test with empty cycle list."""
        result = calculate_miner_damage(
            cycles=[],
            lifetime_model=lambda r, m, p: 1000,
            model_params={}
        )

        assert result.total_damage == 0.0
        assert result.remaining_life_fraction == 1.0
        assert result.is_critical is False
        assert result.details == []

    def test_single_cycle(self):
        """Test with single cycle condition."""
        cycles = [{'range': 100, 'mean': 50, 'count': 100}]

        def model(range_val, mean_val, params):
            return 1000  # Fixed Nf

        result = calculate_miner_damage(cycles, model, {})

        # Damage = n/N = 100/1000 = 0.1
        assert result.total_damage == 0.1
        assert result.remaining_life_fraction == 0.9
        assert result.is_critical is False

    def test_multiple_cycles(self):
        """Test with multiple cycle conditions."""
        cycles = [
            {'range': 100, 'mean': 50, 'count': 500},
            {'range': 80, 'mean': 40, 'count': 300},
        ]

        def model(range_val, mean_val, params):
            return 1000  # Fixed Nf

        result = calculate_miner_damage(cycles, model, {})

        # Damage = (500+300)/1000 = 0.8
        assert result.total_damage == 0.8
        assert result.remaining_life_fraction == 0.2

    def test_critical_damage(self):
        """Test when damage exceeds 1.0 (failure predicted)."""
        cycles = [
            {'range': 100, 'mean': 50, 'count': 600},
            {'range': 80, 'mean': 40, 'count': 500},
        ]

        def model(range_val, mean_val, params):
            return 1000

        result = calculate_miner_damage(cycles, model, {})

        assert result.total_damage == 1.1
        assert result.is_critical is True

    def test_half_cycle_handling(self):
        """Test half cycle inclusion."""
        cycles = [
            {'range': 100, 'mean': 50, 'count': 1.0},
            {'range': 80, 'mean': 40, 'count': 0.5},
        ]

        def model(range_val, mean_val, params):
            return 100

        result = calculate_miner_damage(cycles, model, {}, include_half_cycles=True)

        # Damage = 1.0/100 + 0.5/100 = 0.015
        assert result.total_damage == 0.015

    def test_exclude_half_cycles(self):
        """Test excluding half cycles."""
        cycles = [
            {'range': 100, 'mean': 50, 'count': 1.0},
            {'range': 80, 'mean': 40, 'count': 0.5},
        ]

        def model(range_val, mean_val, params):
            return 100

        result = calculate_miner_damage(cycles, model, {}, include_half_cycles=False)

        # Only 1.0/100 = 0.01
        assert result.total_damage == 0.01

    def test_zero_or_negative_count_skipped(self):
        """Test that zero or negative counts are skipped."""
        cycles = [
            {'range': 100, 'mean': 50, 'count': 100},
            {'range': 80, 'mean': 40, 'count': 0},
            {'range': 60, 'mean': 30, 'count': -10},
        ]

        def model(range_val, mean_val, params):
            return 100

        result = calculate_miner_damage(cycles, model, {})

        # Only 100/100 = 1.0
        assert result.total_damage == 1.0

    def test_zero_range_skipped(self):
        """Test that zero range is skipped."""
        cycles = [
            {'range': 100, 'mean': 50, 'count': 100},
            {'range': 0, 'mean': 0, 'count': 100},
        ]

        def model(range_val, mean_val, params):
            return 100

        result = calculate_miner_damage(cycles, model, {})

        # Only 100/100 = 1.0
        assert result.total_damage == 1.0

    def test_model_error_handling(self):
        """Test handling of model errors."""
        cycles = [{'range': 100, 'mean': 50, 'count': 100}]

        def failing_model(range_val, mean_val, params):
            raise ValueError("Model error")

        result = calculate_miner_damage(cycles, failing_model, {})

        # Should handle error gracefully (treat as infinite Nf, damage=0)
        assert result.total_damage == 0.0

    def test_infinite_nf_handling(self):
        """Test handling of infinite cycles to failure."""
        cycles = [{'range': 1, 'mean': 0.5, 'count': 100}]

        def infinite_model(range_val, mean_val, params):
            return float('inf')

        result = calculate_miner_damage(cycles, infinite_model, {})

        # Damage should be 0 for infinite Nf
        assert result.total_damage == 0.0

    def test_details_structure(self):
        """Test that details are properly populated."""
        cycles = [
            {'range': 100, 'mean': 50, 'count': 100},
            {'range': 80, 'mean': 40, 'count': 200},
        ]

        def model(range_val, mean_val, params):
            return 1000

        result = calculate_miner_damage(cycles, model, {})

        assert len(result.details) == 2

        # Check first detail
        detail = result.details[0]
        assert 'cycle_index' in detail
        assert 'range' in detail
        assert 'mean' in detail
        assert 'count' in detail
        assert 'cycles_to_failure' in detail
        assert 'damage_contribution' in detail
        assert detail['damage_contribution'] == 0.1

    def test_damage_fraction_calculation(self):
        """Test damage fraction in details."""
        cycles = [
            {'range': 100, 'mean': 50, 'count': 100},
            {'range': 80, 'mean': 40, 'count': 100},
        ]

        def model(range_val, mean_val, params):
            return 1000

        result = calculate_miner_damage(cycles, model, {})

        # Each contributes 0.1, so fraction is 0.5 each
        assert result.details[0]['damage_fraction'] == 0.5
        assert result.details[1]['damage_fraction'] == 0.5


class TestEstimateRemainingCycles:
    """Test remaining cycles estimation."""

    def test_no_damage(self):
        """Test with zero damage."""
        remaining = estimate_remaining_cycles(
            current_damage=0.0,
            lifetime_model=lambda r, m, p: 10000,
            model_params={},
            cycle_condition={'range': 100, 'mean': 50}
        )

        # Should be full Nf
        assert remaining == 10000

    def test_half_damage(self):
        """Test with 50% damage."""
        remaining = estimate_remaining_cycles(
            current_damage=0.5,
            lifetime_model=lambda r, m, p: 10000,
            model_params={},
            cycle_condition={'range': 100, 'mean': 50}
        )

        # Should be half of Nf
        assert remaining == 5000

    def test_full_damage(self):
        """Test with damage at 1.0 (failed)."""
        remaining = estimate_remaining_cycles(
            current_damage=1.0,
            lifetime_model=lambda r, m, p: 10000,
            model_params={},
            cycle_condition={'range': 100, 'mean': 50}
        )

        # No remaining cycles
        assert remaining == 0.0

    def test_over_damage(self):
        """Test with damage exceeding 1.0."""
        remaining = estimate_remaining_cycles(
            current_damage=1.5,
            lifetime_model=lambda r, m, p: 10000,
            model_params={},
            cycle_condition={'range': 100, 'mean': 50}
        )

        # Already failed
        assert remaining == 0.0

    def test_negative_damage_clamped(self):
        """Test with negative damage (should be clamped to 0)."""
        remaining = estimate_remaining_cycles(
            current_damage=-0.1,
            lifetime_model=lambda r, m, p: 10000,
            model_params={},
            cycle_condition={'range': 100, 'mean': 50}
        )

        # Should treat as 0 damage
        assert remaining == 10000

    def test_zero_range_infinite_remaining(self):
        """Test with zero range (no stress)."""
        remaining = estimate_remaining_cycles(
            current_damage=0.5,
            lifetime_model=lambda r, m, p: 10000,
            model_params={},
            cycle_condition={'range': 0, 'mean': 0}
        )

        # Zero range means infinite life remaining
        assert remaining == float('inf')

    def test_model_error_handling(self):
        """Test handling of model errors."""
        remaining = estimate_remaining_cycles(
            current_damage=0.5,
            lifetime_model=lambda r, m, p: (_ for _ in ()).throw(ValueError("Error")),
            model_params={},
            cycle_condition={'range': 100, 'mean': 50}
        )

        # Should return 0 on error
        assert remaining == 0.0


class TestCalculateDamageRate:
    """Test damage rate calculation."""

    def test_basic_rate(self):
        """Test basic damage rate."""
        cycles = [{'range': 100, 'mean': 50, 'count': 100}]

        rate = calculate_damage_rate(
            cycles=cycles,
            time_period=10,
            lifetime_model=lambda r, m, p: 1000,
            model_params={}
        )

        # Damage = 0.1, rate = 0.1/10 = 0.01 per time unit
        assert rate == 0.01

    def test_rate_with_multiple_cycles(self):
        """Test rate with multiple cycle conditions."""
        cycles = [
            {'range': 100, 'mean': 50, 'count': 500},
            {'range': 80, 'mean': 40, 'count': 300},
        ]

        rate = calculate_damage_rate(
            cycles=cycles,
            time_period=10,
            lifetime_model=lambda r, m, p: 1000,
            model_params={}
        )

        # Total damage = 0.8, rate = 0.8/10 = 0.08
        assert rate == 0.08

    def test_zero_time_period_raises_error(self):
        """Test that zero time period raises error."""
        cycles = [{'range': 100, 'mean': 50, 'count': 100}]

        with pytest.raises(ValueError, match="positive"):
            calculate_damage_rate(
                cycles=cycles,
                time_period=0,
                lifetime_model=lambda r, m, p: 1000,
                model_params={}
            )

    def test_negative_time_period_raises_error(self):
        """Test that negative time period raises error."""
        cycles = [{'range': 100, 'mean': 50, 'count': 100}]

        with pytest.raises(ValueError, match="positive"):
            calculate_damage_rate(
                cycles=cycles,
                time_period=-10,
                lifetime_model=lambda r, m, p: 1000,
                model_params={}
            )


class TestPerformSensitivityAnalysis:
    """Test sensitivity analysis on damage."""

    def test_basic_sensitivity(self):
        """Test basic sensitivity calculation."""
        cycles = [{'range': 100, 'mean': 50, 'count': 100}]
        base_params = {'A': 1e6, 'alpha': 2.0}

        def model(range_val, mean_val, params):
            return params['A'] * (range_val ** (-params['alpha']))

        param_range = np.linspace(0.5e6, 1.5e6, 5)

        result = perform_sensitivity_analysis(
            cycles=cycles,
            lifetime_model=model,
            base_params=base_params,
            param_name='A',
            param_range=param_range
        )

        assert isinstance(result, DamageSensitivity)
        assert result.parameter_name == 'A'
        assert len(result.resulting_damage) == 5

    def test_relative_variation(self):
        """Test relative variation mode."""
        cycles = [{'range': 100, 'mean': 50, 'count': 100}]
        base_params = {'A': 1e6, 'alpha': 2.0}

        def model(range_val, mean_val, params):
            return params['A'] * (range_val ** (-params['alpha']))

        param_range = np.array([0.5, 1.0, 1.5])

        result = perform_sensitivity_analysis(
            cycles=cycles,
            lifetime_model=model,
            base_params=base_params,
            param_name='A',
            param_range=param_range,
            relative_variation=True
        )

        # Check that values were multiplied
        assert len(result.resulting_damage) == 3

    def test_sensitivity_coefficient_calculation(self):
        """Test sensitivity coefficient calculation."""
        cycles = [{'range': 100, 'mean': 50, 'count': 100}]
        base_params = {'A': 1e6}

        def model(range_val, mean_val, params):
            return params['A']

        # Linear relationship should give coefficient
        param_range = np.array([1e6, 2e6, 3e6])

        result = perform_sensitivity_analysis(
            cycles=cycles,
            lifetime_model=model,
            base_params=base_params,
            param_name='A',
            param_range=param_range
        )

        # Should have some sensitivity coefficient
        assert isinstance(result.sensitivity_coefficient, float)


class TestCalculateConfidenceInterval:
    """Test confidence interval calculation."""

    def test_basic_confidence_interval(self):
        """Test basic confidence interval."""
        cycles = [{'range': 100, 'mean': 50, 'count': 100}]

        ci = calculate_confidence_interval(
            cycles=cycles,
            lifetime_model=lambda r, m, p: 1000,
            model_params={},
            scatter_factor=1.2,
            confidence_level=0.95
        )

        assert len(ci) == 2
        assert ci[0] >= 0  # Lower bound non-negative
        assert ci[1] >= ci[0]  # Upper >= lower

    def test_scatter_factor_effect(self):
        """Test effect of scatter factor."""
        cycles = [{'range': 100, 'mean': 50, 'count': 100}]

        ci1 = calculate_confidence_interval(
            cycles=cycles,
            lifetime_model=lambda r, m, p: 1000,
            model_params={},
            scatter_factor=1.0,
            confidence_level=0.95
        )

        ci2 = calculate_confidence_interval(
            cycles=cycles,
            lifetime_model=lambda r, m, p: 1000,
            model_params={},
            scatter_factor=2.0,
            confidence_level=0.95
        )

        # Higher scatter factor should give wider interval
        width1 = ci1[1] - ci1[0]
        width2 = ci2[1] - ci2[0]
        assert width2 > width1

    def test_68_percent_confidence(self):
        """Test 68% confidence interval (1 std)."""
        cycles = [{'range': 100, 'mean': 50, 'count': 100}]

        ci = calculate_confidence_interval(
            cycles=cycles,
            lifetime_model=lambda r, m, p: 1000,
            model_params={},
            scatter_factor=1.2,
            confidence_level=0.68
        )

        assert len(ci) == 2

    def test_99_percent_confidence(self):
        """Test 99% confidence interval."""
        cycles = [{'range': 100, 'mean': 50, 'count': 100}]

        ci = calculate_confidence_interval(
            cycles=cycles,
            lifetime_model=lambda r, m, p: 1000,
            model_params={},
            scatter_factor=1.2,
            confidence_level=0.99
        )

        assert len(ci) == 2


class TestCombineDamageStates:
    """Test combining multiple damage states."""

    def test_simple_average(self):
        """Test simple averaging of damage states."""
        damages = [0.5, 0.7, 0.3]

        combined = combine_damage_states(damages)

        assert combined == pytest.approx(0.5)  # (0.5 + 0.7 + 0.3) / 3

    def test_weighted_average(self):
        """Test weighted averaging."""
        damages = [0.5, 0.7, 0.3]
        weights = [0.5, 0.3, 0.2]

        combined = combine_damage_states(damages, weights)

        expected = (0.5*0.5 + 0.7*0.3 + 0.3*0.2) / (0.5 + 0.3 + 0.2)
        assert combined == pytest.approx(expected)

    def test_empty_damage_states(self):
        """Test with empty damage list."""
        combined = combine_damage_states([])

        assert combined == 0.0

    def test_zero_weights(self):
        """Test with all zero weights."""
        damages = [0.5, 0.7, 0.3]
        weights = [0, 0, 0]

        combined = combine_damage_states(damages, weights)

        # Should fall back to simple average
        assert combined == pytest.approx(0.5)

    def test_mismatched_lengths_raises_error(self):
        """Test that mismatched lengths raise error."""
        damages = [0.5, 0.7]
        weights = [0.5]  # Only one weight

        with pytest.raises(ValueError, match="same length"):
            combine_damage_states(damages, weights)


class TestPredictTimeToFailure:
    """Test time to failure prediction."""

    def test_basic_prediction(self):
        """Test basic TTF prediction."""
        ttf = predict_time_to_failure(
            current_damage=0.5,
            damage_rate=0.01,
            time_unit='hours'
        )

        # TTF = (1 - 0.5) / 0.01 = 50 hours
        assert ttf == 50.0

    def test_no_damage(self):
        """Test TTF with no damage."""
        ttf = predict_time_to_failure(
            current_damage=0.0,
            damage_rate=0.01
        )

        assert ttf == 100.0  # 1.0 / 0.01

    def test_already_failed(self):
        """Test TTF when already failed."""
        ttf = predict_time_to_failure(
            current_damage=1.0,
            damage_rate=0.01
        )

        assert ttf == 0.0

    def test_zero_damage_rate(self):
        """Test TTF with zero damage rate."""
        ttf = predict_time_to_failure(
            current_damage=0.5,
            damage_rate=0.0
        )

        assert ttf == float('inf')

    def test_negative_damage_rate(self):
        """Test TTF with negative damage rate."""
        ttf = predict_time_to_failure(
            current_damage=0.5,
            damage_rate=-0.01
        )

        assert ttf == float('inf')


class TestAdjustForSequenceEffect:
    """Test sequence effect adjustment."""

    def test_no_correction(self):
        """Test with sequence_factor=1.0 (no correction)."""
        cycles = [{'range': 100, 'mean': 50, 'count': 100}]

        def model(range_val, mean_val, params):
            return 1000

        result = adjust_for_sequence_effect(
            cycles=cycles,
            lifetime_model=model,
            model_params={},
            sequence_factor=1.0
        )

        # Should be same as base calculation
        assert result.total_damage == 0.1

    def test_conservative_correction(self):
        """Test with conservative correction factor."""
        cycles = [{'range': 100, 'mean': 50, 'count': 100}]

        def model(range_val, mean_val, params):
            return 1000

        result = adjust_for_sequence_effect(
            cycles=cycles,
            lifetime_model=model,
            model_params={},
            sequence_factor=1.5  # 50% more damage
        )

        assert result.total_damage == 0.15  # 0.1 * 1.5

    def test_optimistic_correction(self):
        """Test with optimistic correction factor."""
        cycles = [{'range': 100, 'mean': 50, 'count': 100}]

        def model(range_val, mean_val, params):
            return 1000

        result = adjust_for_sequence_effect(
            cycles=cycles,
            lifetime_model=model,
            model_params={},
            sequence_factor=0.8  # 20% less damage
        )

        assert result.total_damage == 0.08  # 0.1 * 0.8

    def test_critical_after_adjustment(self):
        """Test that critical status updates after adjustment."""
        cycles = [{'range': 100, 'mean': 50, 'count': 1100}]

        def model(range_val, mean_val, params):
            return 1000

        result = adjust_for_sequence_effect(
            cycles=cycles,
            lifetime_model=model,
            model_params={},
            sequence_factor=1.0
        )

        # Base damage is 1.1, so should be critical
        assert result.is_critical is True


class TestDamageResult:
    """Test DamageResult dataclass."""

    def test_result_creation(self):
        """Test creating a DamageResult."""
        result = DamageResult(
            total_damage=0.5,
            remaining_life_fraction=0.5,
            is_critical=False,
            details=[]
        )

        assert result.total_damage == 0.5
        assert result.remaining_life_fraction == 0.5
        assert result.is_critical is False

    def test_result_repr(self):
        """Test DamageResult string representation."""
        result = DamageResult(
            total_damage=0.5,
            remaining_life_fraction=0.5,
            is_critical=False,
            details=[]
        )

        repr_str = repr(result)

        assert "DamageResult" in repr_str
        assert "0.5" in repr_str
        assert "OK" in repr_str

    def test_critical_repr(self):
        """Test string representation for critical damage."""
        result = DamageResult(
            total_damage=1.5,
            remaining_life_fraction=0.0,
            is_critical=True,
            details=[]
        )

        repr_str = repr(result)

        assert "CRITICAL" in repr_str


class TestRealWorldScenarios:
    """Test with realistic scenarios."""

    def test_power_module_thermal_cycling(self):
        """Test realistic power module thermal cycling damage."""
        # Simulate 3 types of thermal cycles
        cycles = [
            {'range': 80, 'mean': 90, 'count': 1000},   # Daily cycles
            {'range': 40, 'mean': 60, 'count': 5000},   # Minor fluctuations
            {'range': 100, 'mean': 110, 'count': 100},  # Extreme events
        ]

        def cips_model(range_val, mean_val, params):
            # Simplified CIPS-like model
            K = params.get('K', 1e10)
            return K * (range_val ** -4.5) * 100

        result = calculate_miner_damage(cycles, cips_model, {'K': 1e10})

        assert result.total_damage > 0
        assert isinstance(result.details, list)

    def test_miner_rule_accumulation(self):
        """Test classic Miner's rule accumulation.

        Miner's rule: D = Σ(ni/Ni)
        Failure occurs when D >= 1
        """
        # Design for exactly 1.0 damage
        cycles = [
            {'range': 100, 'mean': 50, 'count': 300},
            {'range': 80, 'mean': 40, 'count': 400},
            {'range': 60, 'mean': 30, 'count': 300},
        ]

        def model(range_val, mean_val, params):
            # Nf scales such that total damage = 1.0
            if range_val == 100:
                return 1000
            elif range_val == 80:
                return 800
            else:  # 60
                return 600

        result = calculate_miner_damage(cycles, model, {})

        # D = 300/1000 + 400/800 + 300/600 = 0.3 + 0.5 + 0.5 = 1.3
        assert result.total_damage == pytest.approx(1.3, rel=0.01)
        assert result.is_critical is True

    def test_low_damage_accumulation(self):
        """Test low damage accumulation scenario."""
        cycles = [
            {'range': 20, 'mean': 10, 'count': 100},
            {'range': 30, 'mean': 15, 'count': 50},
        ]

        def model(range_val, mean_val, params):
            return 100000  # High Nf (low stress)

        result = calculate_miner_damage(cycles, model, {})

        # Should be very low damage
        assert result.total_damage < 0.01
        assert result.remaining_life_fraction > 0.99

"""
Unit tests for safety margin calculation module.

Tests cover safety factor calculations, statistical margins,
partial safety factors, and design adequacy assessment.
"""
import pytest
import numpy as np
from app.core.safety_margin import (
    SafetyMarginResult,
    SafetyMarginDistribution,
    calculate_safety_margin,
    calculate_required_safety_factor,
    calculate_statistical_safety_margin,
    calculate_partial_safety_factors,
    combine_safety_factors,
    assess_design_adequacy,
    calculate_reserve_factor,
    calculate_damage_safety_margin
)


class TestCalculateSafetyMargin:
    """Tests for basic safety margin calculation."""

    def test_perfect_design(self):
        """Test with predicted life exactly equal to design life."""
        result = calculate_safety_margin(
            design_life=10000,
            predicted_life=10000,
            safety_factor=1.0
        )

        assert result.margin_percentage == 0.0
        assert result.is_acceptable
        assert result.utilization == 1.0

    def test_conservative_design(self):
        """Test with predicted life greater than design life."""
        result = calculate_safety_margin(
            design_life=10000,
            predicted_life=15000,
            safety_factor=1.0
        )

        assert result.margin_percentage == 50.0
        assert result.is_acceptable
        assert result.utilization < 1.0

    def test_inadequate_design(self):
        """Test with predicted life less than design life."""
        result = calculate_safety_margin(
            design_life=10000,
            predicted_life=8000,
            safety_factor=1.0
        )

        assert result.margin_percentage < 0
        assert not result.is_acceptable
        assert result.utilization > 1.0

    def test_with_safety_factor(self):
        """Test calculation with safety factor applied."""
        result = calculate_safety_margin(
            design_life=10000,
            predicted_life=15000,
            safety_factor=1.2
        )

        # Adjusted design = 10000 * 1.2 = 12000
        # Margin = (15000 / 12000 - 1) * 100 = 25%
        assert np.isclose(result.margin_percentage, 25.0, rtol=0.01)

    def test_minimum_acceptable_margin(self):
        """Test with minimum acceptable margin requirement."""
        result = calculate_safety_margin(
            design_life=10000,
            predicted_life=11000,
            safety_factor=1.0,
            minimum_acceptable_margin=20.0
        )

        # 10% margin < 20% required
        assert not result.is_acceptable

    def test_margin_value(self):
        """Test absolute margin value."""
        result = calculate_safety_margin(
            design_life=10000,
            predicted_life=15000,
            safety_factor=1.0
        )

        assert result.margin_value == 5000

    def test_invalid_inputs(self):
        """Test with invalid inputs."""
        with pytest.raises(ValueError):
            calculate_safety_margin(0, 10000, 1.0)

        with pytest.raises(ValueError):
            calculate_safety_margin(10000, 0, 1.0)

        with pytest.raises(ValueError):
            calculate_safety_margin(10000, 10000, 0)


class TestCalculateRequiredSafetyFactor:
    """Tests for required safety factor calculation."""

    def test_no_margin_required(self):
        """Test with target margin of 0%."""
        sf = calculate_required_safety_factor(
            design_life=10000,
            predicted_life=10000,
            target_margin=0.0
        )

        assert np.isclose(sf, 1.0)

    def test_positive_margin_required(self):
        """Test with positive target margin."""
        sf = calculate_required_safety_factor(
            design_life=10000,
            predicted_life=15000,
            target_margin=20.0
        )

        # Should be less than 1.0 (already have 50% margin)
        assert sf < 1.0

    def test_negative_margin_allowed(self):
        """Test with negative target margin."""
        sf = calculate_required_safety_factor(
            design_life=10000,
            predicted_life=8000,
            target_margin=-10.0
        )

        # SF should be less than 1 to allow -10% margin
        assert sf < 1.0

    def test_invalid_target_margin(self):
        """Test with invalid target margin."""
        with pytest.raises(ValueError):
            calculate_required_safety_factor(
                10000, 10000, target_margin=-150
            )


class TestCalculateStatisticalSafetyMargin:
    """Tests for statistical safety margin calculation."""

    def test_mean_margin_only(self):
        """Test with zero standard deviation."""
        result = calculate_statistical_safety_margin(
            design_life=10000,
            predicted_life_mean=15000,
            predicted_life_std=0,
            safety_factor=1.0
        )

        assert result.mean_margin == 50.0
        assert result.std_margin == 0.0
        assert result.probability_acceptable == 1.0

    def test_with_uncertainty(self):
        """Test with uncertainty in predicted life."""
        result = calculate_statistical_safety_margin(
            design_life=10000,
            predicted_life_mean=15000,
            predicted_life_std=3000,
            safety_factor=1.0
        )

        assert result.mean_margin == 50.0
        assert result.std_margin > 0
        assert result.percentile_5 < result.mean_margin
        assert result.percentile_95 > result.mean_margin

    def test_probability_acceptable(self):
        """Test probability of acceptability calculation."""
        # High uncertainty, low mean margin
        result = calculate_statistical_safety_margin(
            design_life=10000,
            predicted_life_mean=11000,
            predicted_life_std=2000,
            safety_factor=1.0,
            minimum_acceptable_margin=0.0
        )

        # Probability should be between 0 and 1
        assert 0 <= result.probability_acceptable <= 1

    def test_high_confidence_margin(self):
        """Test that 95th percentile is more conservative than mean."""
        result = calculate_statistical_safety_margin(
            design_life=10000,
            predicted_life_mean=15000,
            predicted_life_std=3000,
            safety_factor=1.0
        )

        # p5 should be less than mean, p95 should be greater
        assert result.percentile_5 < result.mean_margin
        assert result.percentile_95 > result.mean_margin


class TestCalculatePartialSafetyFactors:
    """Tests for partial safety factors calculation."""

    def test_load_factor(self):
        """Test partial safety factor for load."""
        contributions = {
            'load': {
                'coefficient_of_variation': 0.1,
                'sensitivity': 0.7
            }
        }

        factors = calculate_partial_safety_factors(
            contributions,
            target_reliability_index=3.0
        )

        # Load factor should be > 1 (conservative: increase design load)
        assert factors['load'] > 1.0

    def test_material_factor(self):
        """Test partial safety factor for material resistance."""
        contributions = {
            'material': {
                'coefficient_of_variation': 0.15,
                'sensitivity': 0.5
            }
        }

        factors = calculate_partial_safety_factors(
            contributions,
            target_reliability_index=3.0
        )

        # Material factor should be < 1 (conservative: reduce strength)
        assert factors['material'] < 1.0

    def test_multiple_factors(self):
        """Test multiple partial factors."""
        contributions = {
            'load': {'cov': 0.1, 'sensitivity': 0.7},
            'material': {'cov': 0.15, 'sensitivity': 0.5},
            'geometry': {'cov': 0.05, 'sensitivity': 0.3}
        }

        factors = calculate_partial_safety_factors(contributions)

        assert len(factors) == 3
        assert factors['load'] > 1.0
        assert factors['material'] < 1.0
        assert factors['geometry'] < 1.0

    def test_high_reliability(self):
        """Test with higher reliability target."""
        contributions = {
            'load': {'cov': 0.1, 'sensitivity': 0.7}
        }

        factors_beta_3 = calculate_partial_safety_factors(
            contributions, target_reliability_index=3.0
        )
        factors_beta_4 = calculate_partial_safety_factors(
            contributions, target_reliability_index=4.0
        )

        # Higher reliability index should give larger factor
        assert factors_beta_4['load'] > factors_beta_3['load']

    def test_invalid_sensitivity(self):
        """Test with invalid sensitivity values."""
        contributions = {
            'load': {'cov': 0.1, 'sensitivity': 1.5}  # Invalid: > 1
        }

        with pytest.raises(ValueError):
            calculate_partial_safety_factors(contributions)

    def test_invalid_cov(self):
        """Test with negative COV."""
        contributions = {
            'load': {'cov': -0.1, 'sensitivity': 0.7}
        }

        with pytest.raises(ValueError):
            calculate_partial_safety_factors(contributions)


class TestCombineSafetyFactors:
    """Tests for combining multiple safety factors."""

    def test_product_combination(self):
        """Test multiplicative combination."""
        factors = {'load': 1.2, 'material': 1.1, 'geometry': 1.05}

        combined = combine_safety_factors(factors, factor_type='product')

        # 1.2 * 1.1 * 1.05 = 1.386
        expected = 1.2 * 1.1 * 1.05
        assert np.isclose(combined, expected, rtol=0.01)

    def test_rss_combination(self):
        """Test root-sum-square combination."""
        factors = {'load': 1.2, 'material': 1.1}

        combined = combine_safety_factors(factors, factor_type='root_sum_square')

        # sqrt(1 + (0.2^2 + 0.1^2)) ≈ 1.0235
        expected = np.sqrt(1 + (0.2**2 + 0.1**2))
        assert np.isclose(combined, expected, rtol=0.01)

    def test_max_combination(self):
        """Test maximum combination."""
        factors = {'load': 1.2, 'material': 1.1, 'geometry': 1.05}

        combined = combine_safety_factors(factors, factor_type='max')

        assert combined == 1.2

    def test_empty_factors(self):
        """Test with empty factor dict."""
        combined = combine_safety_factors({}, factor_type='product')
        assert combined == 1.0

    def test_invalid_combination_type(self):
        """Test with invalid combination type."""
        factors = {'load': 1.2}

        with pytest.raises(ValueError):
            combine_safety_factors(factors, factor_type='invalid')


class TestAssessDesignAdequacy:
    """Tests for design adequacy assessment."""

    def test_inadequate_design(self):
        """Test assessment of inadequate design."""
        margin_result = SafetyMarginResult(
            safety_factor=1.0,
            design_life_cycles=10000,
            predicted_life_cycles=8000,
            margin_percentage=-20.0,
            is_acceptable=False,
            margin_value=-2000,
            utilization=1.25
        )

        assessment = assess_design_adequacy(margin_result)

        assert assessment['adequacy_level'] == "INADEQUATE"
        assert assessment['color_code'] == "red"
        assert assessment['margin_category'] == "critical"

    def test_marginal_design(self):
        """Test assessment of marginal design."""
        margin_result = SafetyMarginResult(
            safety_factor=1.0,
            design_life_cycles=10000,
            predicted_life_cycles=10500,
            margin_percentage=5.0,
            is_acceptable=True,
            margin_value=500,
            utilization=0.95
        )

        assessment = assess_design_adequacy(margin_result)

        assert assessment['adequacy_level'] == "MARGINAL"
        assert assessment['color_code'] == "yellow"

    def test_adequate_design(self):
        """Test assessment of adequate design."""
        margin_result = SafetyMarginResult(
            safety_factor=1.0,
            design_life_cycles=10000,
            predicted_life_cycles=12000,
            margin_percentage=20.0,
            is_acceptable=True,
            margin_value=2000,
            utilization=0.83
        )

        assessment = assess_design_adequacy(margin_result)

        assert assessment['adequacy_level'] == "ADEQUATE"
        assert assessment['color_code'] == "green"

    def test_good_design(self):
        """Test assessment of good design."""
        margin_result = SafetyMarginResult(
            safety_factor=1.0,
            design_life_cycles=10000,
            predicted_life_cycles=15000,
            margin_percentage=50.0,
            is_acceptable=True,
            margin_value=5000,
            utilization=0.67
        )

        assessment = assess_design_adequacy(margin_result)

        assert assessment['adequacy_level'] == "GOOD"
        assert assessment['margin_category'] == "high"

    def test_excellent_design(self):
        """Test assessment of excellent design."""
        margin_result = SafetyMarginResult(
            safety_factor=1.0,
            design_life_cycles=10000,
            predicted_life_cycles=20000,
            margin_percentage=100.0,
            is_acceptable=True,
            margin_value=10000,
            utilization=0.5
        )

        assessment = assess_design_adequacy(margin_result)

        assert assessment['adequacy_level'] == "EXCELLENT"
        assert assessment['margin_category'] == "very_high"

    def test_with_inspection_interval(self):
        """Test assessment with inspection interval."""
        margin_result = SafetyMarginResult(
            safety_factor=1.0,
            design_life_cycles=10000,
            predicted_life_cycles=15000,
            margin_percentage=50.0,
            is_acceptable=True,
            margin_value=5000,
            utilization=0.67
        )

        assessment = assess_design_adequacy(margin_result, inspection_interval=1000)

        assert 'inspection_recommendation' in assessment
        assert assessment['inspection_recommendation'] is not None


class TestCalculateReserveFactor:
    """Tests for reserve factor calculation."""

    def test_safe_design(self):
        """Test reserve factor for safe design."""
        rf = calculate_reserve_factor(allowable_value=150, applied_value=100)

        assert np.isclose(rf, 1.5)

    def test_exact_design(self):
        """Test reserve factor at limit."""
        rf = calculate_reserve_factor(allowable_value=100, applied_value=100)

        assert np.isclose(rf, 1.0)

    def test_unsafe_design(self):
        """Test reserve factor for unsafe design."""
        rf = calculate_reserve_factor(allowable_value=80, applied_value=100)

        assert rf < 1.0

    def test_zero_applied_value(self):
        """Test with zero applied value."""
        with pytest.raises(ValueError):
            calculate_reserve_factor(100, 0)


class TestCalculateDamageSafetyMargin:
    """Tests for damage-based safety margin."""

    def test_no_damage(self):
        """Test with zero current damage."""
        margin = calculate_damage_safety_margin(current_damage=0.0)

        assert margin == 1.0

    def test_half_damage(self):
        """Test with 50% damage."""
        margin = calculate_damage_safety_margin(current_damage=0.5)

        assert margin == 0.5

    def test_full_damage(self):
        """Test with 100% damage."""
        margin = calculate_damage_safety_margin(current_damage=1.0)

        assert margin == 0.0

    def test_excessive_damage(self):
        """Test with damage > 1.0."""
        margin = calculate_damage_safety_margin(current_damage=1.2)

        assert margin == 0.0

    def test_negative_damage(self):
        """Test with negative damage."""
        margin = calculate_damage_safety_margin(current_damage=-0.1)

        assert margin == 1.0

    def test_custom_critical_damage(self):
        """Test with custom critical damage level."""
        margin = calculate_damage_safety_margin(
            current_damage=0.5,
            critical_damage=0.8
        )

        # (1 - 0.5/0.8) = 0.375
        expected = 1 - (0.5 / 0.8)
        assert np.isclose(margin, expected)

    def test_zero_critical_damage(self):
        """Test with zero critical damage."""
        with pytest.raises(ValueError):
            calculate_damage_safety_margin(0.5, critical_damage=0)


class TestSafetyMarginResult:
    """Tests for SafetyMarginResult dataclass."""

    def test_result_creation(self):
        """Test SafetyMarginResult creation."""
        result = SafetyMarginResult(
            safety_factor=1.2,
            design_life_cycles=10000,
            predicted_life_cycles=15000,
            margin_percentage=25.0,
            is_acceptable=True,
            margin_value=5000,
            utilization=0.8
        )

        assert result.safety_factor == 1.2
        assert result.design_life_cycles == 10000
        assert result.predicted_life_cycles == 15000
        assert result.margin_percentage == 25.0
        assert result.is_acceptable is True

    def test_result_repr(self):
        """Test SafetyMarginResult string representation."""
        result = SafetyMarginResult(
            safety_factor=1.0,
            design_life_cycles=10000,
            predicted_life_cycles=15000,
            margin_percentage=50.0,
            is_acceptable=True,
            margin_value=5000,
            utilization=0.67
        )

        repr_str = repr(result)
        assert 'SafetyMarginResult' in repr_str
        assert 'ACCEPTABLE' in repr_str


class TestSafetyMarginDistribution:
    """Tests for SafetyMarginDistribution dataclass."""

    def test_distribution_creation(self):
        """Test SafetyMarginDistribution creation."""
        dist = SafetyMarginDistribution(
            mean_margin=50.0,
            std_margin=10.0,
            percentile_5=35.0,
            percentile_95=65.0,
            probability_acceptable=0.95
        )

        assert dist.mean_margin == 50.0
        assert dist.std_margin == 10.0
        assert dist.percentile_5 == 35.0
        assert dist.percentile_95 == 65.0
        assert dist.probability_acceptable == 0.95

"""
Unit tests for Weibull analysis module.

Tests cover:
- Distribution fitting with various data scenarios
- B-life calculations
- Probability plot generation
- Reliability and hazard rate calculations
- Error handling for edge cases
"""

import pytest
import numpy as np
from numpy.testing import assert_allclose

from app.core.weibull import (
    fit_weibull,
    calculate_b_life,
    weibull_probability_plot_data,
    calculate_reliability,
    calculate_hazard_rate,
    weibull_pdf,
    weibull_cdf,
    WeibullResult,
    WeibullAnalysisError
)


class TestCalculateBLife:
    """Test B-life calculation function."""

    def test_b10_life(self):
        """Test B10 life calculation."""
        # For beta=2, eta=1000, B10 should be approximately 316
        result = calculate_b_life(shape=2.0, scale=1000, percentile=0.1)
        expected = 1000 * (-np.log(0.9)) ** 0.5
        assert_allclose(result, expected, rtol=1e-10)

    def test_b50_life(self):
        """Test B50 (median) life calculation."""
        result = calculate_b_life(shape=2.0, scale=1000, percentile=0.5)
        expected = 1000 * (-np.log(0.5)) ** 0.5
        assert_allclose(result, expected, rtol=1e-10)

    def test_b63_2_life_equals_scale(self):
        """Test that B63.2 life equals scale parameter by definition."""
        result = calculate_b_life(shape=2.0, scale=1000, percentile=0.632)
        assert_allclose(result, 1000, rtol=0.01)

    def test_invalid_shape_parameter(self):
        """Test that non-positive shape raises error."""
        with pytest.raises(ValueError, match="Shape parameter must be positive"):
            calculate_b_life(shape=0, scale=1000, percentile=0.1)

    def test_invalid_scale_parameter(self):
        """Test that non-positive scale raises error."""
        with pytest.raises(ValueError, match="Scale parameter must be positive"):
            calculate_b_life(shape=2.0, scale=0, percentile=0.1)

    def test_invalid_percentile(self):
        """Test that out-of-range percentile raises error."""
        with pytest.raises(ValueError, match="Percentile must be between 0 and 1"):
            calculate_b_life(shape=2.0, scale=1000, percentile=1.5)

        with pytest.raises(ValueError, match="Percentile must be between 0 and 1"):
            calculate_b_life(shape=2.0, scale=1000, percentile=-0.1)

    def test_shape_affects_b_life(self):
        """Test that shape parameter affects B-life ranking."""
        b10_beta1 = calculate_b_life(shape=1.0, scale=1000, percentile=0.1)
        b10_beta3 = calculate_b_life(shape=3.0, scale=1000, percentile=0.1)

        # For beta > 1, failures are more concentrated, so B10 should be higher
        assert b10_beta3 > b10_beta1


class TestCalculateReliability:
    """Test reliability calculation function."""

    def test_reliability_at_zero_time(self):
        """Test reliability is 100% at time zero."""
        r = calculate_reliability(shape=2.0, scale=1000, time=0)
        assert_allclose(r, 1.0)

    def test_reliarity_decreases_with_time(self):
        """Test that reliability decreases as time increases."""
        r1 = calculate_reliability(shape=2.0, scale=1000, time=100)
        r2 = calculate_reliability(shape=2.0, scale=1000, time=500)
        r3 = calculate_reliability(shape=2.0, scale=1000, time=1000)

        assert r1 > r2 > r3

    def test_reliability_at_characteristic_life(self):
        """Test reliability at characteristic life (eta)."""
        # At t = eta, reliability should be exp(-1) ≈ 36.8%
        r = calculate_reliability(shape=2.0, scale=1000, time=1000)
        expected = np.exp(-1)
        assert_allclose(r, expected, rtol=1e-10)

    def test_reliability_with_location_parameter(self):
        """Test reliability with non-zero location parameter."""
        # With location=100, reliability should be 1 for t < 100
        r_before = calculate_reliability(shape=2.0, scale=1000, time=50, location=100)
        assert r_before == 1.0

        r_at = calculate_reliability(shape=2.0, scale=1000, time=100, location=100)
        assert r_at > 0.9  # Should be close to 1 but slightly less

    def test_invalid_parameters(self):
        """Test error handling for invalid parameters."""
        with pytest.raises(ValueError):
            calculate_reliability(shape=0, scale=1000, time=100)

        with pytest.raises(ValueError):
            calculate_reliability(shape=2.0, scale=-1, time=100)


class TestCalculateHazardRate:
    """Test hazard rate calculation function."""

    def test_hazard_rate_for_beta_1(self):
        """Test hazard rate is constant for beta=1 (exponential distribution)."""
        h1 = calculate_hazard_rate(shape=1.0, scale=1000, time=100)
        h2 = calculate_hazard_rate(shape=1.0, scale=1000, time=500)
        assert_allclose(h1, h2, rtol=1e-10)

    def test_hazard_rate_increases_for_beta_gt_1(self):
        """Test hazard rate increases for beta > 1 (wear-out)."""
        h1 = calculate_hazard_rate(shape=2.0, scale=1000, time=100)
        h2 = calculate_hazard_rate(shape=2.0, scale=1000, time=500)
        assert h2 > h1

    def test_hazard_rate_zero_before_location(self):
        """Test hazard rate is zero before location parameter."""
        h = calculate_hazard_rate(shape=2.0, scale=1000, time=50, location=100)
        assert h == 0.0


class TestFitWeibull:
    """Test Weibull distribution fitting."""

    def test_fit_with_simulated_data(self):
        """Test fitting with simulated Weibull data."""
        np.random.seed(42)
        true_shape = 2.5
        true_scale = 1000

        # Generate data from known Weibull distribution
        data = np.random.weibull(true_shape, 50) * true_scale

        result = fit_weibull(data.tolist())

        # Check that fitted parameters are reasonably close
        assert result.shape > 1.5 and result.shape < 3.5
        assert result.scale > 700 and result.scale < 1300
        assert result.location == 0.0

    def test_fit_returns_result_object(self):
        """Test that fit_weibull returns proper result object."""
        data = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
        result = fit_weibull(data)

        assert isinstance(result, WeibullResult)
        assert hasattr(result, 'shape')
        assert hasattr(result, 'scale')
        assert hasattr(result, 'location')
        assert hasattr(result, 'r_squared')
        assert hasattr(result, 'b10_life')
        assert hasattr(result, 'b50_life')
        assert hasattr(result, 'b63_life')
        assert hasattr(result, 'mttf')

    def test_fit_with_censored_data(self):
        """Test fitting with censored (suspension) data."""
        failures = [500, 600, 700, 800, 900]
        censored = [1000, 1000, 1000]  # Units that survived to 1000

        result = fit_weibull(failures, censored)

        assert result.shape > 0
        assert result.scale > 0
        assert result.b10_life < result.b50_life < result.b63_life

    def test_fit_empty_data_raises_error(self):
        """Test that empty data raises error."""
        with pytest.raises(WeibullAnalysisError, match="cannot be empty"):
            fit_weibull([])

    def test_fit_single_point_raises_error(self):
        """Test that single data point raises error."""
        with pytest.raises(WeibullAnalysisError, match="At least 2"):
            fit_weibull([100])

    def test_fit_negative_values_raises_error(self):
        """Test that negative values raise error."""
        with pytest.raises(WeibullAnalysisError, match="must be positive"):
            fit_weibull([100, -50, 200])

    def test_monotonic_b_life_values(self):
        """Test that B10 < B50 < B63.2 for typical data."""
        data = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
        result = fit_weibull(data)

        assert result.b10_life < result.b50_life < result.b63_life

    def test_r_squared_in_valid_range(self):
        """Test that R-squared is in valid range [0, 1]."""
        data = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
        result = fit_weibull(data)

        assert 0 <= result.r_squared <= 1

    def test_mttf_reasonable_for_shape_gt_1(self):
        """Test that MTTF is reasonable for shape > 1."""
        # For beta > 1, MTTF should be somewhat less than eta
        data = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
        result = fit_weibull(data)

        # MTTF should be positive
        assert result.mttf > 0


class TestProbabilityPlotData:
    """Test Weibull probability plot data generation."""

    def test_returns_correct_shape(self):
        """Test that output arrays have correct shape."""
        data = [100, 200, 300, 400, 500]
        x, y = weibull_probability_plot_data(data)

        assert len(x) == len(data)
        assert len(y) == len(data)
        assert isinstance(x, np.ndarray)
        assert isinstance(y, np.ndarray)

    def test_x_values_are_log_sorted(self):
        """Test that x-values are log of sorted data."""
        data = [500, 100, 300, 200, 400]
        x, y = weibull_probability_plot_data(data)

        expected_x = np.log(np.sort(data))
        assert_allclose(x, expected_x)

    def test_empty_data_raises_error(self):
        """Test that empty data raises error."""
        with pytest.raises(WeibullAnalysisError):
            weibull_probability_plot_data([])

    def test_single_point_raises_error(self):
        """Test that single point raises error."""
        with pytest.raises(WeibullAnalysisError, match="At least 2"):
            weibull_probability_plot_data([100])


class TestPDFandCDF:
    """Test probability density and cumulative distribution functions."""

    def test_cdf_at_zero(self):
        """Test CDF at time zero (should be 0)."""
        cdf = weibull_cdf(shape=2.0, scale=1000, time=0)
        assert_allclose(cdf, 0.0)

    def test_cdf_at_characteristic_life(self):
        """Test CDF at characteristic life (should be 0.632)."""
        cdf = weibull_cdf(shape=2.0, scale=1000, time=np.array([1000]))
        assert_allclose(cdf[0], 1 - np.exp(-1), rtol=1e-10)

    def test_cdf_monotonic_increasing(self):
        """Test that CDF is monotonically increasing."""
        times = np.array([100, 500, 1000, 1500, 2000])
        cdf_values = weibull_cdf(shape=2.0, scale=1000, time=times)

        for i in range(len(cdf_values) - 1):
            assert cdf_values[i] < cdf_values[i + 1]

    def test_pdf_positive(self):
        """Test that PDF is positive for valid times."""
        times = np.linspace(1, 2000, 100)
        pdf_values = weibull_pdf(shape=2.0, scale=1000, time=times)

        assert np.all(pdf_values >= 0)

    def test_cdf_plus_reliability_equals_one(self):
        """Test that CDF(t) + R(t) = 1."""
        times = np.array([100, 500, 1000])
        cdf = weibull_cdf(shape=2.0, scale=1000, time=times)

        for i, t in enumerate(times):
            r = calculate_reliability(shape=2.0, scale=1000, time=t)
            assert_allclose(cdf[i] + r, 1.0, rtol=1e-10)


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_very_small_shape(self):
        """Test behavior with very small shape (early failure mode)."""
        b_life = calculate_b_life(shape=0.5, scale=1000, percentile=0.1)
        assert b_life > 0

    def test_very_large_shape(self):
        """Test behavior with very large shape (sharp wear-out)."""
        b_life = calculate_b_life(shape=5.0, scale=1000, percentile=0.1)
        assert b_life > 0

    def test_large_scale_parameter(self):
        """Test with large scale parameter."""
        b_life = calculate_b_life(shape=2.0, scale=1e6, percentile=0.1)
        assert b_life > 1e5

    def test_very_small_scale_parameter(self):
        """Test with very small scale parameter."""
        b_life = calculate_b_life(shape=2.0, scale=1.0, percentile=0.1)
        assert b_life < 1.0

    def test_identical_data_points(self):
        """Test fitting with identical data points."""
        data = [500, 500, 500, 500, 500]
        # This should still work, though results may be degenerate
        result = fit_weibull(data)
        assert result is not None


@pytest.fixture
def sample_weibull_data():
    """Fixture providing sample Weibull-distributed data."""
    np.random.seed(123)
    shape = 2.0
    scale = 1000
    return np.random.weibull(shape, 30) * scale


@pytest.fixture
def sample_result(sample_weibull_data):
    """Fixture providing a fitted Weibull result."""
    return fit_weibull(sample_weibull_data.tolist())


class TestIntegration:
    """Integration tests for complete workflows."""

    def test_complete_analysis_workflow(self, sample_weibull_data):
        """Test complete Weibull analysis workflow."""
        # Fit the data
        result = fit_weibull(sample_weibull_data.tolist())

        # Get probability plot data
        x, y = weibull_probability_plot_data(sample_weibull_data.tolist())

        # Calculate reliability at various points
        times = [result.b10_life, result.b50_life, result.b63_life]
        reliabilities = [calculate_reliability(result.shape, result.scale, t) for t in times]

        # Verify consistency
        assert_allclose(reliabilities[0], 0.9, atol=0.05)  # ~90% at B10
        assert_allclose(reliabilities[1], 0.5, atol=0.05)  # ~50% at B50
        assert_allclose(reliabilities[2], 0.368, atol=0.05)  # ~36.8% at B63

    def test_consistency_between_methods(self, sample_result):
        """Test consistency between different calculation methods."""
        # B63.2 should equal scale
        assert_allclose(sample_result.b63_life, sample_result.scale, rtol=0.01)

        # Reliability at B63 should be ~37%
        r_at_b63 = calculate_reliability(sample_result.shape, sample_result.scale, sample_result.b63_life)
        assert_allclose(r_at_b63, np.exp(-1), atol=0.05)

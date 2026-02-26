"""
Unit tests for model fitting module.

Tests cover:
- Non-linear least squares fitting
- Statistical metrics calculation
- CIPS 2008 specialized fitting
- Weighted and robust fitting
- Error handling
"""

import pytest
import numpy as np
from numpy.testing import assert_allclose
from unittest.mock import MagicMock

from app.core.fitting import (
    fit_lifetime_model,
    fit_cips2008_model,
    weighted_fit,
    robust_fit,
    calculate_r_squared,
    calculate_rmse,
    calculate_mae,
    calculate_mape,
    FittingResult,
    FittingError
)


class TestCalculateRSquared:
    """Test R-squared calculation."""

    def test_perfect_fit(self):
        """Test R-squared for perfect fit (should be 1)."""
        y_actual = np.array([1, 2, 3, 4, 5])
        y_predicted = np.array([1, 2, 3, 4, 5])
        r2 = calculate_r_squared(y_actual, y_predicted)
        assert_allclose(r2, 1.0)

    def test_good_fit(self):
        """Test R-squared for good fit."""
        y_actual = np.array([1, 2, 3, 4, 5])
        y_predicted = np.array([1.1, 1.9, 3.1, 4.0, 5.1])
        r2 = calculate_r_squared(y_actual, y_predicted)
        assert r2 > 0.95

    def test_poor_fit(self):
        """Test R-squared for poor fit."""
        y_actual = np.array([1, 2, 3, 4, 5])
        y_predicted = np.array([5, 4, 3, 2, 1])
        r2 = calculate_r_squared(y_actual, y_predicted)
        assert r2 < 0

    def test_constant_actual_values(self):
        """Test R-squared when actual values are constant."""
        y_actual = np.array([5, 5, 5, 5, 5])
        y_predicted = np.array([5, 5, 5, 5, 5])
        r2 = calculate_r_squared(y_actual, y_predicted)
        assert r2 == 1.0


class TestCalculateRMSE:
    """Test RMSE calculation."""

    def test_zero_error(self):
        """Test RMSE for perfect predictions."""
        y_actual = np.array([10, 20, 30])
        y_predicted = np.array([10, 20, 30])
        rmse = calculate_rmse(y_actual, y_predicted)
        assert_allclose(rmse, 0.0)

    def test_uniform_errors(self):
        """Test RMSE for uniform errors."""
        y_actual = np.array([10, 20, 30])
        y_predicted = np.array([12, 22, 32])
        rmse = calculate_rmse(y_actual, y_predicted)
        assert_allclose(rmse, 2.0)

    def test_rmse_units(self):
        """Test that RMSE has same units as input."""
        y_actual = np.array([100, 200, 300])
        y_predicted = np.array([110, 190, 310])
        rmse = calculate_rmse(y_actual, y_predicted)
        # RMSE should be in same scale as data
        assert 0 < rmse < 100


class TestCalculateMAE:
    """Test Mean Absolute Error calculation."""

    def test_zero_error(self):
        """Test MAE for perfect predictions."""
        y_actual = np.array([10, 20, 30])
        y_predicted = np.array([10, 20, 30])
        mae = calculate_mae(y_actual, y_predicted)
        assert_allclose(mae, 0.0)

    def test_uniform_absolute_errors(self):
        """Test MAE for uniform absolute errors."""
        y_actual = np.array([10, 20, 30])
        y_predicted = np.array([12, 22, 32])
        mae = calculate_mae(y_actual, y_predicted)
        assert_allclose(mae, 2.0)

    def test_mae_less_sensitive_to_outliers(self):
        """Test that MAE is less sensitive to outliers than RMSE."""
        y_actual = np.array([1, 2, 3, 4, 5])
        y_predicted_good = np.array([1.1, 1.9, 3.1, 4.0, 5.1])
        y_predicted_bad = np.array([1.1, 1.9, 3.1, 4.0, 20])  # One outlier

        mae_good = calculate_mae(y_actual, y_predicted_good)
        mae_bad = calculate_mae(y_actual, y_predicted_bad)
        rmse_good = calculate_rmse(y_actual, y_predicted_good)
        rmse_bad = calculate_rmse(y_actual, y_predicted_bad)

        # MAE increases less than RMSE with outlier
        assert (mae_bad / mae_good) < (rmse_bad / rmse_good)


class TestCalculateMAPE:
    """Test Mean Absolute Percentage Error calculation."""

    def test_zero_error(self):
        """Test MAPE for perfect predictions."""
        y_actual = np.array([10, 20, 30])
        y_predicted = np.array([10, 20, 30])
        mape = calculate_mape(y_actual, y_predicted)
        assert_allclose(mape, 0.0)

    def test_percentage_scale(self):
        """Test that MAPE is in percentage."""
        y_actual = np.array([100, 200, 300])
        y_predicted = np.array([110, 190, 310])
        mape = calculate_mape(y_actual, y_predicted)
        # Should be around 5%
        assert 0 < mape < 10


class TestFitLifetimeModel:
    """Test general lifetime model fitting."""

    def test_linear_model_fitting(self):
        """Test fitting a simple linear model."""
        def linear_model(x, slope, intercept):
            return slope * x + intercept

        x_data = np.array([1, 2, 3, 4, 5])
        y_data = np.array([2.1, 3.9, 6.1, 8.0, 10.0])  # Approx y = 2x

        initial_params = {'slope': 1.0, 'intercept': 0.0}
        result = fit_lifetime_model(linear_model, x_data, y_data, initial_params)

        # Check fitted parameters are close to true values
        assert_allclose(result.parameters['slope'], 2.0, atol=0.2)
        assert_allclose(result.parameters['intercept'], 0.0, atol=0.5)

        # Check high R-squared for good fit
        assert result.r_squared > 0.95

    def test_exponential_model_fitting(self):
        """Test fitting an exponential decay model."""
        def exponential_model(x, A, k):
            return A * np.exp(-k * x)

        x_data = np.linspace(0, 5, 20)
        true_A, true_k = 100, 0.5
        y_data = true_A * np.exp(-true_k * x_data) + np.random.normal(0, 2, len(x_data))

        initial_params = {'A': 100, 'k': 0.1}
        result = fit_lifetime_model(exponential_model, x_data, y_data, initial_params)

        assert_allclose(result.parameters['A'], true_A, atol=20)
        assert_allclose(result.parameters['k'], true_k, atol=0.2)

    def test_fitting_with_bounds(self):
        """Test fitting with parameter bounds."""
        def linear_model(x, slope, intercept):
            return slope * x + intercept

        x_data = np.array([1, 2, 3, 4, 5])
        y_data = np.array([2, 4, 6, 8, 10])

        initial_params = {'slope': 1.0, 'intercept': 0.0}
        bounds = {'slope': (0, 10), 'intercept': (-5, 5)}

        result = fit_lifetime_model(linear_model, x_data, y_data, initial_params, bounds)

        # Check parameters are within bounds
        assert 0 <= result.parameters['slope'] <= 10
        assert -5 <= result.parameters['intercept'] <= 5

    def test_returns_result_object(self):
        """Test that fitting returns proper result object."""
        def linear_model(x, slope):
            return slope * x

        x_data = np.array([1, 2, 3])
        y_data = np.array([2, 4, 6])
        initial_params = {'slope': 1.0}

        result = fit_lifetime_model(linear_model, x_data, y_data, initial_params)

        assert isinstance(result, FittingResult)
        assert hasattr(result, 'parameters')
        assert hasattr(result, 'std_errors')
        assert hasattr(result, 'r_squared')
        assert hasattr(result, 'rmse')
        assert hasattr(result, 'residuals')
        assert hasattr(result, 'covariance')
        assert hasattr(result, 'confidence_intervals')

    def test_mismatched_data_lengths(self):
        """Test that mismatched data lengths raise error."""
        def model(x, a):
            return a * x

        x_data = np.array([1, 2, 3])
        y_data = np.array([1, 2])  # Different length

        with pytest.raises(FittingError, match="same length"):
            fit_lifetime_model(model, x_data, y_data, {'a': 1})

    def test_insufficient_data(self):
        """Test that insufficient data raises error."""
        def model(x, a):
            return a * x

        x_data = np.array([1])  # Only one point
        y_data = np.array([2])

        with pytest.raises(FittingError, match="At least 2"):
            fit_lifetime_model(model, x_data, y_data, {'a': 1})


class TestFitCIPS2008Model:
    """Test CIPS 2008 specialized fitting."""

    def test_cips2008_model_structure(self):
        """Test CIPS 2008 model fitting with valid data."""
        # Create synthetic data with known parameters
        data = []
        base_Nf = 1e6

        for dTj in [60, 80, 100, 120]:
            for Tj_max in [100, 125, 150]:
                row = {
                    'dTj': dTj,
                    'Tj_max': Tj_max,
                    't_on': 1.0,
                    'I': 1.0,
                    'V': 1.0,
                    'D': 0.5,
                    'Nf': base_Nf * (80 / dTj) ** 2 * np.exp(1500 / (125 + 273.15) - 1500 / (Tj_max + 273.15))
                }
                data.append(row)

        result = fit_cips2008_model(data)

        # Check that fitting succeeded
        assert result is not None
        assert 'K' in result.parameters
        assert 'β1' in result.parameters
        assert 'β2' in result.parameters

    def test_cips2008_with_fixed_params(self):
        """Test CIPS 2008 fitting with some parameters fixed."""
        data = [
            {'dTj': 80, 'Tj_max': 125, 't_on': 1, 'I': 1, 'V': 1, 'D': 0.5, 'Nf': 1e6},
            {'dTj': 100, 'Tj_max': 125, 't_on': 1, 'I': 1, 'V': 1, 'D': 0.5, 'Nf': 5e5},
            {'dTj': 80, 'Tj_max': 150, 't_on': 1, 'I': 1, 'V': 1, 'D': 0.5, 'Nf': 8e5},
            {'dTj': 100, 'Tj_max': 150, 't_on': 1, 'I': 1, 'V': 1, 'D': 0.5, 'Nf': 4e5},
            {'dTj': 80, 'Tj_max': 100, 't_on': 1, 'I': 1, 'V': 1, 'D': 0.5, 'Nf': 1.2e6},
            {'dTj': 100, 'Tj_max': 100, 't_on': 1, 'I': 1, 'V': 1, 'D': 0.5, 'Nf': 6e5},
            {'dTj': 80, 'Tj_max': 175, 't_on': 1, 'I': 1, 'V': 1, 'D': 0.5, 'Nf': 6e5},
            {'dTj': 100, 'Tj_max': 175, 't_on': 1, 'I': 1, 'V': 1, 'D': 0.5, 'Nf': 3e5},
            {'dTj': 90, 'Tj_max': 140, 't_on': 1, 'I': 1, 'V': 1, 'D': 0.5, 'Nf': 9e5},
        ]

        fixed_params = {'β3': -0.5, 'β4': -1.5, 'β5': -1.0, 'β6': -0.3}

        result = fit_cips2008_model(data, fixed_params)

        # Fixed parameters should be in results
        assert 'β3' in result.parameters
        assert result.parameters['β3'] == -0.5
        assert 'K' in result.parameters
        assert 'β1' in result.parameters

    def test_empty_data_raises_error(self):
        """Test that empty data raises error."""
        with pytest.raises(FittingError, match="cannot be empty"):
            fit_cips2008_model([])

    def test_insufficient_data_points(self):
        """Test that insufficient data raises error."""
        # Only 6 points, need at least 7 for full parameter fitting
        data = [
            {'dTj': 80, 'Tj_max': 125, 't_on': 1, 'I': 1, 'V': 1, 'D': 0.5, 'Nf': 1e6},
        ] * 6

        with pytest.raises(FittingError, match="At least .* data points"):
            fit_cips2008_model(data)

    def test_missing_required_field(self):
        """Test that missing required field raises error."""
        data = [
            {'dTj': 80, 'Tj_max': 125, 'I': 1, 'V': 1, 'D': 0.5, 'Nf': 1e6},
            # Missing 't_on'
        ] * 7

        with pytest.raises(FittingError, match="Missing required field"):
            fit_cips2008_model(data)


class TestWeightedFit:
    """Test weighted least squares fitting."""

    def test_weighted_fit_with_varying_errors(self):
        """Test weighted fitting with varying measurement errors."""
        def linear_model(x, slope, intercept):
            return slope * x + intercept

        x_data = np.array([1, 2, 3, 4, 5])
        y_data = np.array([2, 4, 6, 8, 10])

        # Varying errors - last point has larger error
        y_errors = np.array([0.1, 0.1, 0.1, 0.1, 1.0])

        initial_params = {'slope': 1.0, 'intercept': 0.0}
        result = weighted_fit(linear_model, x_data, y_data, y_errors, initial_params)

        # Fit should still succeed
        assert result.parameters['slope'] > 1.5  # Should be close to 2

    def test_weighted_fit_error_handling(self):
        """Test error handling for weighted fit."""
        def model(x, a):
            return a * x

        x_data = np.array([1, 2, 3])
        y_data = np.array([2, 4, 6])

        # Mismatched error array length
        y_errors = np.array([1, 2])  # Wrong length

        with pytest.raises(FittingError, match="same length"):
            weighted_fit(model, x_data, y_data, y_errors, {'a': 1})

    def test_zero_errors_raise_error(self):
        """Test that zero errors raise error (would cause infinite weight)."""
        def model(x, a):
            return a * x

        x_data = np.array([1, 2, 3])
        y_data = np.array([2, 4, 6])
        y_errors = np.array([1, 0, 1])  # One zero error

        with pytest.raises(FittingError, match="must be positive"):
            weighted_fit(model, x_data, y_data, y_errors, {'a': 1})


class TestRobustFit:
    """Test robust fitting with outlier rejection."""

    def test_robust_fit_with_outliers(self):
        """Test robust fitting handles outliers better than standard fit."""
        def linear_model(x, slope, intercept):
            return slope * x + intercept

        x_data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        y_data = np.array([2, 4, 6, 8, 10, 12, 14, 16, 18, 100])  # Last point is outlier

        initial_params = {'slope': 1.0, 'intercept': 0.0}

        result = robust_fit(linear_model, x_data, y_data, initial_params)

        # Robust fit should be less affected by outlier
        # Slope should be closer to true value of 2
        assert result.parameters['slope'] > 1.5
        assert result.parameters['slope'] < 3

    def test_robust_fit_without_outliers(self):
        """Test robust fit works well even without outliers."""
        def linear_model(x, slope, intercept):
            return slope * x + intercept

        x_data = np.array([1, 2, 3, 4, 5])
        y_data = np.array([2, 4, 6, 8, 10])

        initial_params = {'slope': 1.0, 'intercept': 0.0}

        result = robust_fit(linear_model, x_data, y_data, initial_params)

        # Should still get good fit
        assert_allclose(result.parameters['slope'], 2.0, atol=0.5)


class TestConfidenceIntervals:
    """Test confidence interval calculations."""

    def test_confidence_intervals_exist(self):
        """Test that confidence intervals are calculated."""
        def linear_model(x, slope):
            return slope * x

        # Add some noise to avoid degenerate confidence intervals
        np.random.seed(42)
        x_data = np.array([1, 2, 3, 4, 5, 6, 7, 8])
        y_data = x_data * 2 + np.random.normal(0, 0.1, len(x_data))

        result = fit_lifetime_model(linear_model, x_data, y_data, {'slope': 1.0})

        assert 'slope' in result.confidence_intervals
        lower, upper = result.confidence_intervals['slope']
        assert lower <= upper
        # With noise, slope should be close to 2
        assert 1.5 < result.parameters['slope'] < 2.5


class TestResiduals:
    """Test residual calculations."""

    def test_residuals_sum_to_zero_for_good_fit(self):
        """Test that residuals sum to approximately zero for good fit."""
        def linear_model(x, slope):
            return slope * x

        x_data = np.array([1, 2, 3, 4, 5])
        y_data = np.array([2, 4, 6, 8, 10])

        result = fit_lifetime_model(linear_model, x_data, y_data, {'slope': 1.0})

        # Sum of residuals should be close to zero for good fit
        assert_allclose(np.sum(result.residuals), 0, atol=1)


@pytest.fixture
def sample_fitting_data():
    """Fixture providing sample data for fitting tests."""
    np.random.seed(42)
    x = np.linspace(0, 10, 20)
    y_true = 2.5 * x + 3
    y_noise = y_true + np.random.normal(0, 1, len(x))
    return x, y_noise


class TestIntegration:
    """Integration tests for complete fitting workflows."""

    def test_complete_fitting_workflow(self, sample_fitting_data):
        """Test complete fitting workflow with all metrics."""
        x_data, y_data = sample_fitting_data

        def model(x, slope, intercept):
            return slope * x + intercept

        initial_params = {'slope': 1.0, 'intercept': 0.0}
        result = fit_lifetime_model(model, x_data, y_data, initial_params)

        # Check all metrics are available
        assert result.r_squared > 0
        assert result.rmse > 0
        assert len(result.residuals) == len(x_data)
        assert result.covariance.shape == (2, 2)

        # Check predictions
        y_pred = model(x_data, **result.parameters)
        predicted_r2 = calculate_r_squared(y_data, y_pred)
        predicted_rmse = calculate_rmse(y_data, y_pred)

        assert_allclose(predicted_r2, result.r_squared)
        assert_allclose(predicted_rmse, result.rmse)

    def test_fitting_consistency(self, sample_fitting_data):
        """Test that multiple fits give consistent results."""
        x_data, y_data = sample_fitting_data

        def model(x, slope):
            return slope * x

        initial_params = {'slope': 2.0}

        result1 = fit_lifetime_model(model, x_data, y_data, initial_params)
        result2 = fit_lifetime_model(model, x_data, y_data, initial_params)

        # Results should be identical
        assert_allclose(result1.parameters['slope'], result2.parameters['slope'])

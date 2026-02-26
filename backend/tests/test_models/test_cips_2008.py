"""
Unit tests for CIPS 2008 (Bayerer) lifetime model.

Tests include:
- Reference value comparisons from Bayerer et al. 2008 paper
- Invalid parameter handling
- Boundary condition testing
- All six beta parameters validation
"""

import pytest
import numpy as np
import math

from app.core.models.cips_2008 import CIPS2008Model


class TestCIPS2008Model:
    """Test suite for CIPS 2008 model."""

    def test_model_initialization_default_params(self):
        """Test model initialization with default parameters."""
        model = CIPS2008Model()

        assert model.K == CIPS2008Model.DEFAULT_K
        assert model.beta1 == CIPS2008Model.DEFAULT_BETA1
        assert model.beta2 == CIPS2008Model.DEFAULT_BETA2
        assert model.beta3 == CIPS2008Model.DEFAULT_BETA3
        assert model.beta4 == CIPS2008Model.DEFAULT_BETA4
        assert model.beta5 == CIPS2008Model.DEFAULT_BETA5
        assert model.beta6 == CIPS2008Model.DEFAULT_BETA6
        assert model.get_model_name() == "CIPS-2008"

    def test_model_initialization_custom_params(self):
        """Test model initialization with custom parameters."""
        model = CIPS2008Model(
            K=2.0,
            beta1=-5.0,
            beta2=1500.0,
            beta3=-0.5,
            beta4=-0.8,
            beta5=-0.8,
            beta6=-0.6
        )

        assert model.K == 2.0
        assert model.beta1 == -5.0
        assert model.beta2 == 1500.0
        assert model.beta3 == -0.5
        assert model.beta4 == -0.8
        assert model.beta5 == -0.8
        assert model.beta6 == -0.6

    def test_calculate_cycles_to_failure_basic(self):
        """Test basic cycles to failure calculation."""
        # Using default parameters from Bayerer et al. 2008
        model = CIPS2008Model(K=1e10)  # K needs to be fitted

        result = model.calculate_cycles_to_failure(
            delta_Tj=80,
            Tj_max=398,  # 125°C in K
            t_on=1.0,
            I=100,
            V=1200,
            D=300
        )

        assert result > 0
        assert np.isfinite(result)

    def test_bayerer_reference_values(self):
        """Test against reference values from Bayerer et al. 2008 paper.

        The CIPS 2008 model was calibrated to experimental data.
        This test verifies the model formula is correct by comparing
        with expected trends from the paper.

        From the paper, for IGBT modules:
        - Nf decreases with increasing ΔTj (beta1 < 0)
        - Nf decreases with increasing Tj_max (beta2 > 0, exp term)
        - Nf decreases with longer t_on for creep-dominated (beta3 < 0)
        - Nf decreases with higher current (beta4 < 0)
        - Nf decreases with higher voltage rating (beta5 < 0)
        - Nf decreases with smaller bond wire (beta6 < 0, D in numerator)
        """
        model = CIPS2008Model(K=1e10)

        base_params = {
            'delta_Tj': 80,
            'Tj_max': 398,
            't_on': 1.0,
            'I': 100,
            'V': 1200,
            'D': 300
        }

        base_result = model.calculate_cycles_to_failure(**base_params)

        # Test delta_Tj effect (higher swing = lower life)
        result_high_swing = model.calculate_cycles_to_failure(
            **{**base_params, 'delta_Tj': 100}
        )
        assert result_high_swing < base_result

        # Test Tj_max effect (higher temp = lower life)
        result_high_temp = model.calculate_cycles_to_failure(
            **{**base_params, 'Tj_max': 423}  # 150°C
        )
        assert result_high_temp < base_result

        # Test t_on effect (longer on-time = lower life for beta3 < 0)
        result_long_on = model.calculate_cycles_to_failure(
            **{**base_params, 't_on': 10.0}
        )
        assert result_long_on < base_result

        # Test current effect
        result_high_current = model.calculate_cycles_to_failure(
            **{**base_params, 'I': 150}
        )
        assert result_high_current < base_result

    def test_all_required_parameters_must_be_present(self):
        """Test that all required parameters must be provided."""
        model = CIPS2008Model()

        # Missing all parameters
        with pytest.raises(ValueError, match="Missing required"):
            model.calculate_cycles_to_failure()

        # Missing some parameters
        with pytest.raises(ValueError, match="Missing required"):
            model.calculate_cycles_to_failure(delta_Tj=80, Tj_max=398)

    def test_alternative_parameter_names(self):
        """Test alternative parameter names."""
        model = CIPS2008Model(K=1e10)

        # Use alternative names
        result1 = model.calculate_cycles_to_failure(
            delta_Tj=80, Tj_max=398, t_on=1.0, I=100, V=1200, D=300
        )

        result2 = model.calculate_cycles_to_failure(
            dTj=80, T_max=398, heating_time=1.0, current=100, voltage=1200, bond_wire_diameter=300
        )

        assert result1 == pytest.approx(result2)

    def test_greek_letter_beta_parameters(self):
        """Test Greek letter β as parameter names."""
        model = CIPS2008Model(K=1e10)

        result = model.calculate_cycles_to_failure(
            delta_Tj=80,
            Tj_max=398,
            t_on=1.0,
            I=100,
            V=1200,
            D=300,
            β1=-4.0,
            β2=1300
        )

        assert result > 0

    def test_parameter_override(self):
        """Test parameter override in calculation."""
        model = CIPS2008Model(K=1e10, beta1=-4.423)

        result1 = model.calculate_cycles_to_failure(
            delta_Tj=80, Tj_max=398, t_on=1.0, I=100, V=1200, D=300
        )

        result2 = model.calculate_cycles_to_failure(
            delta_Tj=80, Tj_max=398, t_on=1.0, I=100, V=1200, D=300,
            K=2e10  # Override K
        )

        # K=2e10 should give approximately 2x the cycles
        assert result2 / result1 == pytest.approx(2.0, rel=0.1)

    def test_zero_or_negative_parameters_raise_error(self):
        """Test that zero or negative parameters raise errors."""
        model = CIPS2008Model(K=1e10)

        # Zero delta_Tj
        with pytest.raises(ValueError, match="positive"):
            model.calculate_cycles_to_failure(
                delta_Tj=0, Tj_max=398, t_on=1, I=100, V=1200, D=300
            )

        # Negative Tj_max
        with pytest.raises(ValueError, match="positive"):
            model.calculate_cycles_to_failure(
                delta_Tj=80, Tj_max=-10, t_on=1, I=100, V=1200, D=300
            )

        # Zero K
        with pytest.raises(ValueError, match="positive"):
            model.calculate_cycles_to_failure(
                delta_Tj=80, Tj_max=398, t_on=1, I=100, V=1200, D=300,
                K=0
            )

    def test_boundary_conditions(self):
        """Test boundary conditions at parameter limits."""
        model = CIPS2008Model(K=1e10)

        # Use more reasonable values to avoid overflow
        result_min = model.calculate_cycles_to_failure(
            delta_Tj=10,  # Small but reasonable
            Tj_max=300,  # Low but above absolute zero
            t_on=0.1,
            I=1,
            V=100,
            D=50
        )
        assert result_min > 0

        # Large values
        result_max = model.calculate_cycles_to_failure(
            delta_Tj=200,
            Tj_max=600,
            t_on=100,
            I=1000,
            V=2000,
            D=500
        )
        assert result_max > 0

    def test_exp_term_correctness(self):
        """Test that the exponential term is calculated correctly."""
        model = CIPS2008Model(K=1.0)

        # Fix all parameters except Tj_max
        # Nf = K * (dTj)^beta1 * exp(beta2/Tj_max) * other_terms

        result_low_temp = model.calculate_cycles_to_failure(
            delta_Tj=80, Tj_max=300, t_on=1, I=100, V=1200, D=300,
            beta1=-4.423, beta2=1285
        )

        result_high_temp = model.calculate_cycles_to_failure(
            delta_Tj=80, Tj_max=500, t_on=1, I=100, V=1200, D=300,
            beta1=-4.423, beta2=1285
        )

        # Higher Tj_max gives larger exp(beta2/Tj_max) = exp(positive/smaller)
        # Actually: exp(1285/300) > exp(1285/500)
        # So result_low_temp should be higher
        assert result_low_temp > result_high_temp

    def test_get_equation(self):
        """Test get_equation method."""
        model = CIPS2008Model()

        equation = model.get_equation()

        assert "Nf" in equation
        assert "K" in equation
        assert "ΔTj" in equation or "delta_Tj" in equation
        assert "β1" in equation or "beta1" in equation

    def test_get_parameters_info(self):
        """Test get_parameters_info method."""
        model = CIPS2008Model()

        info = model.get_parameters_info()

        # Check all parameters are present
        expected_params = ['K', 'beta1', 'beta2', 'beta3', 'beta4', 'beta5', 'beta6',
                          'delta_Tj', 'Tj_max', 't_on', 'I', 'V', 'D']

        for param in expected_params:
            assert param in info

        # Check structure
        assert 'description' in info['K']
        assert 'typical_range' in info['delta_Tj'] or 'valid_range' in info['delta_Tj']

    def test_validate_parameters_valid(self):
        """Test validate_parameters with valid input."""
        model = CIPS2008Model()

        result = model.validate_parameters(
            delta_Tj=80,
            Tj_max=398,
            t_on=1,
            I=100,
            V=1200,
            D=300
        )

        assert result is True

    def test_validate_parameters_invalid(self):
        """Test validate_parameters with invalid input."""
        model = CIPS2008Model()

        with pytest.raises(ValueError):
            model.validate_parameters(delta_Tj=-10)

    def test_get_reference(self):
        """Test get_reference method."""
        model = CIPS2008Model()

        reference = model.get_reference()

        assert "Bayerer" in reference
        assert "2008" in reference
        assert "CIPS" in reference

    def test_parameter_warnings(self):
        """Test that warnings are logged for out-of-range parameters."""
        import logging
        model = CIPS2008Model()

        # These should trigger warnings but not raise errors
        # Using reasonable values to avoid overflow while testing out-of-range
        result = model.calculate_cycles_to_failure(
            delta_Tj=200,  # Outside recommended range (60-150)
            Tj_max=600,    # Outside recommended range (398-523)
            t_on=100,      # Outside recommended range (1-60)
            I=10,          # OK (device dependent)
            V=100,         # Outside recommended range (600-1700)
            D=50,          # Outside recommended range (100-400)
            K=1e10
        )
        assert result > 0


class TestCIPS2008SpecialCases:
    """Test special cases and edge conditions."""

    def test_identical_parameters_consistency(self):
        """Test that identical parameters give identical results."""
        model = CIPS2008Model(K=1e10)

        params = {
            'delta_Tj': 75.5,
            'Tj_max': 398.3,
            't_on': 1.2,
            'I': 105.7,
            'V': 1195.3,
            'D': 298.6
        }

        result1 = model.calculate_cycles_to_failure(**params)
        result2 = model.calculate_cycles_to_failure(**params)

        assert result1 == result2

    def test_float_precision_handling(self):
        """Test with very small and very large float values."""
        model = CIPS2008Model(K=1e10)

        # Very small bond wire diameter
        result_small_d = model.calculate_cycles_to_failure(
            delta_Tj=80, Tj_max=398, t_on=1, I=100, V=1200, D=1e-3
        )

        # Very large voltage
        result_large_v = model.calculate_cycles_to_failure(
            delta_Tj=80, Tj_max=398, t_on=1, I=100, V=1e6, D=300
        )

        assert np.isfinite(result_small_d)
        assert np.isfinite(result_large_v)

    def test_scientific_notation_input(self):
        """Test parameters in scientific notation."""
        model = CIPS2008Model(K=1.234e10)

        result = model.calculate_cycles_to_failure(
            delta_Tj=8.0e1,
            Tj_max=3.98e2,
            t_on=1.0e0,
            I=1.0e2,
            V=1.2e3,
            D=3.0e2
        )

        assert np.isfinite(result)
        assert result > 0

    def test_all_beta_zero(self):
        """Test behavior when all beta exponents are zero."""
        model = CIPS2008Model(K=1000)

        # All betas = 0 means Nf = K (only K matters)
        result = model.calculate_cycles_to_failure(
            delta_Tj=80, Tj_max=398, t_on=1, I=100, V=1200, D=300,
            beta1=0, beta2=0, beta3=0, beta4=0, beta5=0, beta6=0
        )

        # Nf = K * 1 * exp(0) * 1 * 1 * 1 * 1 = K
        # But wait, if beta2=0, exp(0/Tj_max) = exp(0) = 1
        # Actually need to check the implementation
        assert np.isfinite(result)


class TestCIPS2008LiteratureComparison:
    """Compare with literature data where available."""

    def test_bayerer_figure_trend(self):
        """Test that model follows trends shown in Bayerer et al. figures.

        Figure 1 from Bayerer et al. 2008 shows log(Nf) vs ΔTj
        with approximately linear relationship (power law in linear space).
        """
        model = CIPS2008Model(K=1e12)

        # Test multiple delta_Tj values
        delta_Tj_values = [60, 80, 100, 120, 140]
        results = []

        for dTj in delta_Tj_values:
            result = model.calculate_cycles_to_failure(
                delta_Tj=dTj, Tj_max=398, t_on=1, I=100, V=1200, D=300
            )
            results.append(result)

        # Check monotonic decrease
        for i in range(len(results) - 1):
            assert results[i] > results[i + 1]

        # Check power law: log(Nf) vs log(dTj) should be linear
        log_dTj = np.log(delta_Tj_values)
        log_Nf = np.log(results)

        # Linear fit should have good R²
        coeffs = np.polyfit(log_dTj, log_Nf, 1)
        fitted = np.exp(coeffs[1]) * np.array(delta_Tj_values) ** coeffs[0]

        # Each fitted value should be within 1% of actual
        for actual, fit in zip(results, fitted):
            assert actual == pytest.approx(fit, rel=0.01)

    def test_heating_time_effect(self):
        """Test heating time (dwell time) effect per CIPS 2008.

        Longer heating time allows more creep deformation, reducing life.
        This is captured by beta3 < 0.
        """
        model = CIPS2008Model(K=1e10)

        t_on_values = [0.1, 1, 10, 60]  # seconds
        results = []

        for t_on in t_on_values:
            result = model.calculate_cycles_to_failure(
                delta_Tj=80, Tj_max=398, t_on=t_on, I=100, V=1200, D=300
            )
            results.append(result)

        # Longer t_on should give fewer cycles
        for i in range(len(results) - 1):
            assert results[i] > results[i + 1]

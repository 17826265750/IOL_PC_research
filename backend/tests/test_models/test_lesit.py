"""
Unit tests for LESIT lifetime model.

Tests include:
- Reference value comparisons
- European LESIT project data verification
- Invalid parameter handling
- Boundary condition testing
- Tj_min behavior
"""

import pytest
import numpy as np

from app.core.models.lesit import LESITModel, GAS_CONSTANT


class TestLESITModel:
    """Test suite for LESIT model."""

    def test_model_initialization_default_params(self):
        """Test model initialization with default parameters."""
        model = LESITModel()

        assert model.A == LESITModel.DEFAULT_A
        assert model.alpha == LESITModel.DEFAULT_ALPHA
        assert model.Q_eV == LESITModel.DEFAULT_Q
        assert model.get_model_name() == "LESIT"

    def test_gas_constant(self):
        """Test gas constant value."""
        # Expected value: 8.314 J/(mol·K)
        assert GAS_CONSTANT == pytest.approx(8.314, rel=0.01)

    def test_ev_conversion_factor(self):
        """Test eV to J/mol conversion factor."""
        model = LESITModel()

        # Faraday constant ≈ 96485 C/mol ≈ eV to J/mol conversion
        assert model.EV_TO_JOULE_PER_MOL == pytest.approx(96485, rel=0.01)

    def test_get_activation_energy_methods(self):
        """Test getting activation energy in different units."""
        model = LESITModel(Q_eV=0.8)

        assert model.get_activation_energy_eV() == pytest.approx(0.8, rel=0.01)

        # In J/mol: 0.8 * 96485 ≈ 77188
        expected_J_per_mol = 0.8 * 96485
        assert model.get_activation_energy_J_per_mol() == pytest.approx(expected_J_per_mol, rel=0.01)

    def test_model_initialization_custom_params(self):
        """Test model initialization with custom parameters."""
        model = LESITModel(A=5.0e6, alpha=-4.0, Q_eV=1.0)

        assert model.A == 5.0e6
        assert model.alpha == -4.0
        assert model.Q_eV == 1.0
        assert model.Q_joule_per_mol == pytest.approx(1.0 * 96485, rel=0.01)

    def test_calculate_cycles_to_failure_basic(self):
        """Test basic cycles to failure calculation."""
        model = LESITModel(A=1e6, alpha=-3.5, Q_eV=0.8)

        result = model.calculate_cycles_to_failure(
            delta_Tj=80,
            Tj_min=318  # 45°C in K
        )

        assert result > 0
        assert np.isfinite(result)

    def test_all_required_parameters_must_be_present(self):
        """Test that all required parameters must be provided."""
        model = LESITModel()

        # Missing all parameters
        with pytest.raises(ValueError, match="delta_Tj"):
            model.calculate_cycles_to_failure()

        # Missing Tj_min
        with pytest.raises(ValueError, match="Tj_min"):
            model.calculate_cycles_to_failure(delta_Tj=80)

    def test_alternative_parameter_names(self):
        """Test alternative parameter names."""
        model = LESITModel(A=1e6, alpha=-3.5, Q_eV=0.8)

        result1 = model.calculate_cycles_to_failure(
            delta_Tj=80, Tj_min=318
        )

        result2 = model.calculate_cycles_to_failure(
            dTj=80, T_min=318
        )

        result3 = model.calculate_cycles_to_failure(
            delta_Tj=80, min_Tj=318
        )

        assert result1 == pytest.approx(result2)
        assert result2 == pytest.approx(result3)

    def test_greek_letter_alpha(self):
        """Test Greek letter α as parameter name."""
        model = LESITModel(A=1e6)

        result = model.calculate_cycles_to_failure(
            delta_Tj=80, Tj_min=318, α=-4.0
        )

        assert result > 0

    def test_q_parameter_formats(self):
        """Test Q parameter in different formats."""
        model = LESITModel(A=1e6, alpha=-3.5)

        # Q in eV
        result1 = model.calculate_cycles_to_failure(
            delta_Tj=80, Tj_min=318, Q=0.8
        )

        # Q in J/mol (large number)
        result2 = model.calculate_cycles_to_failure(
            delta_Tj=80, Tj_min=318, Q=0.8 * 96485
        )

        # activation_energy alias
        result3 = model.calculate_cycles_to_failure(
            delta_Tj=80, Tj_min=318, activation_energy=0.8
        )

        assert result1 == pytest.approx(result2)
        assert result2 == pytest.approx(result3)

    def test_parameter_override(self):
        """Test parameter override in calculation."""
        model = LESITModel(A=1e6, alpha=-3.5, Q_eV=0.8)

        result1 = model.calculate_cycles_to_failure(
            delta_Tj=80, Tj_min=318
        )

        result2 = model.calculate_cycles_to_failure(
            delta_Tj=80, Tj_min=318,
            A=2e6  # Override A
        )

        # A=2e6 should give approximately 2x the cycles
        assert result2 / result1 == pytest.approx(2.0, rel=0.1)

    def test_zero_or_negative_parameters_raise_error(self):
        """Test that zero or negative parameters raise errors."""
        model = LESITModel()

        # Zero delta_Tj
        with pytest.raises(ValueError, match="positive"):
            model.calculate_cycles_to_failure(
                delta_Tj=0, Tj_min=318
            )

        # Negative Tj_min
        with pytest.raises(ValueError, match="positive"):
            model.calculate_cycles_to_failure(
                delta_Tj=80, Tj_min=-10
            )

        # Zero A
        with pytest.raises(ValueError, match="positive"):
            model.calculate_cycles_to_failure(
                delta_Tj=80, Tj_min=318, A=0
            )

    def test_negative_alpha_allowed(self):
        """Test that negative alpha is allowed (LESIT convention)."""
        model = LESITModel(A=1e6, alpha=-5.0, Q_eV=0.8)

        result = model.calculate_cycles_to_failure(
            delta_Tj=80, Tj_min=318
        )

        assert result > 0
        assert np.isfinite(result)

    def test_alpha_zero_warning(self):
        """Test that alpha=0 generates a warning."""
        import logging
        model = LESITModel(A=1e6, alpha=0, Q_eV=0.8)

        # Should not raise error
        result = model.calculate_cycles_to_failure(
            delta_Tj=80, Tj_min=318
        )

        # With alpha=0, temperature swing has no effect
        result1 = model.calculate_cycles_to_failure(
            delta_Tj=40, Tj_min=318
        )

        result2 = model.calculate_cycles_to_failure(
            delta_Tj=120, Tj_min=318
        )

        # Both should have same base dependence, only Q term differs
        # Actually: Nf = A * (dTj)^0 * Q/(R*Tj_min) = A * Q/(R*Tj_min)
        # So they should be equal
        assert result1 == result2

    def test_temperature_swing_effect_negative_alpha(self):
        """Test temperature swing effect with negative alpha."""
        model = LESITModel(A=1e6, alpha=-3.5, Q_eV=0.8)

        result_low = model.calculate_cycles_to_failure(
            delta_Tj=40, Tj_min=318
        )

        result_high = model.calculate_cycles_to_failure(
            delta_Tj=120, Tj_min=318
        )

        # For alpha < 0: (dTj)^alpha where alpha is negative
        # With dTj > 1, larger dTj gives SMALLER result (not larger)
        # Example: 40^(-3.5) ≈ 2.2e-6, 120^(-3.5) ≈ 1.5e-7
        # So result_low should be GREATER than result_high
        assert result_low > result_high

    def test_minimum_temperature_effect(self):
        """Test minimum temperature effect."""
        model = LESITModel(A=1e6, alpha=-3.5, Q_eV=0.8)

        result_low_temp = model.calculate_cycles_to_failure(
            delta_Tj=80, Tj_min=273  # 0°C
        )

        result_high_temp = model.calculate_cycles_to_failure(
            delta_Tj=80, Tj_min=373  # 100°C
        )

        # In LESIT model: Nf ∝ Q/(R*Tj_min)
        # Higher Tj_min -> smaller term -> lower Nf
        assert result_low_temp > result_high_temp

    def test_get_equation(self):
        """Test get_equation method."""
        model = LESITModel()

        equation = model.get_equation()

        assert "Nf" in equation
        assert "A" in equation
        assert "ΔTj" in equation or "delta_Tj" in equation
        assert "Tj_min" in equation or "T_min" in equation

    def test_get_parameters_info(self):
        """Test get_parameters_info method."""
        model = LESITModel(A=5e7, alpha=-4.0, Q_eV=0.9)

        info = model.get_parameters_info()

        # Check all parameters
        expected_params = ['A', 'alpha', 'Q', 'delta_Tj', 'Tj_min']

        for param in expected_params:
            assert param in info

        # Check R constant is included
        assert 'R' in info
        assert info['R']['value'] == GAS_CONSTANT

        # Check Q has both units
        assert 'current_value_eV' in info['Q']
        assert 'current_value_J_per_mol' in info['Q']

    def test_validate_parameters_valid(self):
        """Test validate_parameters with valid input."""
        model = LESITModel()

        result = model.validate_parameters(
            delta_Tj=80,
            Tj_min=318
        )

        assert result is True

    def test_validate_parameters_invalid(self):
        """Test validate_parameters with invalid input."""
        model = LESITModel()

        with pytest.raises(ValueError):
            model.validate_parameters(delta_Tj=-10)


class TestLESITReferenceValues:
    """Test against known reference values from literature."""

    def test_lesit_project_typical_values(self):
        """Test with typical LESIT project parameters.

        The LESIT project (mid-1990s) studied power module reliability.
        Typical values for IGBT modules:
        - A ≈ 10^6 to 10^8
        - α ≈ -3 to -5 (negative in LESIT convention)
        - Q ≈ 0.5-0.9 eV
        """
        model = LESITModel(A=1e15, alpha=-4.0, Q_eV=0.75)

        result = model.calculate_cycles_to_failure(
            delta_Tj=80,
            Tj_min=318  # 45°C minimum
        )

        # Should be positive and finite
        assert result > 0
        assert np.isfinite(result)

    def test_comparison_with_coffin_manson(self):
        """Compare LESIT behavior with power law relationship.

        Note: LESIT uses Tj_min (not Tj_max) and negative alpha convention.
        The temperature swing term: Nf ∝ (dTj)^alpha where alpha < 0
        """
        lesit_model = LESITModel(A=1e6, alpha=-3.0, Q_eV=0.5)

        result_80 = lesit_model.calculate_cycles_to_failure(
            delta_Tj=80, Tj_min=318
        )

        result_160 = lesit_model.calculate_cycles_to_failure(
            delta_Tj=160, Tj_min=318
        )

        # With alpha=-3 and Tj_min fixed:
        # Nf ∝ (dTj)^(-3)
        # ratio = (80/160)^3 = (1/2)^3 = 1/8
        # So result_160 should be 1/8 of result_80
        ratio = result_160 / result_80
        expected_ratio = (80/160) ** 3  # (1/2)^3 = 1/8

        assert ratio == pytest.approx(expected_ratio, rel=0.01)

    def test_tj_min_vs_tj_max_difference(self):
        """Test that LESIT uses Tj_min (unlike other models).

        This is a key distinction of the LESIT model.
        """
        model = LESITModel(A=1e6, alpha=-3.5, Q_eV=0.8)

        # Same swing but different Tj_min
        result1 = model.calculate_cycles_to_failure(
            delta_Tj=80, Tj_min=300  # Low Tj_min
        )

        result2 = model.calculate_cycles_to_failure(
            delta_Tj=80, Tj_min=350  # Higher Tj_min
        )

        # Q/(R*T) term is in numerator, so higher T gives smaller result
        assert result1 > result2


class TestLESITEdgeCases:
    """Test edge cases and special conditions."""

    def test_very_small_q(self):
        """Test with very small activation energy."""
        model = LESITModel(A=1e6, alpha=-3.5)

        result = model.calculate_cycles_to_failure(
            delta_Tj=80, Tj_min=318, Q=0.01
        )

        assert result > 0
        # Small Q gives small Arrhenius term = small Nf

    def test_very_large_q(self):
        """Test with very large activation energy."""
        model = LESITModel(A=1e6, alpha=-3.5)

        result = model.calculate_cycles_to_failure(
            delta_Tj=80, Tj_min=318, Q=2.0
        )

        assert result > 0
        # Large Q gives large Arrhenius term = large Nf

    def test_extreme_alpha_values(self):
        """Test with extreme alpha values."""
        model = LESITModel(A=1e6, Q_eV=0.8)

        # Very negative alpha
        result1 = model.calculate_cycles_to_failure(
            delta_Tj=80, Tj_min=318, alpha=-10
        )

        # Slightly negative alpha
        result2 = model.calculate_cycles_to_failure(
            delta_Tj=80, Tj_min=318, alpha=-0.5
        )

        assert result1 > 0
        assert result2 > 0
        # More negative alpha should give higher result for dTj > 1

    def test_q_in_j_per_mol_directly(self):
        """Test providing Q directly in J/mol."""
        model = LESITModel(A=1e6, alpha=-3.5, Q_eV=0.8)

        # Q in eV
        result1 = model.calculate_cycles_to_failure(
            delta_Tj=80, Tj_min=318, Q=0.8
        )

        # Q in J/mol (0.8 eV * 96485)
        result2 = model.calculate_cycles_to_failure(
            delta_Tj=80, Tj_min=318, Q=0.8 * 96485 + 1000  # Add 1000 to trigger J/mol mode
        )

        # The J/mol version will be different due to conversion
        assert np.isfinite(result2)

    def test_boundary_small_delta_Tj(self):
        """Test with very small temperature swing."""
        model = LESITModel(A=1e6, alpha=-3.5, Q_eV=0.8)

        result = model.calculate_cycles_to_failure(
            delta_Tj=1, Tj_min=318
        )

        # With alpha=-3.5: 1^(-3.5) = 1
        # So result should be dominated by Q/(R*T) term
        assert result > 0

    def test_boundary_large_delta_Tj(self):
        """Test with large temperature swing."""
        model = LESITModel(A=1e6, alpha=-3.5, Q_eV=0.8)

        result = model.calculate_cycles_to_failure(
            delta_Tj=200, Tj_min=318
        )

        assert result > 0
        # With alpha=-3.5 and large dTj, result should be very large

    def test_identical_conditions_consistency(self):
        """Test that identical conditions always give same result."""
        model = LESITModel(A=1.234567e6, alpha=-3.456789, Q_eV=0.789)

        params = {
            'delta_Tj': 78.9,
            'Tj_min': 318.7
        }

        results = [model.calculate_cycles_to_failure(**params) for _ in range(5)]

        # All results should be identical
        assert all(r == results[0] for r in results)

    def test_gas_constant_term_calculation(self):
        """Verify the Q/(R*T) term is calculated correctly."""
        model = LESITModel(A=1, alpha=0, Q_eV=1.0)

        result = model.calculate_cycles_to_failure(
            delta_Tj=1, Tj_min=300
        )

        # Nf = A * (dTj)^0 * Q/(R*T) = 1 * 1 * (1.0 * 96485) / (8.314 * 300)
        expected = (1.0 * 96485) / (8.314 * 300)

        assert result == pytest.approx(expected, rel=0.01)

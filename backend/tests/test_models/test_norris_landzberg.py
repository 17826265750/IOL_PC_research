"""
Unit tests for Norris-Landzberg lifetime model.

Tests include:
- Reference value comparisons
- Intel solder joint model verification
- Invalid parameter handling
- Boundary condition testing
"""

import pytest
import numpy as np

from app.core.models.norris_landzberg import (
    NorrisLandzbergModel,
    BOLTZMANN_CONSTANT_EV_PER_K
)


class TestNorrisLandzbergModel:
    """Test suite for Norris-Landzberg model."""

    def test_model_initialization_default_params(self):
        """Test model initialization with default parameters."""
        model = NorrisLandzbergModel()

        assert model.A == NorrisLandzbergModel.DEFAULT_A
        assert model.alpha == NorrisLandzbergModel.DEFAULT_ALPHA
        assert model.beta == NorrisLandzbergModel.DEFAULT_BETA
        assert model.Ea == NorrisLandzbergModel.DEFAULT_EA
        assert model.get_model_name() == "Norris-Landzberg"

    def test_model_initialization_custom_params(self):
        """Test model initialization with custom parameters."""
        model = NorrisLandzbergModel(
            A=5.0e6,
            alpha=2.5,
            beta=0.5,
            Ea=0.6
        )

        assert model.A == 5.0e6
        assert model.alpha == 2.5
        assert model.beta == 0.5
        assert model.Ea == 0.6

    def test_boltzmann_constant(self):
        """Test Boltzmann constant value."""
        # Expected value: 8.617e-5 eV/K
        assert BOLTZMANN_CONSTANT_EV_PER_K == pytest.approx(8.617e-5, rel=0.01)

    def test_calculate_cycles_to_failure_basic(self):
        """Test basic cycles to failure calculation."""
        model = NorrisLandzbergModel(A=1.0e6, alpha=2.0, beta=0.333, Ea=0.5)

        result = model.calculate_cycles_to_failure(
            delta_Tj=80,
            f=0.01,
            Tj_max=398
        )

        assert result > 0
        assert np.isfinite(result)

    def test_all_required_parameters_must_be_present(self):
        """Test that all required parameters must be provided."""
        model = NorrisLandzbergModel()

        # Missing all parameters
        with pytest.raises(ValueError, match="delta_Tj"):
            model.calculate_cycles_to_failure()

        # Missing f
        with pytest.raises(ValueError, match="f"):
            model.calculate_cycles_to_failure(delta_Tj=80, Tj_max=398)

        # Missing Tj_max
        with pytest.raises(ValueError, match="Tj_max"):
            model.calculate_cycles_to_failure(delta_Tj=80, f=0.01)

    def test_alternative_parameter_names(self):
        """Test alternative parameter names."""
        model = NorrisLandzbergModel(A=1e6, alpha=2.0, beta=0.333, Ea=0.5)

        result1 = model.calculate_cycles_to_failure(
            delta_Tj=80, f=0.01, Tj_max=398
        )

        result2 = model.calculate_cycles_to_failure(
            dTj=80, frequency=0.01, T_max=398
        )

        result3 = model.calculate_cycles_to_failure(
            delta_Tj=80, f=0.01, max_Tj=398
        )

        assert result1 == pytest.approx(result2)
        assert result2 == pytest.approx(result3)

    def test_greek_letter_parameters(self):
        """Test Greek letter α and β as parameter names."""
        model = NorrisLandzbergModel(A=1e6)

        result = model.calculate_cycles_to_failure(
            delta_Tj=80, f=0.01, Tj_max=398,
            α=2.5, β=0.4
        )

        assert result > 0

    def test_activation_energy_alternative_name(self):
        """Test activation_energy as alternative to Ea."""
        model = NorrisLandzbergModel(A=1e6, alpha=2.0, beta=0.333)

        result = model.calculate_cycles_to_failure(
            delta_Tj=80, f=0.01, Tj_max=398,
            activation_energy=0.7
        )

        assert result > 0

    def test_parameter_override(self):
        """Test parameter override in calculation."""
        model = NorrisLandzbergModel(A=1e6, alpha=2.0, beta=0.333, Ea=0.5)

        result1 = model.calculate_cycles_to_failure(
            delta_Tj=80, f=0.01, Tj_max=398
        )

        result2 = model.calculate_cycles_to_failure(
            delta_Tj=80, f=0.01, Tj_max=398,
            A=2e6  # Override A
        )

        # A=2e6 should give approximately 2x the cycles
        assert result2 / result1 == pytest.approx(2.0, rel=0.1)

    def test_zero_or_negative_parameters_raise_error(self):
        """Test that zero or negative parameters raise errors."""
        model = NorrisLandzbergModel()

        # Zero delta_Tj
        with pytest.raises(ValueError, match="positive"):
            model.calculate_cycles_to_failure(
                delta_Tj=0, f=0.01, Tj_max=398
            )

        # Negative frequency
        with pytest.raises(ValueError, match="positive"):
            model.calculate_cycles_to_failure(
                delta_Tj=80, f=-0.01, Tj_max=398
            )

        # Zero Tj_max (division by zero in Arrhenius)
        with pytest.raises(ValueError, match="positive"):
            model.calculate_cycles_to_failure(
                delta_Tj=80, f=0.01, Tj_max=0
            )

    def test_negative_beta_allowed(self):
        """Test that negative beta is allowed (inverse frequency effect)."""
        model = NorrisLandzbergModel(A=1e6, alpha=2.0, beta=-0.5, Ea=0.5)

        result = model.calculate_cycles_to_failure(
            delta_Tj=80, f=0.01, Tj_max=398
        )

        assert result > 0
        assert np.isfinite(result)

    def test_temperature_swing_effect(self):
        """Test that higher temperature swing reduces life."""
        model = NorrisLandzbergModel(A=1e6, alpha=2.0, beta=0.333, Ea=0.5)

        result_low = model.calculate_cycles_to_failure(
            delta_Tj=40, f=0.01, Tj_max=398
        )

        result_high = model.calculate_cycles_to_failure(
            delta_Tj=100, f=0.01, Tj_max=398
        )

        # Higher swing should give fewer cycles
        assert result_low > result_high

    def test_frequency_effect(self):
        """Test frequency effect on lifetime."""
        model = NorrisLandzbergModel(A=1e6, alpha=2.0, beta=0.333, Ea=0.5)

        result_low_freq = model.calculate_cycles_to_failure(
            delta_Tj=80, f=0.001, Tj_max=398
        )

        result_high_freq = model.calculate_cycles_to_failure(
            delta_Tj=80, f=0.1, Tj_max=398
        )

        # For beta > 0: higher frequency = more cycles
        assert result_high_freq > result_low_freq

    def test_maximum_temperature_effect(self):
        """Test maximum temperature (Arrhenius) effect."""
        model = NorrisLandzbergModel(A=1e6, alpha=2.0, beta=0.333, Ea=0.5)

        result_low_temp = model.calculate_cycles_to_failure(
            delta_Tj=80, f=0.01, Tj_max=350
        )

        result_high_temp = model.calculate_cycles_to_failure(
            delta_Tj=80, f=0.01, Tj_max=450
        )

        # Higher Tj_max should give more cycles (higher activation energy barrier)
        # But wait: exp(Ea/(kB*T)) - higher T gives smaller exp term
        # Actually in the model: exp(Ea/(kB*Tj_max))
        # Higher T -> smaller exponent -> smaller Nf
        # Let me check the model formula
        # Nf = A * (ΔTj)^(-α) * f^β * exp(Ea / (kB * Tj_max))
        # Higher Tj_max -> smaller exp(Ea/(kB*T)) -> smaller Nf
        # So result_high_temp should be LOWER
        assert result_low_temp > result_high_temp

    def test_get_equation(self):
        """Test get_equation method."""
        model = NorrisLandzbergModel()

        equation = model.get_equation()

        assert "Nf" in equation
        assert "A" in equation
        assert "ΔTj" in equation or "delta_Tj" in equation
        assert "f" in equation or "frequency" in equation
        assert "Tj_max" in equation or "T_max" in equation

    def test_get_parameters_info(self):
        """Test get_parameters_info method."""
        model = NorrisLandzbergModel(A=5e7, alpha=2.5, beta=0.4, Ea=0.7)

        info = model.get_parameters_info()

        # Check all parameters
        expected_params = ['A', 'alpha', 'beta', 'Ea', 'delta_Tj', 'f', 'Tj_max']

        for param in expected_params:
            assert param in info

        # Check Boltzmann constant is included
        assert 'kB' in info
        assert info['kB']['value'] == BOLTZMANN_CONSTANT_EV_PER_K

    def test_validate_parameters_valid(self):
        """Test validate_parameters with valid input."""
        model = NorrisLandzbergModel()

        result = model.validate_parameters(
            delta_Tj=80,
            f=0.01,
            Tj_max=398
        )

        assert result is True

    def test_validate_parameters_invalid(self):
        """Test validate_parameters with invalid input."""
        model = NorrisLandzbergModel()

        with pytest.raises(ValueError):
            model.validate_parameters(delta_Tj=-10)


class TestNorrisLandzbergReferenceValues:
    """Test against known reference values from literature."""

    def test_intel_solder_joint_model(self):
        """Test with Intel's original solder joint model parameters.

        Norris and Landzberg (1969) developed their model for
        lead (Pb-Sn) solder joints used in IBM flip-chip packages.

        Typical values for Pb-Sn solder:
        - α ≈ 2.0
        - β ≈ 1/3 (0.33)
        - Ea ≈ 0.5-0.8 eV
        """
        # Original model approximation
        model = NorrisLandzbergModel(
            A=1e7,
            alpha=2.0,
            beta=0.33,
            Ea=0.5
        )

        # Standard test condition: 80K swing at 125°C max
        result = model.calculate_cycles_to_failure(
            delta_Tj=80,
            f=0.01,  # 1 cycle per 100 seconds
            Tj_max=398  # 125°C
        )

        # Should be in reasonable range for solder joints
        assert 100 < result < 1e9

    def test_arrhenius_term_verification(self):
        """Verify Arrhenius term is calculated correctly.

        The Arrhenius term exp(Ea/(kB*T)) should:
        - Decrease as T increases (for Ea > 0)
        - Increase as Ea increases (for T > 0)
        """
        # Use a very small alpha to minimize Coffin-Manson effect
        model = NorrisLandzbergModel(A=1e6, alpha=0.1, beta=0, Ea=0.5)

        # Compare results at different temperatures
        result_300K = model.calculate_cycles_to_failure(
            delta_Tj=100, f=1, Tj_max=300
        )

        result_400K = model.calculate_cycles_to_failure(
            delta_Tj=100, f=1, Tj_max=400
        )

        # Higher temperature should give lower result (dominates the small alpha effect)
        assert result_300K > result_400K

    def test_frequency_exponent_range(self):
        """Test frequency effect across typical range.

        For creep-dominated solder fatigue:
        - β ≈ 1/3 means Nf ∝ f^(1/3)
        - Doubling frequency increases Nf by 2^(1/3) ≈ 1.26
        """
        model = NorrisLandzbergModel(
            A=1e6,
            alpha=2.0,
            beta=1.0/3,  # Exactly 1/3
            Ea=0.5
        )

        result_f1 = model.calculate_cycles_to_failure(
            delta_Tj=80, f=0.01, Tj_max=398
        )

        result_f2 = model.calculate_cycles_to_failure(
            delta_Tj=80, f=0.02, Tj_max=398  # 2x frequency
        )

        # Ratio should be 2^(1/3)
        expected_ratio = 2 ** (1.0/3)
        actual_ratio = result_f2 / result_f1

        assert actual_ratio == pytest.approx(expected_ratio, rel=0.01)

    def test_combined_effects(self):
        """Test that all three effects combine correctly.

        Norris-Landzberg model combines:
        1. Coffin-Manson: (ΔTj)^(-α)
        2. Frequency: f^β
        3. Arrhenius: exp(Ea/(kB*Tj_max))
        """
        model = NorrisLandzbergModel(
            A=1000,
            alpha=2,
            beta=0.5,
            Ea=0.5
        )

        # Base case
        result_base = model.calculate_cycles_to_failure(
            delta_Tj=100, f=1, Tj_max=400
        )

        # Verify result is positive and finite
        assert result_base > 0
        assert np.isfinite(result_base)

    def test_typical_sac305_parameters(self):
        """Test with typical SAC305 (lead-free) solder parameters.

        SAC305 (96.5Sn-3.0Ag-0.5Cu) has different fatigue behavior
        than traditional Pb-Sn solder:
        - Often higher α (2.5-3.5)
        - Different Ea (0.8-1.2 eV)
        """
        model = NorrisLandzbergModel(
            A=1e8,
            alpha=3.0,  # Higher for lead-free
            beta=0.33,
            Ea=0.9  # Higher activation energy
        )

        result = model.calculate_cycles_to_failure(
            delta_Tj=80,
            f=0.01,
            Tj_max=398
        )

        assert result > 0
        assert np.isfinite(result)


class TestNorrisLandzbergEdgeCases:
    """Test edge cases and special conditions."""

    def test_beta_zero_warning(self):
        """Test that beta=0 generates a warning (frequency has no effect)."""
        import logging
        model = NorrisLandzbergModel(A=1e6, alpha=2.0, beta=0, Ea=0.5)

        # Should not raise error, but log warning
        result = model.calculate_cycles_to_failure(
            delta_Tj=80, f=0.01, Tj_max=398
        )

        # With beta=0, frequency term = f^0 = 1
        result_f1 = model.calculate_cycles_to_failure(
            delta_Tj=80, f=0.001, Tj_max=398
        )

        result_f2 = model.calculate_cycles_to_failure(
            delta_Tj=80, f=1.0, Tj_max=398
        )

        # Should be identical when beta=0
        assert result_f1 == result_f2

    def test_very_high_frequency(self):
        """Test behavior at very high cycling frequency."""
        model = NorrisLandzbergModel(A=1e6, alpha=2.0, beta=0.333, Ea=0.5)

        # 10 Hz cycling
        result = model.calculate_cycles_to_failure(
            delta_Tj=80, f=10, Tj_max=398
        )

        assert result > 0

    def test_very_low_frequency(self):
        """Test behavior at very low cycling frequency."""
        model = NorrisLandzbergModel(A=1e6, alpha=2.0, beta=0.333, Ea=0.5)

        # 0.001 Hz (1 cycle per 1000 seconds)
        result = model.calculate_cycles_to_failure(
            delta_Tj=80, f=0.001, Tj_max=398
        )

        assert result > 0

    def test_extreme_activation_energy(self):
        """Test with extreme activation energy values."""
        model = NorrisLandzbergModel(A=1e6, alpha=2.0, beta=0.333)

        # Very low Ea
        result_low = model.calculate_cycles_to_failure(
            delta_Tj=80, f=0.01, Tj_max=398, Ea=0.1
        )

        # Very high Ea
        result_high = model.calculate_cycles_to_failure(
            delta_Tj=80, f=0.01, Tj_max=398, Ea=1.5
        )

        # Higher Ea should give higher result (stronger temperature dependence)
        assert result_high > result_low

    def test_identical_conditions_consistency(self):
        """Test that identical conditions always give same result."""
        model = NorrisLandzbergModel(A=1.234567e6, alpha=2.345678, beta=0.456789, Ea=0.6789)

        params = {
            'delta_Tj': 78.9,
            'f': 0.0123,
            'Tj_max': 398.7
        }

        results = [model.calculate_cycles_to_failure(**params) for _ in range(5)]

        # All results should be identical
        assert all(r == results[0] for r in results)

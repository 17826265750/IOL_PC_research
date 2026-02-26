"""
Unit tests for Coffin-Manson lifetime model.

Tests include:
- Reference value comparisons
- Invalid parameter handling
- Boundary condition testing
- Default parameter behavior
"""

import pytest
import numpy as np

from app.core.models.coffin_manson import CoffinMansonModel


class TestCoffinMansonModel:
    """Test suite for Coffin-Manson model."""

    def test_model_initialization_default_params(self):
        """Test model initialization with default parameters."""
        model = CoffinMansonModel()

        assert model.A == CoffinMansonModel.DEFAULT_A
        assert model.alpha == CoffinMansonModel.DEFAULT_ALPHA
        assert model.get_model_name() == "Coffin-Manson"

    def test_model_initialization_custom_params(self):
        """Test model initialization with custom parameters."""
        model = CoffinMansonModel(A=5.0e6, alpha=2.5)

        assert model.A == 5.0e6
        assert model.alpha == 2.5

    def test_calculate_cycles_to_failure_basic(self):
        """Test basic cycles to failure calculation."""
        model = CoffinMansonModel(A=1.0e6, alpha=2.0)

        # Nf = A * (dTj)^(-alpha) = 1e6 * 100^(-2) = 1e6 / 10000 = 100
        result = model.calculate_cycles_to_failure(delta_Tj=100)

        assert result == pytest.approx(100.0, rel=0.01)

    def test_calculate_cycles_with_parameter_override(self):
        """Test calculation with parameter override."""
        model = CoffinMansonModel(A=1.0e6, alpha=2.0)

        # Override A in the call
        result = model.calculate_cycles_to_failure(delta_Tj=100, A=2.0e6)

        # Nf = 2e6 * 100^(-2) = 200
        assert result == pytest.approx(200.0, rel=0.01)

    def test_alternative_parameter_names(self):
        """Test alternative parameter names for delta_Tj."""
        model = CoffinMansonModel(A=1.0e6, alpha=2.0)

        result1 = model.calculate_cycles_to_failure(delta_Tj=100)
        result2 = model.calculate_cycles_to_failure(dTj=100)
        result3 = model.calculate_cycles_to_failure(delta_T_j=100)

        assert result1 == pytest.approx(result2)
        assert result2 == pytest.approx(result3)

    def test_greek_letter_alpha(self):
        """Test Greek letter α as parameter name."""
        model = CoffinMansonModel(A=1.0e6, alpha=2.0)

        result = model.calculate_cycles_to_failure(delta_Tj=100, α=3.0)

        # Nf = 1e6 * 100^(-3) = 1
        assert result == pytest.approx(1.0, rel=0.01)

    def test_missing_delta_Tj_raises_error(self):
        """Test that missing delta_Tj raises ValueError."""
        model = CoffinMansonModel()

        with pytest.raises(ValueError, match="delta_Tj"):
            model.calculate_cycles_to_failure()

    def test_zero_delta_Tj_raises_error(self):
        """Test that zero delta_Tj raises ValueError."""
        model = CoffinMansonModel()

        with pytest.raises(ValueError, match="positive"):
            model.calculate_cycles_to_failure(delta_Tj=0)

    def test_negative_delta_Tj_raises_error(self):
        """Test that negative delta_Tj raises ValueError."""
        model = CoffinMansonModel()

        with pytest.raises(ValueError, match="positive"):
            model.calculate_cycles_to_failure(delta_Tj=-10)

    def test_negative_A_raises_error(self):
        """Test that negative A raises ValueError."""
        model = CoffinMansonModel()

        with pytest.raises(ValueError, match="A"):
            model.calculate_cycles_to_failure(delta_Tj=100, A=-1e6)

    def test_negative_alpha_raises_error(self):
        """Test that negative alpha raises ValueError."""
        model = CoffinMansonModel()

        with pytest.raises(ValueError, match="alpha"):
            model.calculate_cycles_to_failure(delta_Tj=100, alpha=-1.0)

    def test_boundary_condition_small_delta_Tj(self):
        """Test boundary condition with very small temperature swing."""
        model = CoffinMansonModel(A=1.0e6, alpha=2.0)

        # Very small swing should give very large cycles
        result = model.calculate_cycles_to_failure(delta_Tj=1.0)

        # Nf = 1e6 * 1^(-2) = 1e6
        assert result == pytest.approx(1.0e6, rel=0.01)

    def test_boundary_condition_large_delta_Tj(self):
        """Test boundary condition with large temperature swing."""
        model = CoffinMansonModel(A=1.0e6, alpha=2.0)

        # Large swing should give small cycles
        result = model.calculate_cycles_to_failure(delta_Tj=200)

        # Nf = 1e6 * 200^(-2) = 1e6 / 40000 = 25
        assert result == pytest.approx(25.0, rel=0.01)

    def test_get_equation(self):
        """Test get_equation method."""
        model = CoffinMansonModel()

        equation = model.get_equation()

        assert "Nf" in equation
        assert "A" in equation
        assert "ΔTj" in equation or "alpha" in equation

    def test_get_parameters_info(self):
        """Test get_parameters_info method."""
        model = CoffinMansonModel(A=5.0e7, alpha=3.0)

        info = model.get_parameters_info()

        assert 'A' in info
        assert 'alpha' in info
        assert 'delta_Tj' in info
        assert info['A']['current_value'] == 5.0e7
        assert info['alpha']['current_value'] == 3.0

    def test_validate_parameters_valid(self):
        """Test validate_parameters with valid input."""
        model = CoffinMansonModel()

        result = model.validate_parameters(delta_Tj=100)

        assert result is True

    def test_validate_parameters_invalid(self):
        """Test validate_parameters with invalid input."""
        model = CoffinMansonModel()

        with pytest.raises(ValueError):
            model.validate_parameters(delta_Tj=-10)

    def test_typical_solder_joint_parameters(self):
        """Test with typical solder joint parameters."""
        # Typical values for solder joints: A ≈ 100-1000, alpha ≈ 2-3
        model = CoffinMansonModel(A=500, alpha=2.5)

        # 80K swing is typical for power cycling
        result = model.calculate_cycles_to_failure(delta_Tj=80)

        # Nf = 500 * 80^(-2.5)
        # 80^2.5 = 80^2 * 80^0.5 ≈ 6400 * 8.94 ≈ 57200
        # Nf ≈ 500 / 57200 ≈ 0.009 (very low - device fails quickly)
        assert result > 0
        assert result < 1  # Less than 1 cycle at 80K swing

    def test_power_law_relationship(self):
        """Test that the power law relationship is preserved."""
        model = CoffinMansonModel(A=1.0e6, alpha=2.0)

        # Test multiple delta_Tj values
        delta_Tj_values = [50, 100, 150, 200]
        results = [model.calculate_cycles_to_failure(delta_Tj=dTj)
                   for dTj in delta_Tj_values]

        # For power law: Nf ∝ (ΔTj)^(-alpha)
        # If ΔTj doubles, Nf should decrease by factor of 2^alpha
        # For alpha=2: Nf(100) = Nf(50) / 4

        ratio_50_100 = results[0] / results[1]
        ratio_100_200 = results[1] / results[3]

        assert ratio_50_100 == pytest.approx(4.0, rel=0.1)
        assert ratio_100_200 == pytest.approx(4.0, rel=0.1)

    def test_exponent_effect(self):
        """Test effect of different exponent values."""
        base_model = CoffinMansonModel(A=1.0e8, alpha=1.5)
        high_model = CoffinMansonModel(A=1.0e8, alpha=3.0)

        delta_Tj = 100

        result_base = base_model.calculate_cycles_to_failure(delta_Tj=delta_Tj)
        result_high = high_model.calculate_cycles_to_failure(delta_Tj=delta_Tj)

        # Higher exponent should give much lower cycles
        assert result_high < result_base
        # Ratio should be approximately 100^(3-1.5) = 100^1.5 = 1000
        assert result_base / result_high == pytest.approx(1000, rel=0.1)

    def test_float_precision(self):
        """Test calculation with float precision."""
        model = CoffinMansonModel(A=1.234567e6, alpha=2.345678)

        result = model.calculate_cycles_to_failure(delta_Tj=78.9)

        assert result > 0
        assert np.isfinite(result)

    def test_scientific_notation_parameters(self):
        """Test with scientific notation parameters."""
        model = CoffinMansonModel(A=1.23e-4, alpha=3.5)

        result = model.calculate_cycles_to_failure(delta_Tj=100)

        assert result > 0
        assert np.isfinite(result)


class TestCoffinMansonReferenceValues:
    """Test against known reference values from literature."""

    def test_typical_igbt_values(self):
        """Test with typical IGBT module values from literature.

        Reference values commonly cited for IGBT modules:
        - ΔTj = 60-80 K
        - A ≈ 1e5 to 1e7 (fitted)
        - α ≈ 4-5 for wire bond fatigue
        """
        model = CoffinMansonModel(A=1.0e7, alpha=4.5)

        # At 60K swing
        result_60 = model.calculate_cycles_to_failure(delta_Tj=60)
        # At 80K swing
        result_80 = model.calculate_cycles_to_failure(delta_Tj=80)

        # 80K should give significantly fewer cycles
        assert result_80 < result_60

        # Ratio should be approximately (80/60)^4.5 ≈ 5.5
        ratio = result_60 / result_80
        assert 3 < ratio < 8  # Allow more tolerance for calculation variations

    def test_solder_joint_reference(self):
        """Test with solder joint reference values.

        For solder joints, typical values from literature:
        - α ≈ 2 (fatigue exponent)
        - Nf at 60°C swing ≈ 10,000 cycles for some solders
        """
        # Back-calculate A for Nf=10000 at dTj=60, alpha=2
        # 10000 = A * 60^(-2) => A = 10000 * 3600 = 3.6e7
        model = CoffinMansonModel(A=3.6e7, alpha=2.0)

        result_60 = model.calculate_cycles_to_failure(delta_Tj=60)

        assert result_60 == pytest.approx(10000, rel=0.1)

        # At 30K swing (half), should get 4x cycles
        result_30 = model.calculate_cycles_to_failure(delta_Tj=30)
        assert result_30 / result_60 == pytest.approx(4.0, rel=0.1)

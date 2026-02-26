"""
Coffin-Manson Model for lifetime prediction.

The Coffin-Manson model is a fundamental model for fatigue life prediction
based on plastic strain range. In power electronics, it relates temperature
swing to number of cycles to failure.

Equation:
    Nf = A × (ΔTj)^(-α)

Where:
    Nf: Number of cycles to failure
    A: Coefficient (material/device specific)
    ΔTj: Junction temperature swing (K)
    α: Temperature swing exponent
"""

from typing import Dict, Any
import logging

from app.core.models.model_base import LifetimeModelBase


logger = logging.getLogger(__name__)


class CoffinMansonModel(LifetimeModelBase):
    """
    Coffin-Manson lifetime model implementation.

    This model relates temperature swing to number of cycles to failure
    using a power law relationship. It is one of the simplest and most
    widely used models for power module lifetime prediction.

    Typical values:
        A: 10^2 to 10^8 (depends on device technology)
        α: 1.0 to 6.0 (typically 2-5 for solder joints)
        ΔTj: 10-200 K (typical operating range)
    """

    # Default parameter values (can be overridden per device)
    DEFAULT_A: float = 1.0e6
    DEFAULT_ALPHA: float = 2.0

    def __init__(self, A: float = DEFAULT_A, alpha: float = DEFAULT_ALPHA):
        """
        Initialize Coffin-Manson model with default parameters.

        Args:
            A: Coefficient (default: 1.0e6)
            alpha: Temperature swing exponent (default: 2.0)
        """
        self.A = A
        self.alpha = alpha

    def get_model_name(self) -> str:
        """Return the model name."""
        return "Coffin-Manson"

    def calculate_cycles_to_failure(self, **params) -> float:
        """
        Calculate number of cycles to failure using Coffin-Manson model.

        Equation: Nf = A × (ΔTj)^(-α)

        Args:
            **params: Must contain either:
                - delta_Tj (or dTj, delta_T_j): Temperature swing in K
                - Can also include A and alpha to override defaults

        Returns:
            float: Number of cycles to failure

        Raises:
            ValueError: If delta_Tj is missing or invalid
        """
        # Get coefficient and exponent (allow override via params)
        A = params.get("A", self.A)
        alpha = params.get("alpha", params.get("α", self.alpha))

        # Get temperature swing (support multiple parameter names)
        delta_Tj = params.get(
            "delta_Tj",
            params.get("dTj", params.get("delta_T_j", None))
        )

        # Validate parameters
        if delta_Tj is None:
            raise ValueError(
                "Temperature swing 'delta_Tj' (or 'dTj', 'delta_T_j') "
                "must be provided"
            )

        delta_Tj = float(delta_Tj)
        self._validate_positive(delta_Tj, "delta_Tj")
        self._validate_positive(A, "A")
        self._validate_positive(alpha, "alpha")

        # Calculate cycles to failure
        try:
            Nf = A * (delta_Tj ** (-alpha))
        except ZeroDivisionError:
            raise ValueError(
                f"Cannot calculate with delta_Tj={delta_Tj} and alpha={alpha}"
            )

        logger.debug(
            f"{self.get_model_name()}: Nf={Nf:.2e} cycles "
            f"(A={A:.2e}, α={alpha:.3f}, ΔTj={delta_Tj:.2f} K)"
        )

        return Nf

    def validate_parameters(self, **params) -> bool:
        """
        Validate input parameters for Coffin-Manson model.

        Args:
            **params: Parameters to validate

        Returns:
            bool: True if all parameters are valid

        Raises:
            ValueError: If any parameter is invalid
        """
        delta_Tj = params.get(
            "delta_Tj",
            params.get("dTj", params.get("delta_T_j", None))
        )

        if delta_Tj is not None:
            delta_Tj = float(delta_Tj)
            self._validate_positive(delta_Tj, "delta_Tj")

        A = params.get("A", self.A)
        alpha = params.get("alpha", params.get("α", self.alpha))

        self._validate_positive(A, "A")
        self._validate_positive(alpha, "alpha")

        return True

    def get_equation(self) -> str:
        """Return the mathematical equation as a string."""
        return "Nf = A × (ΔTj)^(-α)"

    def get_parameters_info(self) -> Dict[str, Any]:
        """
        Get information about model parameters.

        Returns:
            Dict with parameter descriptions and typical ranges
        """
        return {
            "A": {
                "description": "Coefficient (material/device specific)",
                "typical_range": "1e2 to 1e8",
                "current_value": self.A
            },
            "alpha": {
                "description": "Temperature swing exponent",
                "typical_range": "1.0 to 6.0 (typically 2-5)",
                "current_value": self.alpha,
                "alternative_names": ["α"]
            },
            "delta_Tj": {
                "description": "Junction temperature swing",
                "typical_range": "10-200 K",
                "unit": "K",
                "alternative_names": ["dTj", "delta_T_j"]
            }
        }

"""
Coffin-Manson-Arrhenius Model for lifetime prediction.

This model combines the Coffin-Manson relationship (temperature swing effect)
with the Arrhenius equation (temperature dependence of failure rate).
It accounts for both thermal cycling fatigue and mean temperature effects.

Equation:
    Nf = A × (ΔTj)^(-α) × exp(Ea / (kB × Tj_mean))

Where:
    Nf: Number of cycles to failure
    A: Pre-exponential coefficient
    ΔTj: Junction temperature swing (K)
    α: Temperature swing exponent
    Ea: Activation energy (eV)
    kB: Boltzmann constant = 8.617e-5 eV/K
    Tj_mean: Mean junction temperature (K)
"""

from typing import Dict, Any
import logging

from app.core.models.model_base import LifetimeModelBase


logger = logging.getLogger(__name__)

# Boltzmann constant in eV/K
BOLTZMANN_CONSTANT_EV_PER_K = 8.617e-5


class CoffinMansonArrheniusModel(LifetimeModelBase):
    """
    Coffin-Manson-Arrhenius lifetime model implementation.

    This model extends the basic Coffin-Manson model by incorporating
    temperature-dependent acceleration through the Arrhenius term.
    It captures both the effect of temperature swing amplitude and
    the effect of mean operating temperature.

    Typical values:
        A: 10^2 to 10^9
        α: 1.0 to 6.0
        Ea: 0.1 to 1.5 eV (typical for solder fatigue: 0.5-0.9 eV)
        ΔTj: 10-200 K
        Tj_mean: 273-473 K (0-200°C)
    """

    # Default parameter values
    DEFAULT_A: float = 1.0e6
    DEFAULT_ALPHA: float = 2.0
    DEFAULT_EA: float = 0.8  # eV, typical for solder joints

    def __init__(
        self,
        A: float = DEFAULT_A,
        alpha: float = DEFAULT_ALPHA,
        Ea: float = DEFAULT_EA
    ):
        """
        Initialize Coffin-Manson-Arrhenius model with default parameters.

        Args:
            A: Pre-exponential coefficient (default: 1.0e6)
            alpha: Temperature swing exponent (default: 2.0)
            Ea: Activation energy in eV (default: 0.8)
        """
        self.A = A
        self.alpha = alpha
        self.Ea = Ea

    def get_model_name(self) -> str:
        """Return the model name."""
        return "Coffin-Manson-Arrhenius"

    def calculate_cycles_to_failure(self, **params) -> float:
        """
        Calculate number of cycles to failure using Coffin-Manson-Arrhenius model.

        Equation: Nf = A × (ΔTj)^(-α) × exp(Ea / (kB × Tj_mean))

        Args:
            **params: Must contain:
                - delta_Tj: Temperature swing in K
                - Tj_mean: Mean junction temperature in K
                Can also include A, alpha, Ea to override defaults

        Returns:
            float: Number of cycles to failure

        Raises:
            ValueError: If required parameters are missing or invalid
        """
        # Get model parameters (allow override via params)
        A = params.get("A", self.A)
        alpha = params.get("alpha", params.get("α", self.alpha))
        Ea = params.get("Ea", params.get("activation_energy", self.Ea))

        # Get temperature parameters
        delta_Tj = params.get(
            "delta_Tj",
            params.get("dTj", params.get("delta_T_j", None))
        )
        Tj_mean = params.get(
            "Tj_mean",
            params.get("T_mean", params.get("mean_Tj", None))
        )

        # Validate presence of required parameters
        if delta_Tj is None:
            raise ValueError(
                "Temperature swing 'delta_Tj' (or 'dTj', 'delta_T_j') "
                "must be provided"
            )
        if Tj_mean is None:
            raise ValueError(
                "Mean temperature 'Tj_mean' (or 'T_mean', 'mean_Tj') "
                "must be provided"
            )

        # Convert to float
        delta_Tj = float(delta_Tj)
        Tj_mean = float(Tj_mean)

        # Validate parameters
        self._validate_positive(delta_Tj, "delta_Tj")
        self._validate_positive(Tj_mean, "Tj_mean")
        self._validate_positive(A, "A")
        self._validate_positive(alpha, "alpha")
        self._validate_positive(Ea, "Ea")

        # Calculate cycles to failure
        try:
            # Arrhenius term
            arrhenius_term = (Ea / (BOLTZMANN_CONSTANT_EV_PER_K * Tj_mean))
            arrhenius_factor = arrhenius_term  # = Ea / (kB * Tj_mean)

            # Coffin-Manson term
            coffin_manson_factor = delta_Tj ** (-alpha)

            # Combined model
            Nf = A * coffin_manson_factor * arrhenius_factor
        except ZeroDivisionError as e:
            raise ValueError(
                f"Division by zero in calculation: delta_Tj={delta_Tj}, "
                f"Tj_mean={Tj_mean}"
            ) from e

        logger.debug(
            f"{self.get_model_name()}: Nf={Nf:.2e} cycles "
            f"(A={A:.2e}, α={alpha:.3f}, Ea={Ea:.3f} eV, "
            f"ΔTj={delta_Tj:.2f} K, Tj_mean={Tj_mean:.2f} K)"
        )

        return Nf

    def validate_parameters(self, **params) -> bool:
        """
        Validate input parameters for Coffin-Manson-Arrhenius model.

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
        Tj_mean = params.get(
            "Tj_mean",
            params.get("T_mean", params.get("mean_Tj", None))
        )

        if delta_Tj is not None:
            self._validate_positive(float(delta_Tj), "delta_Tj")

        if Tj_mean is not None:
            self._validate_positive(float(Tj_mean), "Tj_mean")

        A = params.get("A", self.A)
        alpha = params.get("alpha", params.get("α", self.alpha))
        Ea = params.get("Ea", params.get("activation_energy", self.Ea))

        self._validate_positive(A, "A")
        self._validate_positive(alpha, "alpha")
        self._validate_positive(Ea, "Ea")

        return True

    def get_equation(self) -> str:
        """Return the mathematical equation as a string."""
        return "Nf = A × (ΔTj)^(-α) × exp(Ea / (kB × Tj_mean))"

    def get_parameters_info(self) -> Dict[str, Any]:
        """
        Get information about model parameters.

        Returns:
            Dict with parameter descriptions and typical ranges
        """
        return {
            "A": {
                "description": "Pre-exponential coefficient",
                "typical_range": "1e2 to 1e9",
                "current_value": self.A
            },
            "alpha": {
                "description": "Temperature swing exponent",
                "typical_range": "1.0 to 6.0",
                "current_value": self.alpha,
                "alternative_names": ["α"]
            },
            "Ea": {
                "description": "Activation energy",
                "typical_range": "0.1 to 1.5 eV (solder: 0.5-0.9 eV)",
                "unit": "eV",
                "current_value": self.Ea,
                "alternative_names": ["activation_energy"]
            },
            "delta_Tj": {
                "description": "Junction temperature swing",
                "typical_range": "10-200 K",
                "unit": "K",
                "alternative_names": ["dTj", "delta_T_j"]
            },
            "Tj_mean": {
                "description": "Mean junction temperature",
                "typical_range": "273-473 K (0-200°C)",
                "unit": "K",
                "alternative_names": ["T_mean", "mean_Tj"]
            },
            "kB": {
                "description": "Boltzmann constant",
                "value": BOLTZMANN_CONSTANT_EV_PER_K,
                "unit": "eV/K"
            }
        }

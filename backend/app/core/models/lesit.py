"""
LESIT Model for lifetime prediction.

The LESIT model was developed in the European LESIT (Life Time Estimation
for Integrated high power Semiconductor Thinning) project. It is an
empirical model for power module lifetime prediction under thermal cycling.

Equation:
    Nf = A × (ΔTj)^α × exp(Q / (R × Tj_min))

Where:
    Nf: Number of cycles to failure
    A: Coefficient (material/device specific)
    ΔTj: Junction temperature swing (K)
    α: Temperature swing exponent
    Q: Activation energy (J/mol)
    R: Gas constant = 8.314 J/(mol·K)
    Tj_min: Minimum junction temperature (K)

Note:
    The LESIT project was a collaborative research project involving
    European industries and research institutions focused on reliability
    of power semiconductor modules.
"""

from typing import Dict, Any
import logging
import math

from app.core.models.model_base import LifetimeModelBase


logger = logging.getLogger(__name__)

# Gas constant in J/(mol·K)
GAS_CONSTANT = 8.314


class LESITModel(LifetimeModelBase):
    """
    LESIT lifetime model implementation.

    The LESIT model was developed from extensive power cycling tests on
    power semiconductor modules. Unlike some other models, it uses the
    minimum junction temperature in the Arrhenius term rather than
    the mean or maximum temperature.

    The model structure is similar to Coffin-Manson-Arrhenius but uses
    Tj_min instead of Tj_mean, which can provide different predictions
    depending on the temperature cycle profile.

    Typical values:
        A: 10^3 to 10^9 (device specific)
        α: -5 to -2 (negative exponent, typically around -3 to -4)
        Q: 0.1 to 1.5 eV converted to J/mol (typical: 0.5-0.9 eV)
        ΔTj: 10-150 K
        Tj_min: 273-423 K (0-150°C)

    References:
        LESIT project final reports and publications, mid-1990s.
        Note: α in LESIT model is typically negative (unlike Coffin-Manson
        where it's often written with positive exponent in denominator).
    """

    # Default parameter values
    DEFAULT_A: float = 1.0e6
    DEFAULT_ALPHA: float = -3.5  # Negative exponent (LESIT convention)
    DEFAULT_Q: float = 0.8  # eV, will be converted to J/mol internally

    # eV to J/mol conversion factor
    EV_TO_JOULE_PER_MOL = 96485  # Faraday constant ≈ e * N_A

    def __init__(
        self,
        A: float = DEFAULT_A,
        alpha: float = DEFAULT_ALPHA,
        Q_eV: float = DEFAULT_Q
    ):
        """
        Initialize LESIT model with default parameters.

        Args:
            A: Coefficient (default: 1.0e6)
            alpha: Temperature swing exponent (default: -3.5)
            Q_eV: Activation energy in eV (default: 0.8)
        """
        self.A = A
        self.alpha = alpha
        self.Q_eV = Q_eV
        # Convert eV to J/mol for calculation with R
        self.Q_joule_per_mol = Q_eV * self.EV_TO_JOULE_PER_MOL

    def get_model_name(self) -> str:
        """Return the model name."""
        return "LESIT"

    def calculate_cycles_to_failure(self, **params) -> float:
        """
        Calculate number of cycles to failure using LESIT model.

        Equation: Nf = A × (ΔTj)^α × exp(Q / (R × Tj_min))

        Note: The LESIT model uses the minimum junction temperature in
        the Arrhenius term, unlike many other models that use mean or
        maximum temperature.

        Args:
            **params: Must contain:
                - delta_Tj: Temperature swing in K
                - Tj_min: Minimum junction temperature in K
                Can also include A, alpha, Q to override defaults

        Returns:
            float: Number of cycles to failure

        Raises:
            ValueError: If required parameters are missing or invalid
        """
        # Get model parameters (allow override via params)
        A = params.get("A", self.A)
        alpha = params.get("alpha", params.get("α", self.alpha))

        # Get activation energy (support eV or J/mol)
        Q_input = params.get("Q", params.get("activation_energy", None))
        if Q_input is not None:
            Q_input = float(Q_input)
            # Assume Q is in eV unless very large (then assume J/mol)
            if Q_input > 1000:  # Likely J/mol
                Q_joule_per_mol = Q_input
            else:  # Assume eV
                Q_joule_per_mol = Q_input * self.EV_TO_JOULE_PER_MOL
        else:
            Q_joule_per_mol = self.Q_joule_per_mol

        # Get temperature parameters
        delta_Tj = params.get(
            "delta_Tj",
            params.get("dTj", params.get("delta_T_j", None))
        )
        Tj_min = params.get(
            "Tj_min",
            params.get("T_min", params.get("min_Tj", None))
        )

        # Validate presence of required parameters
        if delta_Tj is None:
            raise ValueError(
                "Temperature swing 'delta_Tj' (or 'dTj', 'delta_T_j') "
                "must be provided"
            )
        if Tj_min is None:
            raise ValueError(
                "Minimum temperature 'Tj_min' (or 'T_min', 'min_Tj') "
                "must be provided"
            )

        # Convert to float
        delta_Tj = float(delta_Tj)
        Tj_min = float(Tj_min)

        # Validate parameters
        self._validate_positive(delta_Tj, "delta_Tj")
        self._validate_positive(Tj_min, "Tj_min")
        self._validate_positive(A, "A")

        # alpha can be negative (typical for LESIT)
        if alpha == 0:
            logger.warning("alpha=0 means temperature swing has no effect")

        # Calculate cycles to failure
        try:
            # Temperature swing term (power law with negative exponent typical)
            temp_swing_term = delta_Tj ** alpha

            # Arrhenius term using minimum temperature
            arrhenius_term = Q_joule_per_mol / (GAS_CONSTANT * Tj_min)
            arrhenius_factor = math.exp(arrhenius_term)

            # Combined model
            Nf = A * temp_swing_term * arrhenius_factor

        except ZeroDivisionError as e:
            raise ValueError(
                f"Division by zero in calculation: delta_Tj={delta_Tj}, "
                f"Tj_min={Tj_min}"
            ) from e

        logger.debug(
            f"{self.get_model_name()}: Nf={Nf:.2e} cycles "
            f"(A={A:.2e}, α={alpha:.3f}, Q={Q_joule_per_mol:.2e} J/mol, "
            f"ΔTj={delta_Tj:.2f} K, Tj_min={Tj_min:.2f} K)"
        )

        return Nf

    def validate_parameters(self, **params) -> bool:
        """
        Validate input parameters for LESIT model.

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
        Tj_min = params.get(
            "Tj_min",
            params.get("T_min", params.get("min_Tj", None))
        )

        if delta_Tj is not None:
            self._validate_positive(float(delta_Tj), "delta_Tj")

        if Tj_min is not None:
            self._validate_positive(float(Tj_min), "Tj_min")

        A = params.get("A", self.A)
        alpha = params.get("alpha", params.get("α", self.alpha))

        self._validate_positive(A, "A")

        return True

    def get_equation(self) -> str:
        """Return the mathematical equation as a string."""
        return "Nf = A × (ΔTj)^α × exp(Q / (R × Tj_min))"

    def get_parameters_info(self) -> Dict[str, Any]:
        """
        Get information about model parameters.

        Returns:
            Dict with parameter descriptions and typical ranges
        """
        return {
            "A": {
                "description": "Coefficient (material/device specific)",
                "typical_range": "1e3 to 1e9",
                "current_value": self.A
            },
            "alpha": {
                "description": "Temperature swing exponent (typically negative)",
                "typical_range": "-5 to -2 (typically -3 to -4)",
                "current_value": self.alpha,
                "note": "LESIT convention: alpha is negative",
                "alternative_names": ["α"]
            },
            "Q": {
                "description": "Activation energy",
                "typical_range": "0.1 to 1.5 eV (typical: 0.5-0.9 eV)",
                "unit": "eV (internally converted to J/mol)",
                "current_value_eV": self.Q_eV,
                "current_value_J_per_mol": self.Q_joule_per_mol,
                "alternative_names": ["activation_energy"]
            },
            "delta_Tj": {
                "description": "Junction temperature swing",
                "typical_range": "10-150 K",
                "unit": "K",
                "alternative_names": ["dTj", "delta_T_j"]
            },
            "Tj_min": {
                "description": "Minimum junction temperature",
                "typical_range": "273-423 K (0-150°C)",
                "unit": "K",
                "note": "LESIT uses Tj_min (not Tj_mean or Tj_max)",
                "alternative_names": ["T_min", "min_Tj"]
            },
            "R": {
                "description": "Gas constant",
                "value": GAS_CONSTANT,
                "unit": "J/(mol·K)"
            }
        }

    def get_activation_energy_eV(self) -> float:
        """Return activation energy in eV."""
        return self.Q_eV

    def get_activation_energy_J_per_mol(self) -> float:
        """Return activation energy in J/mol."""
        return self.Q_joule_per_mol

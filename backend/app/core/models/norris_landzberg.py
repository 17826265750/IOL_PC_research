"""
Norris-Landzberg Model for lifetime prediction.

The Norris-Landzberg model is an empirical model developed by Intel
for solder joint fatigue prediction. It accounts for temperature swing,
frequency, and maximum temperature effects.

Equation:
    Nf = A × (ΔTj)^(-α) × f^β × exp(Ea / (kB × Tj_max))

Where:
    Nf: Number of cycles to failure
    A: Coefficient (material/device specific)
    ΔTj: Junction temperature swing (K)
    f: Cycling frequency (Hz)
    α: Temperature swing exponent
    β: Frequency exponent
    Ea: Activation energy (eV)
    kB: Boltzmann constant = 8.617e-5 eV/K
    Tj_max: Maximum junction temperature (K)
"""

from typing import Dict, Any
import logging
import math

from app.core.models.model_base import LifetimeModelBase


logger = logging.getLogger(__name__)

# Boltzmann constant in eV/K
BOLTZMANN_CONSTANT_EV_PER_K = 8.617e-5


class NorrisLandzbergModel(LifetimeModelBase):
    """
    Norris-Landzberg lifetime model implementation.

    The Norris-Landzberg model is specifically designed for solder joint
    fatigue in electronic components. It was developed by Intel based
    on extensive testing of lead solder joints.

    The model considers:
    - Temperature swing amplitude (fatigue damage)
    - Cycling frequency (time-dependent effects)
    - Maximum temperature (Arrhenius acceleration)

    Typical values:
        A: 10^2 to 10^8 (device specific)
        α: 1.5 to 3.0 (typically ~2.0 for solder)
        β: -0.5 to 0.5 (often ~1/3 for creep-dominated failure)
        Ea: 0.3 to 1.2 eV (typically ~0.5 eV for solder)
        f: 10^-4 to 10 Hz (0.36 mHz to 10 Hz typical)
        ΔTj: 10-150 K
        Tj_max: 273-473 K (0-200°C)

    References:
        Norris, K.C. and Landzberg, A.H., "Reliability of Controlled
        Collapse Interconnections", IBM Journal of Research and
        Development, Vol. 13, No. 3, pp. 266-271, 1969.
    """

    # Default parameter values
    DEFAULT_A: float = 1.0e6
    DEFAULT_ALPHA: float = 2.0
    DEFAULT_BETA: float = 0.333  # ~1/3 for creep-dominated failure
    DEFAULT_EA: float = 0.5  # eV, typical for solder

    def __init__(
        self,
        A: float = DEFAULT_A,
        alpha: float = DEFAULT_ALPHA,
        beta: float = DEFAULT_BETA,
        Ea: float = DEFAULT_EA
    ):
        """
        Initialize Norris-Landzberg model with default parameters.

        Args:
            A: Coefficient (default: 1.0e6)
            alpha: Temperature swing exponent (default: 2.0)
            beta: Frequency exponent (default: 0.333)
            Ea: Activation energy in eV (default: 0.5)
        """
        self.A = A
        self.alpha = alpha
        self.beta = beta
        self.Ea = Ea

    def get_model_name(self) -> str:
        """Return the model name."""
        return "Norris-Landzberg"

    def calculate_cycles_to_failure(self, **params) -> float:
        """
        Calculate number of cycles to failure using Norris-Landzberg model.

        Equation: Nf = A × (ΔTj)^(-α) × f^β × exp(Ea / (kB × Tj_max))

        Args:
            **params: Must contain:
                - delta_Tj: Temperature swing in K
                - f: Cycling frequency in Hz
                - Tj_max: Maximum junction temperature in K
                Can also include A, alpha, beta, Ea to override defaults

        Returns:
            float: Number of cycles to failure

        Raises:
            ValueError: If required parameters are missing or invalid
        """
        # Get model parameters (allow override via params)
        A = params.get("A", self.A)
        alpha = params.get("alpha", params.get("α", self.alpha))
        beta = params.get("beta", params.get("β", self.beta))
        Ea = params.get("Ea", params.get("activation_energy", self.Ea))

        # Get temperature and frequency parameters
        delta_Tj = params.get(
            "delta_Tj",
            params.get("dTj", params.get("delta_T_j", None))
        )
        f = params.get("f", params.get("frequency", None))
        Tj_max = params.get(
            "Tj_max",
            params.get("T_max", params.get("max_Tj", None))
        )

        # Validate presence of required parameters
        if delta_Tj is None:
            raise ValueError(
                "Temperature swing 'delta_Tj' (or 'dTj', 'delta_T_j') "
                "must be provided"
            )
        if f is None:
            raise ValueError(
                "Cycling frequency 'f' (or 'frequency') must be provided"
            )
        if Tj_max is None:
            raise ValueError(
                "Maximum temperature 'Tj_max' (or 'T_max', 'max_Tj') "
                "must be provided"
            )

        # Convert to float
        delta_Tj = float(delta_Tj)
        f = float(f)
        Tj_max = float(Tj_max)

        # Validate parameters
        self._validate_positive(delta_Tj, "delta_Tj")
        self._validate_positive(f, "f")
        self._validate_positive(Tj_max, "Tj_max")
        self._validate_positive(A, "A")
        self._validate_positive(alpha, "alpha")
        self._validate_positive(Ea, "Ea")

        # beta can be negative or positive, but should not be zero
        if beta == 0:
            logger.warning("beta=0 means frequency has no effect")

        # Calculate cycles to failure
        try:
            # Temperature swing term (Coffin-Manson)
            temp_term = delta_Tj ** (-alpha)

            # Frequency term
            freq_term = f ** beta

            # Arrhenius term (temperature acceleration)
            arrhenius_factor = math.exp(Ea / (BOLTZMANN_CONSTANT_EV_PER_K * Tj_max))

            # Combined model
            Nf = A * temp_term * freq_term * arrhenius_factor
        except ZeroDivisionError as e:
            raise ValueError(
                f"Division by zero in calculation: delta_Tj={delta_Tj}, "
                f"f={f}, Tj_max={Tj_max}"
            ) from e

        logger.debug(
            f"{self.get_model_name()}: Nf={Nf:.2e} cycles "
            f"(A={A:.2e}, α={alpha:.3f}, β={beta:.3f}, Ea={Ea:.3f} eV, "
            f"ΔTj={delta_Tj:.2f} K, f={f:.4f} Hz, Tj_max={Tj_max:.2f} K)"
        )

        return Nf

    def validate_parameters(self, **params) -> bool:
        """
        Validate input parameters for Norris-Landzberg model.

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
        f = params.get("f", params.get("frequency", None))
        Tj_max = params.get(
            "Tj_max",
            params.get("T_max", params.get("max_Tj", None))
        )

        if delta_Tj is not None:
            self._validate_positive(float(delta_Tj), "delta_Tj")

        if f is not None:
            self._validate_positive(float(f), "f")

        if Tj_max is not None:
            self._validate_positive(float(Tj_max), "Tj_max")

        A = params.get("A", self.A)
        alpha = params.get("alpha", params.get("α", self.alpha))
        beta = params.get("beta", params.get("β", self.beta))
        Ea = params.get("Ea", params.get("activation_energy", self.Ea))

        self._validate_positive(A, "A")
        self._validate_positive(alpha, "alpha")
        self._validate_positive(Ea, "Ea")

        return True

    def get_equation(self) -> str:
        """Return the mathematical equation as a string."""
        return "Nf = A × (ΔTj)^(-α) × f^β × exp(Ea / (kB × Tj_max))"

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
                "typical_range": "1.5 to 3.0 (typically ~2.0)",
                "current_value": self.alpha,
                "alternative_names": ["α"]
            },
            "beta": {
                "description": "Frequency exponent",
                "typical_range": "-0.5 to 0.5 (often ~0.33)",
                "current_value": self.beta,
                "note": "~1/3 for creep-dominated failure",
                "alternative_names": ["β"]
            },
            "Ea": {
                "description": "Activation energy",
                "typical_range": "0.3 to 1.2 eV (typically ~0.5 eV)",
                "unit": "eV",
                "current_value": self.Ea,
                "alternative_names": ["activation_energy"]
            },
            "delta_Tj": {
                "description": "Junction temperature swing",
                "typical_range": "10-150 K",
                "unit": "K",
                "alternative_names": ["dTj", "delta_T_j"]
            },
            "f": {
                "description": "Cycling frequency",
                "typical_range": "1e-4 to 10 Hz",
                "unit": "Hz",
                "alternative_names": ["frequency"]
            },
            "Tj_max": {
                "description": "Maximum junction temperature",
                "typical_range": "273-473 K (0-200°C)",
                "unit": "K",
                "alternative_names": ["T_max", "max_Tj"]
            },
            "kB": {
                "description": "Boltzmann constant",
                "value": BOLTZMANN_CONSTANT_EV_PER_K,
                "unit": "eV/K"
            }
        }

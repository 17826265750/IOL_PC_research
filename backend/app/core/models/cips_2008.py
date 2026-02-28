"""
CIPS 2008 (Bayerer) Model for lifetime prediction.

功率模块寿命分析软件 - CIPS 2008寿命模型
Author: GSH

The CIPS 2008 model, also known as the Bayerer model, is a comprehensive
lifetime prediction model developed by Bayerer et al. (2008) for IGBT
modules. It accounts for multiple stress factors affecting power module
lifetime.

Equation:
    Nf = K × (ΔTj)^β1 × exp(β2/Tj_max) × t_on^β3 × I^β4 × V^β5 × D^β6

Where:
    Nf: Number of cycles to failure
    K: Technology coefficient
    ΔTj: Junction temperature swing (K)
    Tj_max: Maximum junction temperature (K)
    t_on: Heating time / on-time (s)
    I: Load current (A)
    V: Blocking voltage rating (V)
    D: Bond wire diameter (μm)
    β1~β6: Fitting exponents

Reference:
    Bayerer, R., et al., "Model for Power Cycling Lifetime of IGBT
    Modules - various factors influencing lifetime", CIPS 2008,
    6th International Conference on Integrated Power Electronic
    Systems, pp. 1-6, 2008.
"""

from typing import Dict, Any, Optional
import logging
import math

from app.core.models.model_base import LifetimeModelBase


logger = logging.getLogger(__name__)


class CIPS2008Model(LifetimeModelBase):
    """
    CIPS 2008 (Bayerer) lifetime model implementation.

    This is the main model for IGBT module lifetime prediction.
    It was developed based on extensive accelerated testing of IGBT
    modules from multiple manufacturers.

    The model comprehensively accounts for:
    - Temperature swing amplitude (thermal cycling fatigue)
    - Maximum temperature (Arrhenius acceleration)
    - Heating time / dwell time (creep effects)
    - Load current (current stress)
    - Voltage rating (device rating)
    - Bond wire diameter (construction parameter)

    Typical parameter values (from Bayerer et al. 2008):
        K: Technology dependent (requires fitting)
        β1: -4.423 (temperature swing exponent)
        β2: 1285 (maximum temperature exponent)
        β3: -0.462 (heating time exponent)
        β4: -0.716 (current exponent)
        β5: -0.761 (voltage exponent)
        β6: -0.5 (bond wire diameter exponent)

    Valid parameter ranges:
        ΔTj: 60-150 K
        Tj_max: 398-523 K (125-250°C)
        t_on: 1-60 s
        I: Device dependent (typically 10-200 A)
        V: 600-1700 V
        D: 100-400 μm
    """

    # Default parameter values from Bayerer et al. 2008
    DEFAULT_K: float = 1.0  # Technology coefficient (needs fitting)
    DEFAULT_BETA1: float = -4.423  # Temperature swing exponent
    DEFAULT_BETA2: float = 1285.0  # Max temperature exponent
    DEFAULT_BETA3: float = -0.462  # Heating time exponent
    DEFAULT_BETA4: float = -0.716  # Current exponent
    DEFAULT_BETA5: float = -0.761  # Voltage exponent
    DEFAULT_BETA6: float = -0.5  # Bond wire diameter exponent

    def __init__(
        self,
        K: float = DEFAULT_K,
        beta1: float = DEFAULT_BETA1,
        beta2: float = DEFAULT_BETA2,
        beta3: float = DEFAULT_BETA3,
        beta4: float = DEFAULT_BETA4,
        beta5: float = DEFAULT_BETA5,
        beta6: float = DEFAULT_BETA6
    ):
        """
        Initialize CIPS 2008 model with default parameters.

        Args:
            K: Technology coefficient (default: 1.0, requires fitting)
            beta1: Temperature swing exponent (default: -4.423)
            beta2: Max temperature exponent (default: 1285.0)
            beta3: Heating time exponent (default: -0.462)
            beta4: Current exponent (default: -0.716)
            beta5: Voltage exponent (default: -0.761)
            beta6: Bond wire diameter exponent (default: -0.5)
        """
        self.K = K
        self.beta1 = beta1
        self.beta2 = beta2
        self.beta3 = beta3
        self.beta4 = beta4
        self.beta5 = beta5
        self.beta6 = beta6

    def get_model_name(self) -> str:
        """Return the model name."""
        return "CIPS-2008"

    def calculate_cycles_to_failure(self, **params) -> float:
        """
        Calculate number of cycles to failure using CIPS 2008 model.

        Equation:
        Nf = K × (ΔTj)^β1 × exp(β2/Tj_max) × t_on^β3 × I^β4 × V^β5 × D^β6

        Args:
            **params: Must contain:
                - delta_Tj: Junction temperature swing in K
                - Tj_max: Maximum junction temperature in K
                - t_on: Heating time / on-time in s
                - I: Load current in A
                - V: Blocking voltage rating in V
                - D: Bond wire diameter in μm
                Can also include K, beta1-beta6 to override defaults

        Returns:
            float: Number of cycles to failure

        Raises:
            ValueError: If required parameters are missing or invalid
        """
        # Get model parameters (allow override via params)
        K = params.get("K", self.K)
        beta1 = params.get("beta1", params.get("β1", self.beta1))
        beta2 = params.get("beta2", params.get("β2", self.beta2))
        beta3 = params.get("beta3", params.get("β3", self.beta3))
        beta4 = params.get("beta4", params.get("β4", self.beta4))
        beta5 = params.get("beta5", params.get("β5", self.beta5))
        beta6 = params.get("beta6", params.get("β6", self.beta6))

        # Get required physical parameters
        delta_Tj = params.get(
            "delta_Tj",
            params.get("dTj", params.get("delta_T_j", None))
        )
        Tj_max = params.get(
            "Tj_max",
            params.get("T_max", params.get("max_Tj", None))
        )
        t_on = params.get("t_on", params.get("t_on_seconds", params.get("heating_time", None)))
        I = params.get("I", params.get("current", None))
        V = params.get("V", params.get("voltage", None))
        D = params.get("D", params.get("bond_wire_diameter", None))

        # Validate presence of required parameters
        missing_params = []
        if delta_Tj is None:
            missing_params.append("delta_Tj")
        if Tj_max is None:
            missing_params.append("Tj_max")
        if t_on is None:
            missing_params.append("t_on")
        if I is None:
            missing_params.append("I")
        if V is None:
            missing_params.append("V")
        if D is None:
            missing_params.append("D")

        if missing_params:
            raise ValueError(
                f"Missing required parameters: {', '.join(missing_params)}"
            )

        # Convert to float
        delta_Tj = float(delta_Tj)
        Tj_max = float(Tj_max)
        t_on = float(t_on)
        I = float(I)
        V = float(V)
        D = float(D)

        # Validate parameters
        self._validate_positive(delta_Tj, "delta_Tj")
        self._validate_positive(Tj_max, "Tj_max")
        self._validate_positive(t_on, "t_on")
        self._validate_positive(I, "I")
        self._validate_positive(V, "V")
        self._validate_positive(D, "D")
        self._validate_positive(K, "K")

        # Validate parameter ranges (warnings only)
        self._check_parameter_range(delta_Tj, "delta_Tj", 60, 150)
        self._check_parameter_range(Tj_max, "Tj_max", 398, 523)
        self._check_parameter_range(t_on, "t_on", 1, 60)
        self._check_parameter_range(V, "V", 600, 1700)
        self._check_parameter_range(D, "D", 100, 400)

        # Calculate cycles to failure using CIPS 2008 formula:
        # Nf = K × (ΔTj)^β1 × exp(β2/Tj_max) × t_on^β3 × I^β4 × V^β5 × D^β6
        try:
            Nf = (K *
                  (delta_Tj ** beta1) *
                  math.exp(beta2 / Tj_max) *
                  (t_on ** beta3) *
                  (I ** beta4) *
                  (V ** beta5) *
                  (D ** beta6))

        except ZeroDivisionError as e:
            raise ValueError(
                f"Division by zero in calculation: Tj_max={Tj_max}"
            ) from e
        except ValueError as e:
            raise ValueError(
                f"Math domain error in calculation: {e}"
            ) from e

        logger.debug(
            f"{self.get_model_name()}: Nf={Nf:.2e} cycles "
            f"(K={K:.2e}, ΔTj={delta_Tj:.2f} K, Tj_max={Tj_max:.2f} K, "
            f"t_on={t_on:.2f} s, I={I:.2f} A, V={V:.0f} V, D={D:.0f} μm)"
        )

        return Nf

    def _check_parameter_range(
        self,
        value: float,
        param_name: str,
        min_val: float,
        max_val: float
    ) -> None:
        """
        Check if parameter is within valid range (warning only).

        Args:
            value: Parameter value to check
            param_name: Name of the parameter
            min_val: Minimum recommended value
            max_val: Maximum recommended value
        """
        if value < min_val or value > max_val:
            logger.warning(
                f"Parameter '{param_name}'={value:.2f} is outside "
                f"recommended range [{min_val}, {max_val}]"
            )

    def validate_parameters(self, **params) -> bool:
        """
        Validate input parameters for CIPS 2008 model.

        Args:
            **params: Parameters to validate

        Returns:
            bool: True if all parameters are valid

        Raises:
            ValueError: If any parameter is invalid
        """
        # Get parameters if provided
        delta_Tj = params.get(
            "delta_Tj",
            params.get("dTj", params.get("delta_T_j", None))
        )
        Tj_max = params.get(
            "Tj_max",
            params.get("T_max", params.get("max_Tj", None))
        )
        t_on = params.get("t_on", params.get("t_on_seconds", None))
        I = params.get("I", params.get("current", None))
        V = params.get("V", params.get("voltage", None))
        D = params.get("D", params.get("bond_wire_diameter", None))

        # Validate if provided
        if delta_Tj is not None:
            self._validate_positive(float(delta_Tj), "delta_Tj")
        if Tj_max is not None:
            self._validate_positive(float(Tj_max), "Tj_max")
        if t_on is not None:
            self._validate_positive(float(t_on), "t_on")
        if I is not None:
            self._validate_positive(float(I), "I")
        if V is not None:
            self._validate_positive(float(V), "V")
        if D is not None:
            self._validate_positive(float(D), "D")

        # Validate model parameters
        K = params.get("K", self.K)
        self._validate_positive(K, "K")

        return True

    def get_equation(self) -> str:
        """Return the mathematical equation as a string."""
        return ("Nf = K × (ΔTj)^β1 × exp(β2/Tj_max) × t_on^β3 × "
                "I^β4 × V^β5 × D^β6")

    def get_parameters_info(self) -> Dict[str, Any]:
        """
        Get information about model parameters.

        Returns:
            Dict with parameter descriptions and typical ranges
        """
        return {
            "K": {
                "description": "Technology coefficient",
                "typical_range": "Device dependent (requires fitting)",
                "current_value": self.K,
                "note": "Needs to be fitted to test data"
            },
            "beta1": {
                "description": "Temperature swing exponent",
                "typical_value": -4.423,
                "current_value": self.beta1,
                "alternative_names": ["β1"]
            },
            "beta2": {
                "description": "Maximum temperature exponent",
                "typical_value": 1285.0,
                "current_value": self.beta2,
                "alternative_names": ["β2"]
            },
            "beta3": {
                "description": "Heating time exponent",
                "typical_value": -0.462,
                "current_value": self.beta3,
                "alternative_names": ["β3"]
            },
            "beta4": {
                "description": "Current exponent",
                "typical_value": -0.716,
                "current_value": self.beta4,
                "alternative_names": ["β4"]
            },
            "beta5": {
                "description": "Voltage exponent",
                "typical_value": -0.761,
                "current_value": self.beta5,
                "alternative_names": ["β5"]
            },
            "beta6": {
                "description": "Bond wire diameter exponent",
                "typical_value": -0.5,
                "current_value": self.beta6,
                "alternative_names": ["β6"]
            },
            "delta_Tj": {
                "description": "Junction temperature swing",
                "valid_range": "60-150 K",
                "unit": "K",
                "alternative_names": ["dTj", "delta_T_j"]
            },
            "Tj_max": {
                "description": "Maximum junction temperature",
                "valid_range": "398-523 K (125-250°C)",
                "unit": "K",
                "alternative_names": ["T_max", "max_Tj"]
            },
            "t_on": {
                "description": "Heating time / on-time",
                "valid_range": "1-60 s",
                "unit": "s",
                "alternative_names": ["t_on_seconds", "heating_time"]
            },
            "I": {
                "description": "Load current",
                "device_dependent": True,
                "unit": "A",
                "alternative_names": ["current"]
            },
            "V": {
                "description": "Blocking voltage rating",
                "valid_range": "600-1700 V",
                "unit": "V",
                "alternative_names": ["voltage"]
            },
            "D": {
                "description": "Bond wire diameter",
                "valid_range": "100-400 μm",
                "unit": "μm",
                "alternative_names": ["bond_wire_diameter"]
            }
        }

    def get_reference(self) -> str:
        """Return the reference for the model."""
        return (
            "Bayerer, R., et al., 'Model for Power Cycling Lifetime of "
            "IGBT Modules - various factors influencing lifetime', "
            "CIPS 2008, 6th International Conference on Integrated Power "
            "Electronic Systems, pp. 1-6, 2008."
        )

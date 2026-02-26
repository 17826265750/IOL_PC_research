"""
Abstract base class for all lifetime prediction models.

All lifetime models must inherit from this class and implement
the required abstract methods.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import logging


logger = logging.getLogger(__name__)


class LifetimeModelBase(ABC):
    """
    Abstract base class for lifetime prediction models.

    All concrete model implementations must inherit from this class
    and implement the abstract methods defined below.
    """

    @abstractmethod
    def calculate_cycles_to_failure(self, **params) -> float:
        """
        Calculate the number of cycles to failure.

        Args:
            **params: Model-specific parameters

        Returns:
            float: Number of cycles to failure (Nf)

        Raises:
            ValueError: If required parameters are missing or invalid
        """
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        """
        Return the name of the model.

        Returns:
            str: Model name identifier
        """
        pass

    def get_equation(self) -> str:
        """
        Return the model equation as a string.

        Returns:
            str: Model equation in human-readable format

        Default implementation returns an empty string.
        Override in subclasses to provide the actual equation.
        """
        return ""

    def validate_parameters(self, **params) -> bool:
        """
        Validate input parameters.

        Default implementation checks for positive values.
        Override in subclasses for model-specific validation.

        Args:
            **params: Parameters to validate

        Returns:
            bool: True if all parameters are valid

        Raises:
            ValueError: If any parameter is invalid
        """
        for key, value in params.items():
            if isinstance(value, (int, float)):
                if value < 0:
                    raise ValueError(
                        f"Parameter '{key}' must be non-negative, got {value}"
                    )
        return True

    def _get_required_param(
        self,
        params: Dict[str, Any],
        param_name: str,
        default: Optional[float] = None
    ) -> float:
        """
        Helper method to get a required parameter from params dict.

        Args:
            params: Parameters dictionary
            param_name: Name of the parameter to retrieve
            default: Default value if parameter not found

        Returns:
            float: Parameter value

        Raises:
            ValueError: If parameter is missing and no default provided
        """
        value = params.get(param_name, default)
        if value is None:
            raise ValueError(
                f"Required parameter '{param_name}' not provided "
                f"and no default value available"
            )
        return float(value)

    def _validate_positive(self, value: float, param_name: str) -> None:
        """
        Validate that a value is positive.

        Args:
            value: Value to validate
            param_name: Name of the parameter (for error message)

        Raises:
            ValueError: If value is not positive
        """
        if value <= 0:
            raise ValueError(
                f"Parameter '{param_name}' must be positive, got {value}"
            )

    def _validate_non_negative(self, value: float, param_name: str) -> None:
        """
        Validate that a value is non-negative.

        Args:
            value: Value to validate
            param_name: Name of the parameter (for error message)

        Raises:
            ValueError: If value is negative
        """
        if value < 0:
            raise ValueError(
                f"Parameter '{param_name}' must be non-negative, got {value}"
            )

    def _validate_range(
        self,
        value: float,
        param_name: str,
        min_val: Optional[float] = None,
        max_val: Optional[float] = None
    ) -> None:
        """
        Validate that a value is within a specified range.

        Args:
            value: Value to validate
            param_name: Name of the parameter (for error message)
            min_val: Minimum allowed value (None = no minimum)
            max_val: Maximum allowed value (None = no maximum)

        Raises:
            ValueError: If value is outside the specified range
        """
        if min_val is not None and value < min_val:
            raise ValueError(
                f"Parameter '{param_name}' must be >= {min_val}, got {value}"
            )
        if max_val is not None and value > max_val:
            raise ValueError(
                f"Parameter '{param_name}' must be <= {max_val}, got {value}"
            )

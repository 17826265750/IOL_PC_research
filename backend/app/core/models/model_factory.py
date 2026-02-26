"""
Factory pattern for lifetime model selection and instantiation.

The ModelFactory provides a centralized way to create and manage
lifetime prediction model instances. It supports model registration
and retrieval by name.
"""

from typing import Dict, Type, Optional, List
import logging

from app.core.models.model_base import LifetimeModelBase


logger = logging.getLogger(__name__)


class ModelFactory:
    """
    Factory for creating lifetime model instances.

    This class implements the factory pattern to allow dynamic model
    selection and instantiation. Models can be registered with a name
    and then retrieved using that name.

    Usage:
        # Register models (typically done at module import)
        ModelFactory.register_model("coffin-manson", CoffinMansonModel)
        ModelFactory.register_model("cips-2008", CIPS2008Model)

        # Get a model instance
        model = ModelFactory.get_model("cips-2008")
        cycles = model.calculate_cycles_to_failure(delta_Tj=100, ...)

        # List available models
        available = ModelFactory.list_models()
    """

    # Registry for model classes
    _models: Dict[str, Type[LifetimeModelBase]] = {}

    # Registry for model instances (singleton pattern)
    _instances: Dict[str, LifetimeModelBase] = {}

    @classmethod
    def register_model(
        cls,
        name: str,
        model_class: Type[LifetimeModelBase]
    ) -> None:
        """
        Register a model class with a name.

        Args:
            name: Name identifier for the model
            model_class: Model class (must inherit from LifetimeModelBase)

        Raises:
            TypeError: If model_class doesn't inherit from LifetimeModelBase
            ValueError: If name is already registered
        """
        if not issubclass(model_class, LifetimeModelBase):
            raise TypeError(
                f"Model class must inherit from LifetimeModelBase, "
                f"got {model_class.__name__}"
            )

        if name in cls._models:
            logger.warning(
                f"Model '{name}' is already registered. "
                f"Overwriting with {model_class.__name__}"
            )

        cls._models[name] = model_class
        cls._instances.pop(name, None)  # Clear any cached instance
        logger.debug(f"Registered model '{name}' -> {model_class.__name__}")

    @classmethod
    def unregister_model(cls, name: str) -> None:
        """
        Unregister a model.

        Args:
            name: Name of the model to unregister
        """
        if name in cls._models:
            del cls._models[name]
            cls._instances.pop(name, None)
            logger.debug(f"Unregistered model '{name}'")

    @classmethod
    def get_model(cls, name: str, use_singleton: bool = True) -> LifetimeModelBase:
        """
        Get a model instance by name.

        Args:
            name: Name of the model to retrieve
            use_singleton: If True, return cached instance (default)
                          If False, create new instance each time

        Returns:
            LifetimeModelBase: Model instance

        Raises:
            ValueError: If model name is not registered
        """
        if name not in cls._models:
            available = ", ".join(cls.list_models())
            raise ValueError(
                f"Unknown model: '{name}'. "
                f"Available models: {available}"
            )

        if use_singleton and name in cls._instances:
            return cls._instances[name]

        # Create new instance
        model_class = cls._models[name]
        instance = model_class()

        if use_singleton:
            cls._instances[name] = instance

        logger.debug(f"Created model instance: {name}")
        return instance

    @classmethod
    def create_model(
        cls,
        name: str,
        **kwargs
    ) -> LifetimeModelBase:
        """
        Create a new model instance with custom initialization parameters.

        This method creates a fresh instance (not using singleton cache)
        and passes initialization parameters to the model constructor.

        Args:
            name: Name of the model to create
            **kwargs: Initialization parameters for the model

        Returns:
            LifetimeModelBase: New model instance

        Raises:
            ValueError: If model name is not registered

        Example:
            model = ModelFactory.create_model(
                "coffin-manson",
                A=1.0e7,
                alpha=2.5
            )
        """
        if name not in cls._models:
            available = ", ".join(cls.list_models())
            raise ValueError(
                f"Unknown model: '{name}'. "
                f"Available models: {available}"
            )

        model_class = cls._models[name]
        instance = model_class(**kwargs)

        logger.debug(f"Created model instance with custom params: {name}")
        return instance

    @classmethod
    def list_models(cls) -> List[str]:
        """
        List all registered model names.

        Returns:
            List[str]: List of registered model names
        """
        return list(cls._models.keys())

    @classmethod
    def is_registered(cls, name: str) -> bool:
        """
        Check if a model is registered.

        Args:
            name: Model name to check

        Returns:
            bool: True if model is registered
        """
        return name in cls._models

    @classmethod
    def get_model_info(cls, name: str) -> Dict[str, str]:
        """
        Get information about a registered model.

        Args:
            name: Model name

        Returns:
            Dict with model information including name, class name,
            and available parameters

        Raises:
            ValueError: If model name is not registered
        """
        if name not in cls._models:
            raise ValueError(f"Unknown model: '{name}'")

        model_class = cls._models[name]

        # Create temporary instance to get info
        temp_instance = model_class()

        info = {
            "name": name,
            "class": model_class.__name__,
            "model_name": temp_instance.get_model_name(),
            "equation": temp_instance.get_equation(),
            "module": model_class.__module__,
        }

        # Add parameters info if available
        if hasattr(temp_instance, "get_parameters_info"):
            info["parameters"] = temp_instance.get_parameters_info()

        return info

    @classmethod
    def clear_cache(cls) -> None:
        """Clear all cached model instances."""
        cls._instances.clear()
        logger.debug("Cleared model instance cache")

    @classmethod
    def register_all(cls) -> None:
        """
        Register all available models.

        This method should be called during application initialization
        to register all available lifetime models.
        """
        # Import models here to avoid circular imports
        from app.core.models.coffin_manson import CoffinMansonModel
        from app.core.models.coffin_manson_arrhenius import CoffinMansonArrheniusModel
        from app.core.models.norris_landzberg import NorrisLandzbergModel
        from app.core.models.cips_2008 import CIPS2008Model
        from app.core.models.lesit import LESITModel

        # Register all models
        cls.register_model("coffin-manson", CoffinMansonModel)
        cls.register_model("coffin-manson-arrhenius", CoffinMansonArrheniusModel)
        cls.register_model("norris-landzberg", NorrisLandzbergModel)
        cls.register_model("cips-2008", CIPS2008Model)
        cls.register_model("lesit", LESITModel)

        logger.info(f"Registered {len(cls._models)} lifetime models")

    @classmethod
    def get_default_model(cls) -> str:
        """
        Get the default model name.

        Returns:
            str: Default model name (cips-2008)
        """
        return "cips-2008"


# Convenience functions for common operations
def get_model(name: str) -> LifetimeModelBase:
    """Convenience function to get a model instance."""
    return ModelFactory.get_model(name)


def create_model(name: str, **kwargs) -> LifetimeModelBase:
    """Convenience function to create a model with custom parameters."""
    return ModelFactory.create_model(name, **kwargs)


def list_models() -> List[str]:
    """Convenience function to list all available models."""
    return ModelFactory.list_models()

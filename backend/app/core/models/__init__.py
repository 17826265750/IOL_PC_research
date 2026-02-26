"""
CIPS 2008 lifetime model implementations.

This package provides various lifetime prediction models for power electronics,
specifically focusing on IGBT module reliability analysis.

Available Models:
- Coffin-Manson: Basic temperature swing model
- Coffin-Manson-Arrhenius: Temperature swing + mean temperature effects
- Norris-Landzberg: Temperature swing + frequency + max temperature
- CIPS-2008 (Bayerer): Comprehensive model with multiple stress factors
- LESIT: Temperature swing + minimum temperature effects

Usage:
    from app.core.models import get_model, list_models

    # List available models
    models = list_models()

    # Get a specific model
    model = get_model("cips-2008")
    cycles = model.calculate_cycles_to_failure(
        delta_Tj=100,
        Tj_max=423,
        t_on=5,
        I=50,
        V=1200,
        D=300
    )
"""

from app.core.models.model_base import LifetimeModelBase
from app.core.models.coffin_manson import CoffinMansonModel
from app.core.models.coffin_manson_arrhenius import CoffinMansonArrheniusModel
from app.core.models.norris_landzberg import NorrisLandzbergModel
from app.core.models.cips_2008 import CIPS2008Model
from app.core.models.lesit import LESITModel
from app.core.models.model_factory import (
    ModelFactory,
    get_model,
    create_model,
    list_models
)

# Register all models on import
ModelFactory.register_all()

__all__ = [
    # Base classes
    "LifetimeModelBase",
    "ModelFactory",

    # Model classes
    "CoffinMansonModel",
    "CoffinMansonArrheniusModel",
    "NorrisLandzbergModel",
    "CIPS2008Model",
    "LESITModel",

    # Convenience functions
    "get_model",
    "create_model",
    "list_models",
]

# Model name constants for type safety
MODEL_COFFIN_MASON = "coffin-manson"
MODEL_COFFIN_MASON_ARRHENIUS = "coffin-manson-arrhenius"
MODEL_NORRIS_LANDZBERG = "norris-landzberg"
MODEL_CIPS_2008 = "cips-2008"
MODEL_LESIT = "lesit"

ALL_MODELS = [
    MODEL_COFFIN_MASON,
    MODEL_COFFIN_MASON_ARRHENIUS,
    MODEL_NORRIS_LANDZBERG,
    MODEL_CIPS_2008,
    MODEL_LESIT,
]

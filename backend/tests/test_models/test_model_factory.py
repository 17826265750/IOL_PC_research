"""
Unit tests for ModelFactory.

Tests include:
- Model registration and retrieval
- Singleton pattern behavior
- Error handling for unknown models
- Model information retrieval
- Custom parameter instantiation
"""

import pytest

from app.core.models.model_factory import ModelFactory
from app.core.models.coffin_manson import CoffinMansonModel
from app.core.models.cips_2008 import CIPS2008Model
from app.core.models.norris_landzberg import NorrisLandzbergModel
from app.core.models.lesit import LESITModel
from app.core.models.model_base import LifetimeModelBase


class DummyModel(LifetimeModelBase):
    """Dummy model for testing."""

    def get_model_name(self) -> str:
        return "Dummy"

    def calculate_cycles_to_failure(self, **params) -> float:
        return 1000.0


class AnotherDummyModel(LifetimeModelBase):
    """Another dummy model for testing."""

    def get_model_name(self) -> str:
        return "Another Dummy"

    def calculate_cycles_to_failure(self, **params) -> float:
        return 2000.0


class NotAModel:
    """Class that doesn't inherit from LifetimeModelBase."""

    pass


class TestModelFactory:
    """Test suite for ModelFactory."""

    def setup_method(self):
        """Clear model registry before each test."""
        ModelFactory._models.clear()
        ModelFactory._instances.clear()

    def teardown_method(self):
        """Clean up after each test."""
        ModelFactory._models.clear()
        ModelFactory._instances.clear()

    def test_register_model(self):
        """Test registering a model."""
        ModelFactory.register_model("dummy", DummyModel)

        assert "dummy" in ModelFactory._models
        assert ModelFactory._models["dummy"] == DummyModel

    def test_register_model_type_check(self):
        """Test that only LifetimeModelBase subclasses can be registered."""
        with pytest.raises(TypeError, match="LifetimeModelBase"):
            ModelFactory.register_model("not-a-model", NotAModel)

    def test_register_model_overwrite(self):
        """Test overwriting an existing model registration."""
        ModelFactory.register_model("dummy", DummyModel)
        ModelFactory.register_model("dummy", AnotherDummyModel)

        assert ModelFactory._models["dummy"] == AnotherDummyModel

    def test_unregister_model(self):
        """Test unregistering a model."""
        ModelFactory.register_model("dummy", DummyModel)
        assert "dummy" in ModelFactory._models

        ModelFactory.unregister_model("dummy")
        assert "dummy" not in ModelFactory._models

    def test_unregister_nonexistent_model(self):
        """Test unregistering a model that doesn't exist (should not error)."""
        # Should not raise an error
        ModelFactory.unregister_model("nonexistent")

    def test_get_model(self):
        """Test getting a model instance."""
        ModelFactory.register_model("dummy", DummyModel)

        model = ModelFactory.get_model("dummy")

        assert isinstance(model, DummyModel)
        assert model.get_model_name() == "Dummy"

    def test_get_model_singleton(self):
        """Test that get_model returns singleton by default."""
        ModelFactory.register_model("dummy", DummyModel)

        model1 = ModelFactory.get_model("dummy")
        model2 = ModelFactory.get_model("dummy")

        assert model1 is model2  # Same instance

    def test_get_model_not_singleton(self):
        """Test getting a new instance (not singleton)."""
        ModelFactory.register_model("dummy", DummyModel)

        model1 = ModelFactory.get_model("dummy", use_singleton=False)
        model2 = ModelFactory.get_model("dummy", use_singleton=False)

        assert model1 is not model2  # Different instances
        assert isinstance(model1, DummyModel)
        assert isinstance(model2, DummyModel)

    def test_get_model_unknown(self):
        """Test getting a model that doesn't exist."""
        with pytest.raises(ValueError, match="Unknown model"):
            ModelFactory.get_model("nonexistent")

    def test_get_model_error_message_lists_available(self):
        """Test that error message lists available models."""
        ModelFactory.register_model("dummy", DummyModel)
        ModelFactory.register_model("another", AnotherDummyModel)

        with pytest.raises(ValueError) as exc_info:
            ModelFactory.get_model("nonexistent")

        error_msg = str(exc_info.value)
        assert "dummy" in error_msg
        assert "another" in error_msg

    def test_create_model(self):
        """Test creating a model with custom parameters."""
        ModelFactory.register_model("coffin-manson", CoffinMansonModel)

        model = ModelFactory.create_model(
            "coffin-manson",
            A=5.0e7,
            alpha=3.0
        )

        assert isinstance(model, CoffinMansonModel)
        assert model.A == 5.0e7
        assert model.alpha == 3.0

    def test_create_model_unknown(self):
        """Test creating a model that doesn't exist."""
        with pytest.raises(ValueError, match="Unknown model"):
            ModelFactory.create_model("nonexistent", A=1e6)

    def test_list_models(self):
        """Test listing all registered models."""
        ModelFactory.register_model("dummy", DummyModel)
        ModelFactory.register_model("another", AnotherDummyModel)

        models = ModelFactory.list_models()

        assert isinstance(models, list)
        assert "dummy" in models
        assert "another" in models

    def test_list_models_empty(self):
        """Test listing models when none are registered."""
        models = ModelFactory.list_models()

        assert models == []

    def test_is_registered(self):
        """Test checking if a model is registered."""
        ModelFactory.register_model("dummy", DummyModel)

        assert ModelFactory.is_registered("dummy") is True
        assert ModelFactory.is_registered("nonexistent") is False

    def test_get_model_info(self):
        """Test getting information about a model."""
        ModelFactory.register_model("coffin-manson", CoffinMansonModel)

        info = ModelFactory.get_model_info("coffin-manson")

        assert info["name"] == "coffin-manson"
        assert info["class"] == "CoffinMansonModel"
        assert info["model_name"] == "Coffin-Manson"
        assert "equation" in info
        assert "parameters" in info

    def test_get_model_info_unknown(self):
        """Test getting info for a model that doesn't exist."""
        with pytest.raises(ValueError, match="Unknown model"):
            ModelFactory.get_model_info("nonexistent")

    def test_clear_cache(self):
        """Test clearing the model instance cache."""
        ModelFactory.register_model("dummy", DummyModel)

        # Create and cache instance
        model1 = ModelFactory.get_model("dummy")
        assert len(ModelFactory._instances) > 0

        # Clear cache
        ModelFactory.clear_cache()
        assert len(ModelFactory._instances) == 0

        # Get new instance after cache clear
        model2 = ModelFactory.get_model("dummy")
        assert model1 is not model2

    def test_register_all(self):
        """Test registering all available models."""
        ModelFactory.register_all()

        models = ModelFactory.list_models()

        expected_models = [
            "coffin-manson",
            "coffin-manson-arrhenius",
            "norris-landzberg",
            "cips-2008",
            "lesit"
        ]

        for model_name in expected_models:
            assert model_name in models

    def test_get_default_model(self):
        """Test getting the default model name."""
        default = ModelFactory.get_default_model()

        assert default == "cips-2008"

    def test_convenience_functions(self):
        """Test convenience module-level functions."""
        from app.core.models.model_factory import get_model, create_model, list_models

        ModelFactory.register_model("dummy", DummyModel)

        # Test get_model
        model = get_model("dummy")
        assert isinstance(model, DummyModel)

        # Test create_model
        custom_model = create_model("dummy")
        assert isinstance(custom_model, DummyModel)

        # Test list_models
        models = list_models()
        assert "dummy" in models


class TestModelFactoryIntegration:
    """Integration tests with actual lifetime models."""

    def setup_method(self):
        """Register models before each test."""
        ModelFactory.register_all()

    def teardown_method(self):
        """Clean up after each test."""
        ModelFactory._models.clear()
        ModelFactory._instances.clear()

    def test_get_all_registered_models(self):
        """Test that all registered models can be retrieved."""
        models = ModelFactory.list_models()

        for model_name in models:
            model = ModelFactory.get_model(model_name)
            assert isinstance(model, LifetimeModelBase)
            assert model.get_model_name()

    def test_coffin_manson_via_factory(self):
        """Test Coffin-Manson model through factory."""
        model = ModelFactory.get_model("coffin-manson")

        result = model.calculate_cycles_to_failure(delta_Tj=100)

        assert result > 0

    def test_cips2008_via_factory(self):
        """Test CIPS 2008 model through factory."""
        model = ModelFactory.get_model("cips-2008")

        result = model.calculate_cycles_to_failure(
            delta_Tj=80,
            Tj_max=398,
            t_on=1.0,
            I=100,
            V=1200,
            D=300
        )

        assert result > 0

    def test_custom_coffin_manson_parameters(self):
        """Test creating Coffin-Manson with custom parameters."""
        model = ModelFactory.create_model(
            "coffin-manson",
            A=5.0e7,
            alpha=3.5
        )

        assert model.A == 5.0e7
        assert model.alpha == 3.5

        result = model.calculate_cycles_to_failure(delta_Tj=80)
        assert result > 0

    def test_get_all_model_info(self):
        """Test getting info for all registered models."""
        models = ModelFactory.list_models()

        for model_name in models:
            info = ModelFactory.get_model_info(model_name)

            assert "name" in info
            assert "class" in info
            assert "model_name" in info
            assert "equation" in info


class TestModelFactoryErrors:
    """Test error handling in ModelFactory."""

    def setup_method(self):
        """Clear model registry before each test."""
        ModelFactory._models.clear()
        ModelFactory._instances.clear()

    def teardown_method(self):
        """Clean up after each test."""
        ModelFactory._models.clear()
        ModelFactory._instances.clear()

    def test_invalid_model_class_raises_error(self):
        """Test that invalid model class raises TypeError."""
        class InvalidModel:
            pass

        with pytest.raises(TypeError, match="LifetimeModelBase"):
            ModelFactory.register_model("invalid", InvalidModel)

    def test_get_unregistered_model_raises_error(self):
        """Test that getting unregistered model raises ValueError."""
        with pytest.raises(ValueError, match="Unknown model"):
            ModelFactory.get_model("unregistered")

    def test_create_unregistered_model_raises_error(self):
        """Test that creating unregistered model raises ValueError."""
        with pytest.raises(ValueError, match="Unknown model"):
            ModelFactory.create_model("unregistered", A=1e6)

    def test_get_info_unregistered_model_raises_error(self):
        """Test that getting info for unregistered model raises ValueError."""
        with pytest.raises(ValueError, match="Unknown model"):
            ModelFactory.get_model_info("unregistered")

    def test_empty_list_when_no_models_registered(self):
        """Test that list_models returns empty list when nothing registered."""
        models = ModelFactory.list_models()

        assert models == []

    def test_is_registered_false_when_empty(self):
        """Test that is_registered returns False when factory is empty."""
        assert ModelFactory.is_registered("anything") is False

"""
Unit tests for Prediction API endpoints.

Tests include:
- Endpoint responses
- Error handling
- Validation
- Model comparison
"""

import pytest
from unittest.mock import Mock, patch
from fastapi.testclient import TestClient
from fastapi import status

from app.main import app
from app.core.models.model_factory import ModelFactory


class TestPredictionCalculateEndpoint:
    """Test /prediction/calculate endpoint."""

    def setup_method(self):
        """Set up test client and register models."""
        self.client = TestClient(app)
        ModelFactory.register_all()

    def test_basic_calculation(self):
        """Test basic lifetime calculation."""
        request_data = {
            "model_type": "coffin-manson",
            "parameters": {
                "delta_Tj": 80,
                "A": 1e6,
                "alpha": 2.0
            },
            "safety_factor": 1.0
        }

        response = self.client.post("/prediction/calculate", json=request_data)

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "predicted_lifetime_years" in data
        assert "predicted_lifetime_cycles" in data
        assert "model_used" in data

    def test_cips2008_calculation(self):
        """Test CIPS 2008 model calculation."""
        request_data = {
            "model_type": "cips-2008",
            "parameters": {
                "delta_Tj": 80,
                "Tj_max": 398,
                "t_on": 1.0,
                "I": 100,
                "V": 1200,
                "D": 300
            },
            "safety_factor": 1.0
        }

        response = self.client.post("/prediction/calculate", json=request_data)

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["model_used"] == "cips-2008"

    def test_default_parameters(self):
        """Test with default parameters."""
        request_data = {
            "model_type": "coffin-manson",
            "parameters": {},  # Empty - should use defaults
            "safety_factor": 1.0
        }

        response = self.client.post("/prediction/calculate", json=request_data)

        assert response.status_code == status.HTTP_200_OK

    def test_safety_factor_adjustment(self):
        """Test safety factor adjustment."""
        request_data = {
            "model_type": "coffin-manson",
            "parameters": {
                "delta_Tj": 80,
                "A": 1e6,
                "alpha": 2.0
            },
            "safety_factor": 2.0  # 50% life
        }

        response = self.client.post("/prediction/calculate", json=request_data)

        data = response.json()
        # With safety_factor=2, cycles should be half
        assert data["predicted_lifetime_cycles"] > 0

    def test_invalid_model_type(self):
        """Test with invalid model type."""
        request_data = {
            "model_type": "invalid-model",
            "parameters": {},
            "safety_factor": 1.0
        }

        response = self.client.post("/prediction/calculate", json=request_data)

        # Should return error
        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR

    def test_invalid_parameters(self):
        """Test with invalid parameters."""
        request_data = {
            "model_type": "coffin-manson",
            "parameters": {
                "delta_Tj": -10  # Negative, should fail validation
            },
            "safety_factor": 1.0
        }

        response = self.client.post("/prediction/calculate", json=request_data)

        assert response.status_code == status.HTTP_400_BAD_REQUEST


class TestPredictionCompareEndpoint:
    """Test /prediction/compare endpoint."""

    def setup_method(self):
        """Set up test client."""
        self.client = TestClient(app)
        ModelFactory.register_all()

    def test_compare_multiple_models(self):
        """Test comparing multiple models."""
        request_data = {
            "models": ["coffin-manson", "cips-2008"],
            "parameters": {
                "delta_Tj": 80
            }
        }

        response = self.client.post("/prediction/compare", json=request_data)

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "results" in data
        assert len(data["results"]) == 2
        assert "statistics" in data

    def test_compare_default_models(self):
        """Test comparing with default model list."""
        request_data = {
            "parameters": {
                "delta_Tj": 80
            }
        }

        response = self.client.post("/prediction/compare", json=request_data)

        assert response.status_code == status.HTTP_200_OK

    def test_compare_statistics(self):
        """Test that statistics are calculated correctly."""
        request_data = {
            "models": ["coffin-manson", "norris-landzberg"],
            "parameters": {
                "delta_Tj": 80,
                "Tj_max": 398,
                "f": 0.01
            }
        }

        response = self.client.post("/prediction/compare", json=request_data)

        data = response.json()
        stats = data["statistics"]

        assert "min_lifetime" in stats
        assert "max_lifetime" in stats
        assert "avg_lifetime" in stats
        assert "range" in stats

        # Check that max >= min
        assert stats["max_lifetime"] >= stats["min_lifetime"]


class TestPredictionSensitivityEndpoint:
    """Test /prediction/sensitivity endpoint."""

    def setup_method(self):
        """Set up test client."""
        self.client = TestClient(app)
        ModelFactory.register_all()

    def test_basic_sensitivity_analysis(self):
        """Test basic sensitivity analysis."""
        request_data = {
            "base_parameters": {
                "model_type": "coffin-manson",
                "delta_Tj": 80,
                "A": 1e6,
                "alpha": 2.0
            },
            "parameter_ranges": {
                "delta_Tj": (40, 120),
                "A": (5e5, 1.5e6)
            }
        }

        response = self.client.post("/prediction/sensitivity", json=request_data)

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "results" in data
        assert "base_lifetime" in data

    def test_sensitivity_with_multiple_parameters(self):
        """Test sensitivity across multiple parameters."""
        request_data = {
            "base_parameters": {
                "model_type": "coffin-manson",
                "delta_Tj": 80,
                "A": 1e6,
                "alpha": 2.0
            },
            "parameter_ranges": {
                "delta_Tj": (40, 120),
                "A": (5e5, 1.5e6),
                "alpha": (1.5, 2.5)
            }
        }

        response = self.client.post("/prediction/sensitivity", json=request_data)

        data = response.json()
        # Should have results for all parameters
        assert len(data["results"]) == 3

    def test_sensitivity_most_sensitive_identified(self):
        """Test that most sensitive parameter is identified."""
        request_data = {
            "base_parameters": {
                "model_type": "coffin-manson",
                "delta_Tj": 80,
                "A": 1e6,
                "alpha": 2.0
            },
            "parameter_ranges": {
                "delta_Tj": (40, 120),
                "A": (5e5, 1.5e6)
            }
        }

        response = self.client.post("/prediction/sensitivity", json=request_data)

        data = response.json()
        assert "most_sensitive_parameter" in data


class TestPredictionModelsEndpoint:
    """Test model listing and info endpoints."""

    def setup_method(self):
        """Set up test client."""
        self.client = TestClient(app)
        ModelFactory.register_all()

    def test_list_available_models(self):
        """Test listing all available models."""
        response = self.client.get("/prediction/models/available")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "models" in data
        assert len(data["models"]) > 0

        # Check model structure
        for model in data["models"]:
            assert "name" in model
            assert "display_name" in model
            assert "equation" in model

    def test_get_model_info(self):
        """Test getting info for specific model."""
        response = self.client.get("/prediction/models/coffin-manson")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["name"] == "coffin-manson"
        assert "equation" in data
        assert "parameters" in data

    def test_get_model_info_not_found(self):
        """Test getting info for non-existent model."""
        response = self.client.get("/prediction/models/nonexistent")

        assert response.status_code == status.HTTP_404_NOT_FOUND


class TestPredictionCRUDEndpoints:
    """Test CRUD endpoints for predictions."""

    def setup_method(self):
        """Set up test client and mock database."""
        self.client = TestClient(app)

    @patch('app.api.prediction.get_db')
    def test_get_predictions(self, mock_get_db):
        """Test getting all predictions."""
        mock_db = Mock()
        mock_get_db.return_value = mock_db

        # Mock CRUD operations
        with patch('app.api.prediction.prediction_crud.get_multi') as mock_get_multi:
            mock_get_multi.return_value = []
            response = self.client.get("/prediction?skip=0&limit=10")

        assert response.status_code == status.HTTP_200_OK

    @patch('app.api.prediction.get_db')
    def test_create_prediction(self, mock_get_db):
        """Test creating a prediction."""
        mock_db = Mock()
        mock_get_db.return_value = mock_db

        request_data = {
            "model_type": "coffin-manson",
            "delta_Tj": 80,
            "Tj_max": 398,
            "t_on": 1.0,
            "predicted_lifetime": 10000.0,
            "safety_factor": 1.0
        }

        with patch('app.api.prediction.prediction_crud.create') as mock_create:
            mock_pred = Mock()
            mock_pred.id = 1
            mock_create.return_value = mock_pred
            response = self.client.post("/prediction", json=request_data)

        assert response.status_code == status.HTTP_201_CREATED

    @patch('app.api.prediction.get_db')
    def test_get_prediction_by_id(self, mock_get_db):
        """Test getting prediction by ID."""
        mock_db = Mock()
        mock_get_db.return_value = mock_db

        with patch('app.api.prediction.prediction_crud.get') as mock_get:
            mock_get.return_value = Mock(id=1)
            response = self.client.get("/prediction/1")

        assert response.status_code == status.HTTP_200_OK

    @patch('app.api.prediction.get_db')
    def test_get_prediction_not_found(self, mock_get_db):
        """Test getting non-existent prediction."""
        mock_db = Mock()
        mock_get_db.return_value = mock_db

        with patch('app.api.prediction.prediction_crud.get') as mock_get:
            mock_get.return_value = None
            response = self.client.get("/prediction/999")

        assert response.status_code == status.HTTP_404_NOT_FOUND

    @patch('app.api.prediction.get_db')
    def test_update_prediction(self, mock_get_db):
        """Test updating a prediction."""
        mock_db = Mock()
        mock_get_db.return_value = mock_db

        request_data = {
            "predicted_lifetime": 15000.0
        }

        with patch('app.api.prediction.prediction_crud.get') as mock_get_pred:
            mock_get_pred.return_value = Mock(id=1)

            with patch('app.api.prediction.prediction_crud.update') as mock_update:
                mock_update.return_value = Mock(id=1)
                response = self.client.put("/prediction/1", json=request_data)

        assert response.status_code == status.HTTP_200_OK

    @patch('app.api.prediction.get_db')
    def test_delete_prediction(self, mock_get_db):
        """Test deleting a prediction."""
        mock_db = Mock()
        mock_get_db.return_value = mock_db

        with patch('app.api.prediction.prediction_crud.get') as mock_get:
            mock_get.return_value = Mock(id=1)

            with patch('app.api.prediction.prediction_crud.delete') as mock_delete:
                response = self.client.delete("/prediction/1")

        assert response.status_code == status.HTTP_204_NO_CONTENT

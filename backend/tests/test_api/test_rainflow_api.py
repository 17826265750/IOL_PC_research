"""
Unit tests for Rainflow API endpoints.

Tests include:
- Cycle counting endpoint
- Histogram generation
- Matrix computation
- Peak/valley extraction
- Error handling
"""

import pytest
import numpy as np
from fastapi.testclient import TestClient
from fastapi import status

from app.main import app
from app.core.rainflow import Cycle


class TestRainflowCountEndpoint:
    """Test /api/rainflow/count endpoint."""

    def setup_method(self):
        """Set up test client."""
        self.client = TestClient(app)

    def test_basic_counting(self):
        """Test basic cycle counting."""
        request_data = {
            "data_points": [
                {"value": 0},
                {"value": 50},
                {"value": 0},
                {"value": 30},
                {"value": 0}
            ],
            "bin_count": 32
        }

        response = self.client.post("/api/rainflow/count", json=request_data)

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "cycles" in data
        assert "total_cycles" in data
        assert "summary" in data

    def test_insufficient_data_points(self):
        """Test with insufficient data points."""
        request_data = {
            "data_points": [
                {"value": 0},
                {"value": 50}
            ],
            "bin_count": 32
        }

        response = self.client.post("/api/rainflow/count", json=request_data)

        assert response.status_code == status.HTTP_400_BAD_REQUEST

    def test_empty_data_points(self):
        """Test with empty data points."""
        request_data = {
            "data_points": [],
            "bin_count": 32
        }

        response = self.client.post("/api/rainflow/count", json=request_data)

        # Should handle gracefully or return empty result
        assert response.status_code in [status.HTTP_200_OK, status.HTTP_400_BAD_REQUEST]

    def test_summary_statistics(self):
        """Test that summary statistics are calculated."""
        request_data = {
            "data_points": [
                {"value": v} for v in [0, 100, 0, 80, 0, 60, 0]
            ],
            "bin_count": 32
        }

        response = self.client.post("/api/rainflow/count", json=request_data)

        data = response.json()
        summary = data["summary"]

        assert "total_cycles" in summary
        assert "unique_cycles" in summary
        assert "max_range" in summary
        assert "mean_range" in summary
        assert "std_range" in summary


class TestRainflowHistogramEndpoint:
    """Test /api/rainflow/histogram endpoint."""

    def setup_method(self):
        """Set up test client."""
        self.client = TestClient(app)

    def test_basic_histogram(self):
        """Test basic histogram generation."""
        request_data = {
            "cycles": [
                {"stress_range": 100, "mean_value": 50, "cycles": 1},
                {"stress_range": 80, "mean_value": 40, "cycles": 0.5},
            ],
            "bin_count": 10
        }

        response = self.client.post("/api/rainflow/histogram", json=request_data)

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "bins" in data
        assert "total_cycles" in data

    def test_bin_count_effect(self):
        """Test that different bin counts work."""
        request_data = {
            "cycles": [
                {"stress_range": 100, "mean_value": 50, "cycles": 1},
            ],
            "bin_count": 50
        }

        response = self.client.post("/api/rainflow/histogram", json=request_data)

        data = response.json()
        assert len(data["bins"]) == 50


class TestRainflowMatrixEndpoint:
    """Test /api/rainflow/matrix endpoint."""

    def setup_method(self):
        """Set up test client."""
        self.client = TestClient(app)

    def test_basic_matrix(self):
        """Test basic cycle matrix generation."""
        cycles = [
            {"stress_range": 100, "mean_value": 50, "cycles": 1},
            {"stress_range": 80, "mean_value": 40, "cycles": 0.5},
        ]

        response = self.client.post("/api/rainflow/matrix?bin_count=32", json=cycles)

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "matrix" in data
        assert "bin_count" in data
        assert "range_axis" in data
        assert "mean_axis" in data
        assert data["bin_count"] == 32

    def test_matrix_shape(self):
        """Test that matrix has correct shape."""
        cycles = [
            {"stress_range": 100, "mean_value": 50, "cycles": 1},
        ]

        response = self.client.post("/api/rainflow/matrix?bin_count=64", json=cycles)

        data = response.json()
        assert "shape" in data
        assert data["shape"] == [64, 64]


class TestRainflowEquivalentEndpoint:
    """Test /api/rainflow/equivalent endpoint."""

    def setup_method(self):
        """Set up test client."""
        self.client = TestClient(app)

    def test_basic_equivalent_amplitude(self):
        """Test equivalent constant amplitude calculation."""
        cycles = [
            {"stress_range": 100, "mean_value": 50, "cycles": 1},
            {"stress_range": 80, "mean_value": 40, "cycles": 1},
        ]

        response = self.client.post("/api/rainflow/equivalent?exponent=3.0", json=cycles)

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "equivalent_range" in data
        assert "exponent" in data

    def test_exponent_effect(self):
        """Test that exponent affects result."""
        cycles = [
            {"stress_range": 100, "mean_value": 50, "cycles": 1},
            {"stress_range": 80, "mean_value": 40, "cycles": 1},
        ]

        response1 = self.client.post("/api/rainflow/equivalent?exponent=2.0", json=cycles)
        response2 = self.client.post("/api/rainflow/equivalent?exponent=4.0", json=cycles)

        data1 = response1.json()
        data2 = response2.json()

        # Different exponents should give different results
        # (assuming same cycles)
        assert data1["equivalent_range"] != data2["equivalent_range"]


class TestRainflowCumulativeEndpoint:
    """Test /api/rainflow/cumulative endpoint."""

    def setup_method(self):
        """Set up test client."""
        self.client = TestClient(app)

    def test_basic_cumulative(self):
        """Test cumulative cycle distribution."""
        cycles = [
            {"stress_range": 40, "mean_value": 20, "cycles": 1},
            {"stress_range": 60, "mean_value": 30, "cycles": 1},
            {"stress_range": 80, "mean_value": 40, "cycles": 1},
        ]

        response = self.client.post("/api/rainflow/cumulative", json=cycles)

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "ranges" in data
        assert "cumulative_counts" in data
        assert "total_cumulative" in data

    def test_cumulative_monotonic(self):
        """Test that cumulative counts are monotonic."""
        cycles = [
            {"stress_range": i * 20, "mean_value": i * 10, "cycles": 1}
            for i in range(1, 6)
        ]

        response = self.client.post("/api/rainflow/cumulative", json=cycles)

        data = response.json()
        cumulative = data["cumulative_counts"]

        # Cumulative should be non-decreasing
        for i in range(len(cumulative) - 1):
            assert cumulative[i] <= cumulative[i + 1]


class TestRainflowPeaksEndpoint:
    """Test /api/rainflow/peaks endpoint."""

    def setup_method(self):
        """Set up test client."""
        self.client = TestClient(app)

    def test_basic_peak_extraction(self):
        """Test peak and valley extraction."""
        data_points = [
            {"value": v} for v in [0, 50, 100, 50, 0, -50, -100, -50, 0]
        ]

        response = self.client.post("/api/rainflow/peaks", json=data_points)

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "peaks_valleys" in data
        assert "count" in data
        assert "reduction" in data

    def test_single_point(self):
        """Test with single data point."""
        data_points = [{"value": 42}]

        response = self.client.post("/api/rainflow/peaks", json=data_points)

        data = response.json()
        assert data["count"] == 1

    def test_two_points(self):
        """Test with two data points."""
        data_points = [{"value": 0}, {"value": 100}]

        response = self.client.post("/api/rainflow/peaks", json=data_points)

        data = response.json()
        assert data["count"] == 2


class TestRainflowAnalyzeEndpoint:
    """Test /api/rainflow/analyze endpoint."""

    def setup_method(self):
        """Set up test client."""
        self.client = TestClient(app)

    def test_comprehensive_analysis(self):
        """Test comprehensive time series analysis."""
        data_points = [
            {"value": v} for v in [0, 100, 0, 80, 0, 60, 0, 120, 0]
        ]

        response = self.client.post("/api/rainflow/analyze", json=data_points)

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "statistics" in data
        assert "peaks_valleys" in data
        assert "cycles" in data

    def test_statistics_calculation(self):
        """Test that statistics are correctly calculated."""
        data_points = [
            {"value": v} for v in [0, 50, 100, 50, 0]
        ]

        response = self.client.post("/api/rainflow/analyze", json=data_points)

        data = response.json()
        stats = data["statistics"]

        assert "count" in stats
        assert "mean" in stats
        assert "std" in stats
        assert "min" in stats
        assert "max" in stats
        assert "range" in stats

    def test_insufficient_data_error(self):
        """Test that insufficient data returns error."""
        data_points = [
            {"value": 0},
            {"value": 50}
        ]

        response = self.client.post("/api/rainflow/analyze", json=data_points)

        assert response.status_code == status.HTTP_400_BAD_REQUEST


class TestRainflowErrorHandling:
    """Test error handling across all endpoints."""

    def setup_method(self):
        """Set up test client."""
        self.client = TestClient(app)

    def test_invalid_json(self):
        """Test with invalid JSON."""
        response = self.client.post(
            "/api/rainflow/count",
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_missing_required_fields(self):
        """Test with missing required fields."""
        request_data = {
            "data_points": []  # Missing bin_count (has default)
        }

        response = self.client.post("/api/rainflow/count", json=request_data)

        # Should still work (bin_count has default)
        # or return error for empty data
        assert response.status_code in [status.HTTP_200_OK, status.HTTP_400_BAD_REQUEST]

    def test_negative_bin_count(self):
        """Test with negative bin count."""
        cycles = [
            {"stress_range": 100, "mean_value": 50, "cycles": 1},
        ]

        response = self.client.post("/api/rainflow/matrix?bin_count=-10", json=cycles)

        # Should handle error or return empty matrix
        assert response.status_code in [status.HTTP_200_OK, status.HTTP_422_UNPROCESSABLE_ENTITY]


class TestRainflowRealWorldScenarios:
    """Test with realistic scenarios."""

    def setup_method(self):
        """Set up test client."""
        self.client = TestClient(app)

    def test_thermal_profile_analysis(self):
        """Test analysis of realistic thermal profile."""
        # Simulate thermal cycling: 40°C -> 125°C -> 25°C cycles
        thermal_data = []
        for _ in range(3):
            thermal_data.extend(np.linspace(40, 125, 10))
            thermal_data.extend(np.linspace(125, 25, 15))
            thermal_data.extend(np.linspace(25, 40, 5))

        data_points = [{"value": float(v)} for v in thermal_data]

        response = self.client.post("/api/rainflow/analyze", json=data_points)

        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        # Should have detected cycles
        assert len(data["cycles"]) > 0
        assert data["statistics"]["rainflow_cycles"] > 0

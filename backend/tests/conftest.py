"""
Pytest configuration and shared fixtures.

This module provides common fixtures for all test modules.
"""

import pytest
import numpy as np
from typing import Dict, Any, List
from unittest.mock import Mock, AsyncMock

# Sample cycle data fixtures
@pytest.fixture
def sample_cycle_data() -> List[Dict[str, float]]:
    """Sample cycle data for testing damage accumulation."""
    return [
        {'range': 100.0, 'mean': 50.0, 'count': 1000},
        {'range': 80.0, 'mean': 40.0, 'count': 2000},
        {'range': 60.0, 'mean': 30.0, 'count': 5000},
        {'range': 40.0, 'mean': 20.0, 'count': 10000},
    ]


@pytest.fixture
def sample_half_cycle_data() -> List[Dict[str, float]]:
    """Sample cycle data including half cycles."""
    return [
        {'range': 100.0, 'mean': 50.0, 'count': 1.0},
        {'range': 80.0, 'mean': 40.0, 'count': 0.5},
        {'range': 60.0, 'mean': 30.0, 'count': 1.0},
        {'range': 40.0, 'mean': 20.0, 'count': 0.5},
    ]


# Sample time series data fixtures
@pytest.fixture
def sample_time_series() -> List[float]:
    """Sample temperature time series data."""
    return [0, 10, -5, 8, -3, 7, -2, 5, 0, 12, -8, 10, -6, 8]


@pytest.fixture
def sample_simple_sine_wave() -> List[float]:
    """Simple sine wave for rainflow testing."""
    return [0, 50, 100, 50, 0, -50, -100, -50, 0]


@pytest.fixture
def sample_irregular_loading() -> List[float]:
    """Irregular loading sequence typical of real applications."""
    return [0, 80, 20, 90, 10, 70, 30, 85, 15, 60, 40, 50]


# Model parameter fixtures
@pytest.fixture
def cips2008_params() -> Dict[str, float]:
    """CIPS 2008 model parameters."""
    return {
        'delta_Tj': 80.0,
        'Tj_max': 398.0,
        't_on': 1.0,
        'I': 100.0,
        'V': 1200.0,
        'D': 300.0,
    }


@pytest.fixture
def coffin_manson_params() -> Dict[str, float]:
    """Coffin-Manson model parameters."""
    return {
        'delta_Tj': 80.0,
        'A': 1.0e6,
        'alpha': 2.0,
    }


@pytest.fixture
def norris_landzberg_params() -> Dict[str, float]:
    """Norris-Landzberg model parameters."""
    return {
        'delta_Tj': 80.0,
        'f': 0.01,
        'Tj_max': 398.0,
        'A': 1.0e6,
        'alpha': 2.0,
        'beta': 0.333,
        'Ea': 0.5,
    }


@pytest.fixture
def lesit_params() -> Dict[str, float]:
    """LESIT model parameters."""
    return {
        'delta_Tj': 80.0,
        'Tj_min': 318.0,
        'A': 1.0e6,
        'alpha': -3.5,
        'Q': 0.8,
    }


# Weibull analysis fixtures
@pytest.fixture
def sample_failure_data() -> List[float]:
    """Sample failure times for Weibull analysis."""
    return [100, 200, 300, 400, 500, 600, 750, 900, 1100, 1300]


@pytest.fixture
def sample_censored_data() -> List[float]:
    """Sample censored (suspension) data."""
    return [800, 1000, 1200]  # Units that didn't fail


# Fitting test fixtures
@pytest.fixture
def sample_fitting_data():
    """Generate synthetic data for fitting tests."""
    np.random.seed(42)
    x_data = np.linspace(50, 150, 20)
    # CIPS 2008-like model with some noise
    K = 1e10
    beta1 = -4.5
    beta2 = 1500

    y_data = K * (x_data ** beta1) * np.exp(beta2 / 400)
    y_data += y_data * 0.05 * np.random.randn(len(y_data))  # 5% noise

    return x_data, y_data


@pytest.fixture
def sample_cips2008_experiment_data() -> List[Dict[str, float]]:
    """Sample CIPS 2008 experiment data for fitting."""
    return [
        {'dTj': 60, 'Tj_max': 125, 't_on': 1, 'I': 100, 'V': 1200, 'D': 300, 'Nf': 2.0e7},
        {'dTj': 80, 'Tj_max': 125, 't_on': 1, 'I': 100, 'V': 1200, 'D': 300, 'Nf': 5.0e6},
        {'dTj': 100, 'Tj_max': 125, 't_on': 1, 'I': 100, 'V': 1200, 'D': 300, 'Nf': 1.5e6},
        {'dTj': 80, 'Tj_max': 150, 't_on': 1, 'I': 100, 'V': 1200, 'D': 300, 'Nf': 2.5e6},
        {'dTj': 80, 'Tj_max': 100, 't_on': 1, 'I': 100, 'V': 1200, 'D': 300, 'Nf': 1.0e7},
        {'dTj': 80, 'Tj_max': 125, 't_on': 10, 'I': 100, 'V': 1200, 'D': 300, 'Nf': 3.0e6},
        {'dTj': 80, 'Tj_max': 125, 't_on': 0.1, 'I': 100, 'V': 1200, 'D': 300, 'Nf': 7.0e6},
    ]


# Sensitivity analysis fixtures
@pytest.fixture
def sensitivity_base_params() -> Dict[str, float]:
    """Base parameters for sensitivity analysis."""
    return {
        'delta_Tj': 80.0,
        'Tj_max': 398.0,
        't_on': 1.0,
        'I': 100.0,
        'V': 1200.0,
        'D': 300.0,
    }


@pytest.fixture
def sensitivity_param_ranges() -> Dict[str, tuple]:
    """Parameter ranges for sensitivity analysis."""
    return {
        'delta_Tj': (40.0, 120.0),
        'Tj_max': (350.0, 450.0),
        't_on': (0.1, 10.0),
        'I': (50.0, 150.0),
    }


# API test fixtures
@pytest.fixture
def mock_db_session():
    """Mock database session for API tests."""
    session = Mock()
    session.add = Mock()
    session.commit = Mock()
    session.refresh = Mock()
    session.rollback = Mock()
    return session


@pytest.fixture
def mock_prediction():
    """Mock prediction object for API tests."""
    pred = Mock()
    pred.id = 1
    pred.model_type = "cips-2008"
    pred.delta_Tj = 80.0
    pred.Tj_max = 398.0
    pred.t_on = 1.0
    pred.predicted_lifetime = 10000.0
    pred.safety_factor = 1.0
    return pred


# Helper function fixture
@pytest.fixture
def simple_lifetime_model():
    """Simple lifetime model function for testing."""
    def model(range_val: float, mean_val: float, params: Dict[str, Any]) -> float:
        # Simple power law model
        A = params.get('A', 1e6)
        alpha = params.get('alpha', 2.0)
        return A * (range_val ** (-alpha))
    return model


# ASTN E1049 standard test fixture
@pytest.fixture
def astm_standard_sequence() -> List[float]:
    """ASTM E1049 standard test sequence.

    From ASTM E1049-85, Section 5.4.4, Example 1.
    """
    return [-2, 1, -3, 5, -1, 3, -4, 4, -2]


@pytest.fixture
def astm_expected_cycles() -> List[Dict[str, float]]:
    """Expected cycle counts for ASTM standard sequence.

    Based on ASTM E1049-85 three-point method.
    """
    return [
        {'range': 6, 'mean': 0.5, 'count': 1.0},
        {'range': 8, 'mean': 0.0, 'count': 1.0},
        {'range': 3, 'mean': 1.0, 'count': 0.5},
    ]

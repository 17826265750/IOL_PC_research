"""
Core algorithms package for CIPS 2008 lifetime prediction.

This package provides:
- Weibull analysis for reliability metrics
- Non-linear model parameter fitting
- Sensitivity and uncertainty analysis
- Rainflow cycle counting
- Damage accumulation models
- Remaining life estimation
- Safety margin calculations
"""

# Core analysis modules
from . import weibull
from . import fitting
from . import sensitivity

# Existing modules
from . import rainflow
from . import damage_accumulation
from . import remaining_life
from . import safety_margin

__all__ = [
    'weibull',
    'fitting',
    'sensitivity',
    'rainflow',
    'damage_accumulation',
    'remaining_life',
    'safety_margin',
]

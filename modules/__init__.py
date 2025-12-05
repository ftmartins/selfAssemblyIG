"""
Core modules for patchy particle optimization.

This package contains three main modules:
- utility_functions: Simulation infrastructure and helper functions
- evaluation_functions: Loss calculation and cluster detection
- optimizer: Optimization algorithms and random search
"""

from . import utility_functions
from . import evaluation_functions
from . import optimizer

__all__ = ['utility_functions', 'evaluation_functions', 'optimizer']

"""
Physics-Informed Bayesian Optimization for ReaxFF
Main package initialization
"""

__version__ = "2.0.0"
__author__ = "PIBO Development Team"

from importlib import import_module

_EXPORTS = {
    'BlockedPIBO': '.core.blocked_pibo',
    'ImprovedGaussianProcess': '.core.improved_gp',
    'DeepKernelGP': '.core.improved_gp',
    'PhysicsInformedBO': '.core.staged_pibo',
    'PIBOVisualizer': '.visualization_pibo',
    'visualize_results': '.visualization_pibo',
}


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(_EXPORTS[name], __name__), name)
    globals()[name] = value
    return value

__all__ = [
    'BlockedPIBO',
    'ImprovedGaussianProcess', 
    'DeepKernelGP',
    'PhysicsInformedBO',
    'PIBOVisualizer',
    'visualize_results'
]

"""
Core modules for Physics-Informed Bayesian Optimization
"""

from importlib import import_module

_EXPORTS = {
    'BlockedPIBO': '.blocked_pibo',
    'ImprovedGaussianProcess': '.improved_gp',
    'DeepKernelGP': '.improved_gp',
    'PhysicsInformedBO': '.staged_pibo',
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
    'PhysicsInformedBO'
]

# Module metadata
__version__ = "2.0.0"
__description__ = "Core PIBO optimization modules with blocked search and staged strategies"

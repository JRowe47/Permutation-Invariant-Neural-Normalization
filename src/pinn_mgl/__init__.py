"""A bounded static analysis profile of the MGL v0.2 design."""

__version__ = "0.1.0"

from .core import Budget, FFN, Tensor, analyze, array_tensor, validate_package

__all__ = ["Budget", "FFN", "Tensor", "analyze", "array_tensor", "validate_package"]

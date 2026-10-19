"""A bounded static analysis profile of the MGL v0.2 design."""

__version__ = "0.2.0"

from .core import Budget, FFN, Tensor, analyze, array_tensor, validate_package
from .contextual import ContextBank, ContextBudget, analyze_contexts, compare_contexts, validate_contextual

__all__ = ["Budget", "FFN", "Tensor", "analyze", "array_tensor", "validate_package",
           "ContextBank", "ContextBudget", "analyze_contexts", "compare_contexts", "validate_contextual"]

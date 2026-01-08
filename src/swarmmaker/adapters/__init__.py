"""Domain adapter plugin system.

This module provides pluggable domain-specific verification, evidence extraction,
and composition capabilities.
"""
from typing import Dict, Type

from .base import BaseDomainAdapter
from .default import DefaultAdapter
from .math import MathAdapter

# Registry of available adapters
ADAPTER_REGISTRY: Dict[str, Type[BaseDomainAdapter]] = {
    "default": DefaultAdapter,
    "math": MathAdapter,
}


def get_adapter(name: str) -> BaseDomainAdapter:
    """Get an adapter instance by name.

    Args:
        name: Adapter name (e.g., "default", "math").

    Returns:
        Instantiated adapter. Falls back to DefaultAdapter if not found.
    """
    adapter_class = ADAPTER_REGISTRY.get(name, DefaultAdapter)
    return adapter_class()


def register_adapter(name: str, adapter_class: Type[BaseDomainAdapter]) -> None:
    """Register a custom adapter.

    Args:
        name: Name to register under.
        adapter_class: Adapter class to register.
    """
    ADAPTER_REGISTRY[name] = adapter_class


__all__ = [
    "BaseDomainAdapter",
    "DefaultAdapter",
    "MathAdapter",
    "get_adapter",
    "register_adapter",
    "ADAPTER_REGISTRY",
]

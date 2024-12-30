"""Model registry for mapping configuration names to model constructors."""

from typing import Any, Callable, Dict, Optional


class ModelRegistry:
    """Central registry for model constructors and their default hyperparameters."""

    def __init__(self):
        self._models: Dict[str, Dict[str, Any]] = {}

    def register(
        self,
        name: str,
        constructor: Callable[..., Any],
        model_type: str,
        default_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Register a model constructor with metadata."""
        self._models[name] = {
            "constructor": constructor,
            "type": model_type,
            "default_params": default_params or {},
        }

    def get(self, name: str) -> Dict[str, Any]:
        """Retrieve model metadata by name."""
        if name not in self._models:
            raise KeyError(f"Model '{name}' is not registered. Available: {list(self._models.keys())}")
        return self._models[name]

    def build(self, name: str, overrides: Optional[Dict[str, Any]] = None) -> Any:
        """Instantiate a model with optional parameter overrides."""
        meta = self.get(name)
        params = dict(meta["default_params"])
        if overrides:
            params.update(overrides)
        return meta["constructor"](**params)

    def list_models(self, model_type: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
        """List registered models, optionally filtered by type."""
        if model_type is None:
            return dict(self._models)
        return {
            name: meta
            for name, meta in self._models.items()
            if meta["type"] == model_type
        }

    def is_registered(self, name: str) -> bool:
        """Check if a model name is registered."""
        return name in self._models


# Global registry instance
REGISTRY = ModelRegistry()


def register_model(
    name: str,
    model_type: str,
    default_params: Optional[Dict[str, Any]] = None,
):
    """Decorator to register a model class in the global registry."""
    def decorator(constructor: Callable[..., Any]) -> Callable[..., Any]:
        REGISTRY.register(name, constructor, model_type, default_params)
        return constructor
    return decorator

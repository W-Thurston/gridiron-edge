"""Registry mapping model keys to model implementations.

Models are registered via the ``@ModelRegistry.register`` decorator and
retrieved by registry key at evaluation, prediction, and CLI time.

The registry is intentionally model-domain agnostic. It stores both:

- game models, e.g. ``win_prob_random_forest``
- prop model families, e.g. ``qb_pass_yards``

Backward-compatible ``PredictorRegistry`` alias is retained during the
migration.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    from gridiron_edge.models.base import Model, ModelSpec


class ModelRegistry:
    """Registry mapping model keys to model classes."""

    _models: ClassVar[dict[str, type[Model]]] = {}

    @classmethod
    def _read_spec(cls, model_cls: type[Model]) -> ModelSpec:
        """Read a model spec from class or instance.

        Game models expose ``spec`` as a class attribute. Prop trainers expose
        ``spec`` as an instance property. This helper supports both patterns.
        """
        spec: Any | None = getattr(model_cls, "spec", None)

        # Class attribute path: game models.
        if spec is not None and hasattr(spec, "name"):
            return spec

        # Instance property path: prop trainers.
        instance: Model = model_cls()
        return instance.spec

    @classmethod
    def register(cls, model_cls: type[Model]) -> type[Model]:
        """Register a model class under its ``spec.name``."""
        spec: ModelSpec = cls._read_spec(model_cls)
        name: str = spec.name

        if name in cls._models:
            raise ValueError(
                f"Model '{name}' is already registered. Each model registry key must be unique."
            )

        cls._models[name] = model_cls
        return model_cls

    @classmethod
    def get(cls, name: str) -> type[Model]:
        """Retrieve a registered model class by key."""
        if name not in cls._models:
            available: list[str] = sorted(cls._models.keys())
            raise KeyError(f"No model registered as '{name}'. Available: {available}")
        return cls._models[name]

    @classmethod
    def all(cls) -> dict[str, type[Model]]:
        """Return a copy of all registered models."""
        return dict(cls._models)

    @classmethod
    def names(cls) -> list[str]:
        """Return sorted model registry keys."""
        return sorted(cls._models.keys())

    @classmethod
    def known_model_names(cls) -> tuple[str, ...]:
        """Return known model_name prefixes for composite-key parsing.

        Game model classes expose ``model_name`` as a class attribute.
        Prop model families use their spec name as the model_name.
        """
        names: set[str] = set()

        for model_cls in cls._models.values():
            class_model_name: Any | None = getattr(model_cls, "model_name", None)
            if isinstance(class_model_name, str) and class_model_name:
                names.add(class_model_name)
                continue

            spec: ModelSpec = cls._read_spec(model_cls)
            names.add(spec.name)

        return tuple(sorted(names, key=len, reverse=True))

    @classmethod
    def is_trainable(cls, name: str) -> bool:
        """Return whether a registered model implements Trainable."""
        from gridiron_edge.models.base import Trainable

        model_cls: type[Model] = cls.get(name)
        return isinstance(model_cls(), Trainable)

    @classmethod
    def trainable_names(cls) -> list[str]:
        """Return sorted registered model keys that implement Trainable."""
        from gridiron_edge.models.base import Trainable

        return sorted(name for name, mcls in cls._models.items() if isinstance(mcls(), Trainable))


# Backward-compatible alias. Existing imports of PredictorRegistry keep
# working during the migration.
PredictorRegistry: type[ModelRegistry] = ModelRegistry

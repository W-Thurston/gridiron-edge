"""Registry mapping model keys to model implementations.

Models are registered via the ``@ModelRegistry.register`` decorator and
retrieved by registry key at evaluation, prediction, and CLI time.

The registry is intentionally model-domain agnostic. It stores both:

- game models, e.g. ``win_prob_random_forest``
- prop model families, e.g. ``qb_pass_yards``

Trainability is exposed through ``ModelSpec.trainable`` as the canonical
declarative signal, and the structural ``Trainable`` protocol provides
type-level enforcement of the train/is_trained surface. Both signals are
checked at registration time so they cannot drift apart at runtime.

checked at registration time so they cannot drift apart at runtime.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    from gridiron_edge.models.base import Model, ModelSpec


class ModelRegistry:
    """Registry mapping model keys to model classes."""

    _models: ClassVar[dict[str, type[Model]]] = {}

    @classmethod
    def _read_spec(cls: type[ModelRegistry], model_cls: type[Model]) -> ModelSpec:
        """Read a model spec from class or instance.

        Game models expose ``spec`` as a class attribute. Prop trainers
        expose ``spec`` as an instance property. This helper supports
        both patterns.
        """
        spec: Any | None = getattr(model_cls, "spec", None)

        if spec is not None and hasattr(spec, "name"):
            return spec

        instance: Model = model_cls()
        return instance.spec

    @classmethod
    def register(cls: type[ModelRegistry], model_cls: type[Model]) -> type[Model]:
        """Register a model class under its ``spec.name``.

        Enforces a consistency invariant between the declarative
        ``spec.trainable`` flag and the structural ``Trainable``
        protocol. If they disagree, registration raises ``TypeError``.

        This guarantees that downstream consumers of either signal see
        the same answer, eliminating drift between the registry's fast
        ``is_trainable`` lookup and the runtime protocol check.
        """
        from gridiron_edge.models.base import Trainable

        spec: ModelSpec = cls._read_spec(model_cls)
        name: str = spec.name

        if name in cls._models:
            raise ValueError(
                f"Model '{name}' is already registered. Each model registry key must be unique."
            )

        instance = model_cls()
        is_structurally_trainable = isinstance(instance, Trainable)

        if spec.trainable and not is_structurally_trainable:
            raise TypeError(
                f"Model '{name}' declares spec.trainable=True but does not "
                f"satisfy the Trainable protocol. Missing train() or "
                f"is_trained()."
            )

        if not spec.trainable and is_structurally_trainable:
            raise TypeError(
                f"Model '{name}' implements the Trainable protocol but "
                f"declares spec.trainable=False. Update its spec or remove "
                f"the train() / is_trained() methods."
            )

        cls._models[name] = model_cls
        return model_cls

    @classmethod
    def get(cls: type[ModelRegistry], name: str) -> type[Model]:
        """Retrieve a registered model class by key."""
        if name not in cls._models:
            available: list[str] = sorted(cls._models.keys())
            raise KeyError(f"No model registered as '{name}'. Available: {available}")
        return cls._models[name]

    @classmethod
    def all(cls: type[ModelRegistry]) -> dict[str, type[Model]]:
        """Return a copy of all registered models."""
        return dict(cls._models)

    @classmethod
    def names(cls: type[ModelRegistry]) -> list[str]:
        """Return sorted model registry keys."""
        return sorted(cls._models.keys())

    @classmethod
    def known_model_names(cls: type[ModelRegistry]) -> tuple[str, ...]:
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
    def is_trainable(cls: type[ModelRegistry], name: str) -> bool:
        """Return whether a registered model is trainable.

        Reads ``model_cls().spec.trainable``. The spec is the canonical
        declarative source of truth; structural consistency between the
        spec and the ``Trainable`` protocol is enforced at registration
        time, so this fast path is always safe.
        """
        spec: ModelSpec = cls._read_spec(cls.get(name))
        return spec.trainable

    @classmethod
    def trainable_names(cls: type[ModelRegistry]) -> list[str]:
        """Return sorted registered model keys whose spec.trainable is True."""
        return sorted(name for name, mcls in cls._models.items() if cls._read_spec(mcls).trainable)

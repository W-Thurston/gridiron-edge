# src/gridiron_edge/models/registry.py

"""Registry mapping model version strings to Predictor implementations.

Predictors are registered via the ``@PredictorRegistry.register`` decorator
and retrieved by name at evaluation and prediction time.

Usage::

    # Registering a predictor
    @PredictorRegistry.register
    class EloV1Predictor:
        spec = PredictorSpec(name="elo_v1", description="Elo ratings v1 (K=20, div=480)")
        ...


    # Retrieving and instantiating a predictor by name
    predictor = PredictorRegistry.get("elo_v1")()

    # Listing all registered predictors
    for name, cls in PredictorRegistry.all().items():
        print(name, cls().spec.description)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from gridiron_edge.models.base import Predictor


class PredictorRegistry:
    """Registry mapping model version strings to Predictor classes.

    Predictors are registered via ``@PredictorRegistry.register`` and
    retrieved by name. The registry is populated at import time when
    predictor modules are imported — the same pattern used by
    ``FeatureRegistry``.
    """

    _predictors: ClassVar[dict[str, type[Predictor]]] = {}

    @classmethod
    def register(cls, predictor_cls: type[Predictor]) -> type[Predictor]:
        """Register a predictor class under its ``spec.name``.

        Used as a class decorator::

            @PredictorRegistry.register
            class EloV1Predictor:
                spec = PredictorSpec(name="elo_v1", ...)

        Args:
            predictor_cls: The predictor class to register. Must have a
                ``spec`` attribute with a ``name`` field.

        Returns:
            The predictor class unchanged (decorator pattern).

        Raises:
            ValueError: If a predictor with the same name is already
                registered.
        """
        name: str = predictor_cls.spec.name  # type: ignore[union-attr]
        if name in cls._predictors:
            raise ValueError(
                f"Predictor '{name}' is already registered. "
                f"Each model version must have a unique name."
            )
        cls._predictors[name] = predictor_cls
        return predictor_cls

    @classmethod
    def get(cls, name: str) -> type[Predictor]:
        """Retrieve a registered predictor class by name.

        Args:
            name: The model version string (e.g. ``"elo_v1"``).

        Returns:
            The predictor class. Call it to instantiate: ``registry.get("elo_v1")()``.

        Raises:
            KeyError: If no predictor with this name has been registered.
        """
        if name not in cls._predictors:
            available = sorted(cls._predictors.keys())
            raise KeyError(f"No predictor registered as '{name}'. Available: {available}")
        return cls._predictors[name]

    @classmethod
    def all(cls) -> dict[str, type[Predictor]]:
        """Return a copy of all registered predictors.

        Returns:
            Dict mapping model version strings to predictor classes.
        """
        return dict(cls._predictors)

    @classmethod
    def names(cls) -> list[str]:
        """Return a sorted list of all registered model version strings.

        Returns:
            Sorted list of model version names.
        """
        return sorted(cls._predictors.keys())

    @classmethod
    def is_trainable(cls, name: str) -> bool:
        """Return whether a registered predictor implements ``Trainable``.

        Args:
            name: The model version string.

        Returns:
            ``True`` if the predictor class implements the ``Trainable``
            protocol (has ``train`` and ``is_trained`` methods).
        """
        from gridiron_edge.models.base import Trainable

        predictor_cls = cls.get(name)
        return isinstance(predictor_cls(), Trainable)

    @classmethod
    def trainable_names(cls) -> list[str]:
        """Return sorted list of model version strings that are trainable.

        Returns:
            Sorted list of model version names implementing ``Trainable``.
        """
        from gridiron_edge.models.base import Trainable

        return sorted(
            name for name, pcls in cls._predictors.items() if isinstance(pcls(), Trainable)
        )

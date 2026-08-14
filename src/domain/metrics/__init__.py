"""Measurements, each registered under a name and its own version.

Importing this package registers every metric module against REGISTRY, so
anything that wants the catalogue only needs to import from here.
"""

# Registration happens on import. Kept at the bottom so the names above are
# available to the modules being imported.
from src.domain.metrics import (  # noqa: E402,F401  (import registers them)
    aim,
    impact,
    replay,
    roles,
    utility,
)
from src.domain.metrics.context import MetricContext
from src.domain.metrics.registry import (
    ENRICHED_ROUNDS,
    REGISTRY,
    MetricRegistry,
    MetricSpec,
    metric,
)

__all__ = [
    "ENRICHED_ROUNDS",
    "REGISTRY",
    "MetricContext",
    "MetricRegistry",
    "MetricSpec",
    "metric",
]

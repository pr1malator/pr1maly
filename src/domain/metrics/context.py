"""What every metric is handed.

The expensive intermediates are built once and shared. `enriched_rounds` in
particular is the central structure the whole analysis hangs off — it is what
gets persisted per round — and rebuilding it per metric would be the single
most wasteful thing this refactor could introduce.

A context can be built two ways, and which one is available is the whole point
of the `requires` declaration on a metric:

  from_parsed()   everything, straight after the demo was parsed
  from_storage()  enriched rounds read back out of the database, with no demo

The second is what makes it possible to recompute a measurement for a match
whose .dem the retention feature deleted months ago.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class MetricContext:
    """Shared inputs for one match's worth of metrics."""

    steam_id: str
    map_name: str
    total_rounds: int
    enriched_rounds: list[dict[str, Any]]

    # Raw parser frames. Empty when the context was built from storage, which
    # is why a metric that needs one must say so in `requires`.
    parsed: dict[str, Any] = field(default_factory=dict)

    # Scratch space for anything derived that more than one metric wants.
    _derived: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def has_demo(self) -> bool:
        """Whether the raw parser frames are available."""
        return bool(self.parsed)

    def frame(self, name: str) -> Any:
        """A parser frame by name.

        Raises rather than returning empty: a metric asking for a frame it did
        not declare, in a context that does not have it, is a bug in the metric
        rather than a match with no data.
        """
        if not self.has_demo:
            raise LookupError(
                f"{name!r} was requested but this context was built without a demo; "
                f"the metric should declare requires={{'parsed:{name}'}} so it is "
                f"skipped instead"
            )
        return self.parsed.get(name)

    def derive(self, key: str, build):
        """Memoise something more than one metric needs."""
        if key not in self._derived:
            self._derived[key] = build()
        return self._derived[key]

    @classmethod
    def from_parsed(
        cls,
        parsed: dict[str, Any],
        steam_id: str,
        *,
        enriched_rounds: list[dict[str, Any]],
        total_rounds: int,
    ) -> MetricContext:
        header = parsed.get("header", {}) or {}
        return cls(
            steam_id=str(steam_id),
            map_name=str(header.get("map_name", "unknown")),
            total_rounds=total_rounds,
            enriched_rounds=enriched_rounds,
            parsed=parsed,
        )

    @classmethod
    def from_storage(
        cls,
        *,
        steam_id: str,
        map_name: str,
        enriched_rounds: list[dict[str, Any]],
        total_rounds: int | None = None,
    ) -> MetricContext:
        """Rebuild from stored rounds, with no demo on disk.

        Only metrics whose requires is satisfied by enriched_rounds alone can
        run against this; the registry is what decides which those are.
        """
        return cls(
            steam_id=str(steam_id),
            map_name=map_name,
            total_rounds=total_rounds if total_rounds is not None else len(enriched_rounds),
            enriched_rounds=enriched_rounds,
            parsed={},
        )

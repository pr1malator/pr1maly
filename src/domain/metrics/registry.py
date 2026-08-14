"""A registry of measurements, each one addressable by name.

The problem this solves is versioning. ANALYZER_VERSION is a single number
stamped on every stored match, so changing anything about any measurement marks
every match in the database stale and asks the user to re-parse their whole
library. Worse, matches whose .dem the retention feature has already deleted can
never be refreshed at all, so they stay stale forever.

Almost none of that is necessary. Registering each measurement with its own
version and its own declared inputs makes two things possible:

  Only what changed is stale. Bumping `aim.reaction_time` marks the matches
  missing that one measurement, not every match ever imported.

  Some of it can be recomputed without the demo. `requires` says what a metric
  reads. Anything satisfied by `enriched_rounds` alone can be recalculated
  straight from SQLite, because those rounds are already stored in
  `round_stats.enriched_json` — including for matches whose demo is long gone.
  Only metrics that reach for raw parser frames need the file back.

It also gives the UI and the AI prompt builder a catalogue to read instead of
key names hardcoded in a template, and gives each measurement somewhere obvious
to be tested.

Registering a metric does not change where its value lands. `output_key` names
the exact position in the stored blob it has always occupied, so the shape on
disk is unchanged.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

# What a metric can declare it needs. Anything outside this set is a typo.
#
# ENRICHED_ROUNDS is the important one: it is persisted per round, so metrics
# that need nothing else can be recomputed from the database alone.
ENRICHED_ROUNDS = "enriched_rounds"
PARSED_PREFIX = "parsed:"


@dataclass(frozen=True)
class MetricSpec:
    """One measurement: what it is called, what it needs, and how to compute it."""

    id: str
    label: str
    group: str
    version: int
    requires: frozenset[str]
    output_key: str
    compute: Callable[[Any], Any]
    description: str = ""

    # Whether the version can be recorded as a key inside the metric's own
    # stored value. True for the ones whose blob is a record of named fields;
    # False where the keys carry meaning of their own — replay_data is keyed by
    # round number, so an extra key there would read as an extra round.
    version_in_blob: bool = True

    @property
    def needs_demo(self) -> bool:
        """True when this metric reads raw parser frames.

        Those are only available while the .dem is on disk. Everything else can
        be recomputed from what is already stored.
        """
        return any(r.startswith(PARSED_PREFIX) for r in self.requires)

    @property
    def parsed_frames(self) -> frozenset[str]:
        return frozenset(
            r[len(PARSED_PREFIX):] for r in self.requires if r.startswith(PARSED_PREFIX)
        )

    def as_catalog_entry(self) -> dict[str, Any]:
        """The public description, for GET /api/metrics and the prompt builder."""
        return {
            "id": self.id,
            "label": self.label,
            "group": self.group,
            "version": self.version,
            "output_key": self.output_key,
            "needs_demo": self.needs_demo,
            "description": self.description,
        }


@dataclass
class MetricRegistry:
    """Every registered measurement, keyed by id."""

    _specs: dict[str, MetricSpec] = field(default_factory=dict)

    def register(self, spec: MetricSpec) -> MetricSpec:
        if spec.id in self._specs:
            raise ValueError(f"metric {spec.id!r} is already registered")
        for requirement in spec.requires:
            if requirement != ENRICHED_ROUNDS and not requirement.startswith(PARSED_PREFIX):
                raise ValueError(
                    f"metric {spec.id!r} requires {requirement!r}, which is neither "
                    f"{ENRICHED_ROUNDS!r} nor a {PARSED_PREFIX}* frame"
                )
        self._specs[spec.id] = spec
        return spec

    def metric(
        self,
        *,
        id: str,
        label: str,
        group: str,
        version: int,
        requires: Iterable[str],
        output_key: str,
        description: str = "",
        version_in_blob: bool = True,
    ) -> Callable[[Callable[[Any], Any]], Callable[[Any], Any]]:
        """Decorator form. The function is returned unchanged, so it stays
        directly callable and directly testable."""

        def decorate(fn: Callable[[Any], Any]) -> Callable[[Any], Any]:
            self.register(MetricSpec(
                id=id,
                label=label,
                group=group,
                version=version,
                requires=frozenset(requires),
                output_key=output_key,
                compute=fn,
                version_in_blob=version_in_blob,
                description=description or (fn.__doc__ or "").strip().split("\n")[0],
            ))
            return fn

        return decorate

    # -- reading -----------------------------------------------------------

    def __len__(self) -> int:
        return len(self._specs)

    def __contains__(self, metric_id: object) -> bool:
        return metric_id in self._specs

    def __iter__(self):
        return iter(self._specs.values())

    def get(self, metric_id: str) -> MetricSpec:
        try:
            return self._specs[metric_id]
        except KeyError:
            raise KeyError(f"no metric registered as {metric_id!r}") from None

    def ids(self) -> list[str]:
        return sorted(self._specs)

    def groups(self) -> dict[str, list[MetricSpec]]:
        out: dict[str, list[MetricSpec]] = {}
        for spec in sorted(self._specs.values(), key=lambda s: s.id):
            out.setdefault(spec.group, []).append(spec)
        return out

    def select(
        self,
        *,
        ids: Iterable[str] | None = None,
        group: str | None = None,
        without_demo: bool = False,
    ) -> list[MetricSpec]:
        """The metrics matching every filter given, in stable id order."""
        chosen = list(self._specs.values())
        if ids is not None:
            wanted = set(ids)
            unknown = wanted - set(self._specs)
            if unknown:
                raise KeyError(f"no metric registered as {sorted(unknown)}")
            chosen = [s for s in chosen if s.id in wanted]
        if group is not None:
            chosen = [s for s in chosen if s.group == group]
        if without_demo:
            chosen = [s for s in chosen if not s.needs_demo]
        return sorted(chosen, key=lambda s: s.id)

    def versions(self) -> dict[str, int]:
        """The stamp written alongside a computed result."""
        return {spec.id: spec.version for spec in self._specs.values()}

    def stale_ids(self, stored_versions: dict[str, int] | None) -> list[str]:
        """Which metrics a stored match is missing or out of date on.

        A match written before per-metric versioning has no stamp at all, so
        everything registered counts as stale for it.
        """
        stored = stored_versions or {}
        return sorted(
            spec.id for spec in self._specs.values()
            if stored.get(spec.id, -1) < spec.version
        )

    def recomputable_without_demo(self, stale: Iterable[str]) -> list[str]:
        """Of those stale metrics, the ones that do not need the .dem back.

        This is the answer that matters to someone whose demos have been
        cleaned up: it is the difference between "re-parse 200 files" and
        "nothing to do".
        """
        return sorted(
            metric_id for metric_id in stale
            if metric_id in self._specs and not self._specs[metric_id].needs_demo
        )


# The application-wide registry. Metric modules register against this on import.
REGISTRY = MetricRegistry()
metric = REGISTRY.metric

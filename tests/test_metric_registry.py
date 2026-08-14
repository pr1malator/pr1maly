"""Tests for the metric registry.

The claim worth proving is the one the whole design rests on: a metric whose
inputs are satisfied by the stored enriched rounds produces the same value
whether it is computed from a freshly parsed demo or rebuilt from the database
with no demo at all. If that is not true, "recompute without the .dem" is a
story rather than a feature.
"""

from __future__ import annotations

import json

import pytest

from src.domain.metrics import REGISTRY, MetricContext
from src.domain.metrics.registry import (
    ENRICHED_ROUNDS,
    MetricRegistry,
    MetricSpec,
)
from tests.test_analysis_golden import STEAM_ID, _build_parsed_match

# ---------------------------------------------------------------------------
# The registry itself
# ---------------------------------------------------------------------------


@pytest.fixture
def registry():
    return MetricRegistry()


def _spec(reg, **overrides):
    defaults = dict(
        id="group.thing", label="Thing", group="group", version=1,
        requires={ENRICHED_ROUNDS}, output_key="thing",
    )
    defaults.update(overrides)

    @reg.metric(**defaults)
    def compute(ctx):
        """A first line."""
        return {"ok": True}

    return reg.get(defaults["id"])


def test_registering_makes_a_metric_addressable(registry):
    spec = _spec(registry)
    assert registry.get("group.thing") is spec
    assert "group.thing" in registry
    assert registry.ids() == ["group.thing"]


def test_the_decorated_function_is_returned_unchanged(registry):
    """It stays directly callable, so a metric can be tested without a context."""
    @registry.metric(id="a.b", label="B", group="a", version=1,
                     requires={ENRICHED_ROUNDS}, output_key="b")
    def compute(ctx):
        return 42

    assert compute(None) == 42


def test_registering_the_same_id_twice_is_refused(registry):
    _spec(registry)
    with pytest.raises(ValueError, match="already registered"):
        _spec(registry)


def test_an_unknown_requirement_is_refused(registry):
    """A typo in `requires` would silently make a metric look demo-free."""
    with pytest.raises(ValueError, match="neither"):
        _spec(registry, requires={"enriched_round"})  # missing the s


def test_asking_for_an_unregistered_metric_says_so(registry):
    with pytest.raises(KeyError, match="no metric registered"):
        registry.get("nope")


def test_description_falls_back_to_the_docstring(registry):
    assert _spec(registry).description == "A first line."


# ---------------------------------------------------------------------------
# requires -> needs_demo
# ---------------------------------------------------------------------------


def test_enriched_rounds_alone_does_not_need_the_demo(registry):
    assert _spec(registry).needs_demo is False


def test_any_parsed_frame_needs_the_demo(registry):
    spec = _spec(registry, requires={ENRICHED_ROUNDS, "parsed:weapon_fire"})
    assert spec.needs_demo is True
    assert spec.parsed_frames == {"weapon_fire"}


def test_selecting_the_metrics_that_survive_without_a_demo(registry):
    _spec(registry, id="a.cheap", output_key="x")
    _spec(registry, id="b.dear", output_key="y", requires={"parsed:player_death"})
    assert [s.id for s in registry.select(without_demo=True)] == ["a.cheap"]


def test_select_by_group_and_by_id(registry):
    _spec(registry, id="aim.one", group="aim", output_key="1")
    _spec(registry, id="aim.two", group="aim", output_key="2")
    _spec(registry, id="util.one", group="util", output_key="3")

    assert [s.id for s in registry.select(group="aim")] == ["aim.one", "aim.two"]
    assert [s.id for s in registry.select(ids=["util.one"])] == ["util.one"]
    with pytest.raises(KeyError):
        registry.select(ids=["nope"])


# ---------------------------------------------------------------------------
# Staleness — the reason per-metric versions exist
# ---------------------------------------------------------------------------


def test_a_match_with_no_stamps_is_stale_on_everything(registry):
    _spec(registry, id="a.one", output_key="1")
    _spec(registry, id="b.two", output_key="2")
    assert registry.stale_ids(None) == ["a.one", "b.two"]
    assert registry.stale_ids({}) == ["a.one", "b.two"]


def test_only_the_bumped_metric_goes_stale(registry):
    """The whole point. Today one version bump marks every match stale and asks
    the user to re-parse their entire library."""
    _spec(registry, id="a.unchanged", output_key="1", version=3)
    _spec(registry, id="b.bumped", output_key="2", version=4)

    stored = {"a.unchanged": 3, "b.bumped": 3}
    assert registry.stale_ids(stored) == ["b.bumped"]


def test_a_current_match_is_stale_on_nothing(registry):
    _spec(registry, id="a.one", output_key="1", version=2)
    assert registry.stale_ids({"a.one": 2}) == []


def test_a_stored_version_ahead_of_the_code_is_not_stale(registry):
    """Downgrading the app should not trigger a re-analysis of everything."""
    _spec(registry, id="a.one", output_key="1", version=2)
    assert registry.stale_ids({"a.one": 5}) == []


def test_which_stale_metrics_need_the_demo_back(registry):
    """The difference between "re-parse 200 files" and "nothing to do"."""
    _spec(registry, id="a.cheap", output_key="1")
    _spec(registry, id="b.dear", output_key="2", requires={"parsed:player_death"})

    stale = registry.stale_ids(None)
    assert stale == ["a.cheap", "b.dear"]
    assert registry.recomputable_without_demo(stale) == ["a.cheap"]


def test_versions_are_what_gets_stamped(registry):
    _spec(registry, id="a.one", output_key="1", version=7)
    assert registry.versions() == {"a.one": 7}


# ---------------------------------------------------------------------------
# The real registry
# ---------------------------------------------------------------------------


def test_every_registered_metric_has_a_distinct_output_key():
    keys = [s.output_key for s in REGISTRY]
    assert len(keys) == len(set(keys)), "two metrics would overwrite each other"


def test_output_keys_match_the_columns_already_stored():
    """Registering a metric must not move where its value lands on disk."""
    stored_blobs = {"aim_stats", "role_data", "utility_data", "impact_stats", "replay_data"}
    assert {s.output_key for s in REGISTRY} == stored_blobs


def test_ids_are_group_prefixed():
    for spec in REGISTRY:
        assert spec.id.startswith(f"{spec.group}."), f"{spec.id} is not in its group"


def test_the_catalog_entry_is_serialisable():
    import json

    for spec in REGISTRY:
        json.loads(json.dumps(spec.as_catalog_entry()))


def test_only_aim_survives_without_a_demo():
    """Recorded so a change here is deliberate: aim is the group that can be
    rebuilt from stored rounds, because it reads nothing else."""
    assert [s.id for s in REGISTRY.select(without_demo=True)] == ["aim.stats"]


# ---------------------------------------------------------------------------
# The claim: same answer with or without the demo
# ---------------------------------------------------------------------------


def test_a_demo_free_metric_gives_the_same_answer_from_storage():
    """Computed from a parsed demo, then rebuilt from the enriched rounds alone
    — as they come back out of round_stats.enriched_json — the values match.

    This is what makes it possible to bump an aim metric and refresh a match
    whose .dem was cleaned up months ago.
    """
    from src.processor import build_enriched_rounds

    parsed = _build_parsed_match()
    total_rounds = 4
    enriched = build_enriched_rounds(parsed, STEAM_ID, total_rounds)

    with_demo = MetricContext.from_parsed(
        parsed, STEAM_ID, enriched_rounds=enriched, total_rounds=total_rounds,
    )
    without_demo = MetricContext.from_storage(
        steam_id=STEAM_ID, map_name="de_mirage",
        enriched_rounds=enriched, total_rounds=total_rounds,
    )

    assert with_demo.has_demo is True
    assert without_demo.has_demo is False

    spec = REGISTRY.get("aim.stats")
    assert spec.needs_demo is False
    assert spec.compute(with_demo) == spec.compute(without_demo)


def test_versions_survive_a_round_trip_through_the_database(tmp_path):
    """The stamp is only useful if it comes back out.

    It is written inside each metric's own blob, which is already opaque TEXT,
    so this needs no schema change — but it does need the blob to be stored and
    reloaded intact.
    """
    from src.database import get_connection, get_match
    from src.processor import calculate_match_stats, stored_metric_versions
    from src.services.import_service import store_match

    stats = calculate_match_stats(_build_parsed_match(), STEAM_ID)
    expected = {s.id: s.version for s in REGISTRY if s.version_in_blob}
    assert stored_metric_versions(stats) == expected

    conn = get_connection(tmp_path / "stamp.db")
    try:
        match_id = store_match(conn, stats, filename="a.dem", steam_id=STEAM_ID)
        row = get_match(conn, match_id)
    finally:
        conn.close()

    reloaded = {
        key: json.loads(row[key])
        for key in ("aim_stats", "role_data", "utility_data", "impact_stats")
        if row.get(key)
    }
    recovered = stored_metric_versions(reloaded)
    assert recovered, "no versions came back off the stored match"
    for metric_id, version in recovered.items():
        assert version == REGISTRY.get(metric_id).version


def test_a_stored_match_with_current_stamps_is_stale_on_nothing():
    from src.processor import calculate_match_stats, stored_metric_versions

    stats = calculate_match_stats(_build_parsed_match(), STEAM_ID)
    stamped = stored_metric_versions(stats)

    # Only the blob-stamped metrics can report; replay opts out deliberately.
    unstamped = {s.id for s in REGISTRY if not s.version_in_blob}
    assert set(REGISTRY.stale_ids(stamped)) == unstamped


def test_replay_frames_are_not_stamped_because_the_keys_are_round_numbers():
    """An extra key in replay_data would read as an extra round."""
    from src.processor import calculate_match_stats

    stats = calculate_match_stats(_build_parsed_match(), STEAM_ID)
    assert REGISTRY.get("replay.frames").version_in_blob is False
    assert "_metric_version" not in stats["replay_data"]
    assert all(isinstance(k, int) for k in stats["replay_data"])


def test_an_old_match_with_no_stamps_reports_everything_stale():
    """A match imported before per-metric versioning existed."""
    from src.processor import stored_metric_versions

    legacy = {"aim_stats": {"aim_rating": 80.0}, "role_data": None}
    assert stored_metric_versions(legacy) == {}
    assert REGISTRY.stale_ids(stored_metric_versions(legacy)) == REGISTRY.ids()


def test_a_storage_context_refuses_a_frame_it_does_not_have():
    """A metric reaching for a frame it did not declare should fail loudly
    rather than quietly measuring nothing."""
    ctx = MetricContext.from_storage(
        steam_id=STEAM_ID, map_name="de_mirage", enriched_rounds=[],
    )
    with pytest.raises(LookupError, match="requires="):
        ctx.frame("player_death")


def test_derived_values_are_computed_once():
    ctx = MetricContext.from_storage(
        steam_id=STEAM_ID, map_name="de_mirage", enriched_rounds=[],
    )
    calls = []
    for _ in range(3):
        ctx.derive("thing", lambda: calls.append(1) or "value")
    assert calls == [1]
    assert ctx.derive("thing", lambda: "other") == "value"


def test_spec_is_frozen():
    """A registered metric must not be mutated at runtime."""
    from dataclasses import FrozenInstanceError

    with pytest.raises(FrozenInstanceError):
        REGISTRY.get("aim.stats").version = 99  # type: ignore[misc]


def test_metric_spec_requires_is_a_frozenset():
    assert isinstance(REGISTRY.get("aim.stats").requires, frozenset)
    with pytest.raises(AttributeError):
        REGISTRY.get("aim.stats").requires.add("parsed:anything")  # type: ignore[attr-defined]


def test_groups_are_stable_and_sorted():
    groups = REGISTRY.groups()
    assert set(groups) == {"aim", "utility", "roles", "impact", "replay"}
    for specs in groups.values():
        assert [s.id for s in specs] == sorted(s.id for s in specs)


def test_a_spec_can_be_built_directly():
    spec = MetricSpec(
        id="x.y", label="Y", group="x", version=1,
        requires=frozenset({ENRICHED_ROUNDS}), output_key="y",
        compute=lambda ctx: None,
    )
    assert spec.needs_demo is False

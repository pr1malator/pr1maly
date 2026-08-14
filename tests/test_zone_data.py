"""Checks on the map zone data.

Two files describe where a player is and what that means: the callout zones say
which rectangle a coordinate falls in, and the role zones say which callouts
indicate which role. They have to agree, and nothing enforced that while both
were Python literals.

The failure mode is silent. A role lists a callout name; the name is compared
against what the lookup produces; a name matching nothing contributes no score.
The role does not error, it just stops being detected — which is how
de_overpass CT ended up with no role claiming the A bombsite, and how Inferno
CT once lost Apartments.
"""

from __future__ import annotations

import json

import pytest

from src.domain.callouts import ZONES_DIR, supported_maps, zone_document, zones_for
from src.domain.metrics.role_zones import ROLE_ZONES_DIR, role_zones, roles_for

MAPS = supported_maps()
SITE_LABELS = ("A Site", "B Site")


# ---------------------------------------------------------------------------
# The callout zones
# ---------------------------------------------------------------------------


def test_there_are_maps_to_check():
    assert MAPS, "no callout zone files were found at all"


@pytest.mark.parametrize("map_name", MAPS)
def test_every_zone_is_a_valid_rectangle(map_name):
    for label, min_x, max_x, min_y, max_y in zones_for(map_name):
        assert min_x < max_x, f"{map_name}/{label}: min_x is not below max_x"
        assert min_y < max_y, f"{map_name}/{label}: min_y is not below max_y"


@pytest.mark.parametrize("map_name", MAPS)
def test_zone_labels_are_unique_within_a_map(map_name):
    """Two zones sharing a label makes the second unreachable — the first
    match wins — and makes the role mapping ambiguous."""
    labels = [z[0] for z in zones_for(map_name)]
    duplicates = {label for label in labels if labels.count(label) > 1}
    assert not duplicates, f"{map_name} has duplicate labels: {sorted(duplicates)}"


@pytest.mark.parametrize("map_name", MAPS)
def test_file_order_is_preserved_by_the_loader(map_name):
    """Order decides which zone wins, so the loader must not sort."""
    doc = zone_document(map_name)
    assert [z["label"] for z in doc["zones"]] == [z[0] for z in zones_for(map_name)]


# Nuke is stacked: the B bombsite is directly underneath the A bombsite, so the
# two share a 2D footprint and their rectangles are identical. Ordering cannot
# separate them — whichever is listed first wins every position on both. Telling
# them apart needs the Z coordinate, which this model does not carry, so B Site
# on Nuke currently reads as A Site. Recorded as a known limitation rather than
# hidden: remove this entry when the lookup grows a height dimension.
_KNOWN_2D_COLLISIONS = {("de_nuke", "B Site", "A Site")}


@pytest.mark.parametrize("map_name", MAPS)
def test_a_specific_zone_is_never_shadowed_by_one_that_encloses_it(map_name):
    """First match wins, so a zone fully inside an earlier one is dead.

    "B Apartments Entrance" sits inside "B Apartments" and is listed first for
    exactly this reason. This check found four zones that were not: Upper
    Tunnels (inverted y bounds, so it could never match at all), Xbox inside
    Mid, Boiler inside Arch, and B Pillar inside B Site.
    """
    zones = zones_for(map_name)
    for i, (label, min_x, max_x, min_y, max_y) in enumerate(zones):
        for earlier_label, e_min_x, e_max_x, e_min_y, e_max_y in zones[:i]:
            if (map_name, label, earlier_label) in _KNOWN_2D_COLLISIONS:
                continue
            encloses = (
                e_min_x <= min_x and e_max_x >= max_x
                and e_min_y <= min_y and e_max_y >= max_y
            )
            assert not encloses, (
                f"{map_name}: {label!r} is entirely inside {earlier_label!r}, "
                f"which is listed first — {label!r} can never match"
            )


def test_the_known_2d_collisions_are_still_collisions():
    """So the exception above disappears the moment it stops being needed."""
    for map_name, inner, outer in _KNOWN_2D_COLLISIONS:
        zones = dict((z[0], z[1:]) for z in zones_for(map_name))
        assert inner in zones and outer in zones
        i_min_x, i_max_x, i_min_y, i_max_y = zones[inner]
        o_min_x, o_max_x, o_min_y, o_max_y = zones[outer]
        assert (
            o_min_x <= i_min_x and o_max_x >= i_max_x
            and o_min_y <= i_min_y and o_max_y >= i_max_y
        ), (
            f"{map_name}: {inner!r} is no longer inside {outer!r} — remove it "
            f"from _KNOWN_2D_COLLISIONS"
        )


@pytest.mark.parametrize("map_name", MAPS)
def test_every_map_defines_both_bombsites(map_name):
    labels = {z[0] for z in zones_for(map_name)}
    for site in SITE_LABELS:
        assert site in labels, f"{map_name} has no {site!r} zone"


@pytest.mark.parametrize("path", sorted(ZONES_DIR.glob("*.json")), ids=lambda p: p.stem)
def test_callout_files_parse_and_are_named_for_their_map(path):
    doc = json.loads(path.read_text(encoding="utf-8"))
    assert doc["map"] == path.stem
    assert doc["zones"], f"{path.name} defines no zones"


# ---------------------------------------------------------------------------
# The role zones, against the callouts
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("map_name", sorted(role_zones()))
def test_every_role_callout_exists_on_that_map(map_name):
    """The check that would have caught both bugs.

    A callout name that matches no zone contributes no score, and the role
    quietly stops being detected.
    """
    labels = {z[0] for z in zones_for(map_name)}
    assert labels, f"{map_name} has role zones but no callout zones"

    dangling = []
    for side, roles in role_zones()[map_name].items():
        for role, callouts in roles.items():
            for callout in callouts:
                if callout not in labels:
                    dangling.append(f"{side}/{role}: {callout!r}")
    assert not dangling, f"{map_name} references callouts that do not exist: {dangling}"


@pytest.mark.parametrize("map_name", sorted(role_zones()))
@pytest.mark.parametrize("side", ["CT", "T"])
def test_both_bombsites_are_claimed_by_some_role(map_name, side):
    """de_overpass CT had no role claiming the A site, so an A anchor scored
    nothing from the position they actually held."""
    claimed = {c for cs in roles_for(map_name, side).values() for c in cs}
    labels = {z[0] for z in zones_for(map_name)}
    for site in SITE_LABELS:
        if site in labels:
            assert site in claimed, (
                f"{map_name} {side}: no role claims {site!r}, so a player holding "
                f"it contributes nothing to their role score"
            )


@pytest.mark.parametrize("map_name", sorted(role_zones()))
@pytest.mark.parametrize("side", ["CT", "T"])
def test_each_side_defines_some_roles(map_name, side):
    roles = roles_for(map_name, side)
    assert roles, f"{map_name} {side} defines no roles"
    for role, callouts in roles.items():
        assert callouts, f"{map_name} {side}/{role} lists no callouts"


@pytest.mark.parametrize("map_name", sorted(role_zones()))
def test_role_names_are_distinct_within_a_side(map_name):
    for side, roles in role_zones()[map_name].items():
        assert len(roles) == len(set(roles)), f"{map_name} {side} has duplicate roles"


def test_every_map_with_callouts_has_role_zones():
    """A map the app can locate players on but cannot classify is a half-built
    map, and shows up as an empty role chart."""
    assert set(role_zones()) == set(MAPS)


@pytest.mark.parametrize(
    "path", sorted(ROLE_ZONES_DIR.glob("*.json")), ids=lambda p: p.stem
)
def test_role_files_parse_and_are_named_for_their_map(path):
    doc = json.loads(path.read_text(encoding="utf-8"))
    assert doc["map"] == path.stem
    assert set(doc["sides"]) <= {"CT", "T"}

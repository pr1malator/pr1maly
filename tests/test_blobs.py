"""Tests for reading the opaque JSON columns.

These blobs are the only surviving record of a match whose demo the retention
feature has deleted, and they were written by whatever analyzer version was
current at the time. A row from a year ago is a shape nobody is thinking about.
A decoder that raises on one takes down a page instead of degrading, so the
cases below are mostly about not raising.
"""

from __future__ import annotations

import json

import pytest

from src.domain.blobs import (
    MATCH_BLOB_COLUMNS,
    decode,
    decode_dict,
    decode_match_blobs,
    stored_value,
)

# ---------------------------------------------------------------------------
# decode
# ---------------------------------------------------------------------------


def test_decodes_a_stored_string():
    assert decode('{"aim_rating": 84.0}') == {"aim_rating": 84.0}


def test_absent_is_the_default_not_an_error():
    """A match analysed before a measurement existed has NULL in that column."""
    assert decode(None) is None
    assert decode("") is None
    assert decode(None, default={}) == {}


@pytest.mark.parametrize("corrupt", ["{not json", "", "   ", "[unclosed", "\x00"])
def test_corrupt_content_falls_back_rather_than_raising(corrupt):
    assert decode(corrupt, default={"fallback": True}) == {"fallback": True}


def test_an_already_decoded_value_passes_through():
    """Some callers decode before handing the row on; this must be idempotent."""
    value = {"aim_rating": 84.0}
    assert decode(value) is value
    assert decode([1, 2, 3]) == [1, 2, 3]


def test_a_json_null_reads_as_the_default():
    """`json.dumps(None)` is the string "null", which is a stored shape."""
    assert decode("null") is None
    assert decode("null", default={}) == {}


def test_bytes_are_accepted():
    assert decode(b'{"a": 1}') == {"a": 1}


# ---------------------------------------------------------------------------
# decode_dict — shape as well as parse
# ---------------------------------------------------------------------------


def test_decode_dict_gives_an_empty_dict_for_anything_unusable():
    for raw in (None, "", "{bad", "null", json.dumps([1, 2]), json.dumps("text"), 7):
        assert decode_dict(raw) == {}, raw


def test_decode_dict_guards_against_a_list_reaching_a_get_call():
    """A blob holding a bare array used to reach code that called .get on it."""
    assert decode_dict('[{"aim_rating": 1}]') == {}


def test_decode_dict_passes_a_real_object_through():
    assert decode_dict('{"a": {"b": 1}}') == {"a": {"b": 1}}


# ---------------------------------------------------------------------------
# decode_match_blobs
# ---------------------------------------------------------------------------


def test_all_four_blob_columns_are_decoded():
    match = {
        "match_id": "m1",
        "aim_stats": json.dumps({"aim_rating": 84.0}),
        "role_data": json.dumps({"map": "de_mirage"}),
        "utility_data": json.dumps({"utility_rating": 60.0}),
        "impact_stats": json.dumps({"total_swing": 3.2}),
        "kills": 20,
    }
    decode_match_blobs(match)

    assert match["aim_stats"]["aim_rating"] == 84.0
    assert match["role_data"]["map"] == "de_mirage"
    assert match["utility_data"]["utility_rating"] == 60.0
    assert match["impact_stats"]["total_swing"] == 3.2
    assert match["kills"] == 20, "non-blob columns must be left alone"


def test_missing_columns_are_not_invented():
    """A row selected without the blob columns must not gain them as None."""
    match = {"match_id": "m1", "kills": 20}
    decode_match_blobs(match)
    assert set(match) == {"match_id", "kills"}


def test_a_row_with_null_blobs_decodes_to_none():
    match = dict.fromkeys(MATCH_BLOB_COLUMNS)
    decode_match_blobs(match)
    assert all(match[c] is None for c in MATCH_BLOB_COLUMNS)


def test_one_corrupt_blob_does_not_take_the_others_with_it():
    match = {
        "aim_stats": "{corrupt",
        "utility_data": json.dumps({"utility_rating": 60.0}),
    }
    decode_match_blobs(match)
    assert match["aim_stats"] is None
    assert match["utility_data"]["utility_rating"] == 60.0


def test_decoding_returns_the_same_dict_for_chaining():
    match = {"aim_stats": json.dumps({"a": 1})}
    assert decode_match_blobs(match) is match


# ---------------------------------------------------------------------------
# stored_value
# ---------------------------------------------------------------------------


def test_stored_value_reaches_into_a_blob():
    match = {"aim_stats": json.dumps({"aim_rating": 84.0})}
    assert stored_value(match, "aim_stats", "aim_rating") == 84.0


def test_stored_value_walks_a_path():
    match = {"aim_stats": json.dumps({"movement": {"median": 23.7, "n": 16}})}
    assert stored_value(match, "aim_stats", "movement.median") == 23.7
    assert stored_value(match, "aim_stats", "movement.n") == 16


def test_stored_value_is_none_when_anything_along_the_way_is_missing():
    match = {"aim_stats": json.dumps({"movement": {"median": 23.7}})}
    assert stored_value(match, "aim_stats", "movement.absent") is None
    assert stored_value(match, "aim_stats", "absent.median") is None
    assert stored_value(match, "utility_data", "utility_rating") is None
    assert stored_value({}, "aim_stats", "aim_rating") is None


def test_stored_value_does_not_walk_into_a_scalar():
    """`movement` being a number rather than an object is an older shape."""
    match = {"aim_stats": json.dumps({"movement": 23.7})}
    assert stored_value(match, "aim_stats", "movement.median") is None


def test_stored_value_survives_a_corrupt_blob():
    assert stored_value({"aim_stats": "{bad"}, "aim_stats", "aim_rating") is None

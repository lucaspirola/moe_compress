"""Tests for moe_compress.stage2.grouping (Task 5 of the plugin refactor).

Provides direct behavioural coverage for _build_grouped_from_assignment +
_promote_orphans. _apply_skip_merge_floor coverage stays in
test_stage2_skip_merge.py.
"""
from __future__ import annotations

import logging

from moe_compress.stage2.grouping import (
    _apply_skip_merge_floor,
    _build_grouped_from_assignment,
    _promote_orphans,
)


# ---------------------------------------------------------------------------
# _build_grouped_from_assignment - direct unit coverage
# ---------------------------------------------------------------------------


def test_build_grouped_from_assignment_assigns_children_to_named_centroids():
    """Assignment indices point into centroid_ids; children resolve via
    noncentroid_ids. Every centroid is keyed by its own EID, with itself as
    the first list element."""
    centroid_ids = [3, 7]
    noncentroid_ids = [1, 2, 5]
    # children 0 (=eid 1) -> centroid 0 (=eid 3),
    # children 1 (=eid 2) -> centroid 1 (=eid 7),
    # children 2 (=eid 5) -> centroid 0 (=eid 3).
    assignment = [0, 1, 0]

    grouped = _build_grouped_from_assignment(assignment, centroid_ids, noncentroid_ids)

    assert grouped == {3: [3, 1, 5], 7: [7, 2]}


def test_build_grouped_from_assignment_skips_negative_entries():
    """A ``-1`` assignment entry means the solver could not place the child;
    the helper must NOT add it to any group. (Orphan promotion is a separate
    step performed by ``_promote_orphans``.)"""
    centroid_ids = [10]
    noncentroid_ids = [4, 8]
    assignment = [0, -1]  # child 4 placed in 10; child 8 unassigned.

    grouped = _build_grouped_from_assignment(assignment, centroid_ids, noncentroid_ids)

    assert grouped == {10: [10, 4]}
    assert 8 not in grouped
    assert -1 not in grouped


def test_build_grouped_from_assignment_empty_inputs_yield_empty_dict():
    """No centroids and no children -> empty grouped dict (degenerate but legal)."""
    assert _build_grouped_from_assignment([], [], []) == {}


# ---------------------------------------------------------------------------
# _promote_orphans - in-place mutation + WARNING emission
# ---------------------------------------------------------------------------


def test_promote_orphans_promotes_unassigned_child_to_singleton(caplog):
    """One orphan (assignment[1] = -1) -> singleton centroid; one WARNING."""
    log = logging.getLogger("moe_compress.test_pipeline_grouping.promote")
    # Some pytest plugins in this env default new loggers to propagate=False;
    # the real stage2 logger has propagate=True, so mirror that here.
    log.propagate = True
    grouped: dict[int, list[int]] = {3: [3, 1], 7: [7]}
    ream_centroid_ids = [3, 7]
    ream_noncentroid_ids = [1, 5]
    assignment = [0, -1]  # child 1 (=eid 5) orphan.

    with caplog.at_level(logging.WARNING):
        _promote_orphans(grouped, ream_centroid_ids, ream_noncentroid_ids,
                         assignment, layer_idx=4, log=log)

    assert grouped == {3: [3, 1], 7: [7], 5: [5]}
    assert ream_centroid_ids == [3, 7, 5]  # appended, not sorted/dedup'd.
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    msg = warnings[0].getMessage()
    assert "layer 4" in msg and "expert 5" in msg
    assert "promoted to singleton centroid" in msg


def test_promote_orphans_no_unassigned_children_is_a_noop(caplog):
    """All assignment entries >= 0 -> no mutation, no WARNING."""
    log = logging.getLogger("moe_compress.test_pipeline_grouping.noop")
    log.propagate = True
    grouped: dict[int, list[int]] = {3: [3, 1], 7: [7, 2]}
    ream_centroid_ids = [3, 7]
    assignment = [0, 1]

    grouped_before = {k: list(v) for k, v in grouped.items()}
    centroids_before = list(ream_centroid_ids)

    with caplog.at_level(logging.WARNING):
        _promote_orphans(grouped, ream_centroid_ids, [1, 2],
                         assignment, layer_idx=0, log=log)

    assert grouped == grouped_before
    assert ream_centroid_ids == centroids_before
    assert [r for r in caplog.records if r.levelno == logging.WARNING] == []


def test_promote_orphans_multiple_orphans_append_in_iteration_order(caplog):
    """Two orphans -> two singleton entries + two WARNINGs, appended in
    ream_noncentroid_ids order (NOT sorted; caller re-sorts via line 1122)."""
    log = logging.getLogger("moe_compress.test_pipeline_grouping.multi")
    log.propagate = True
    # In the real flow the caller's inline >=0 loop has already populated the
    # placed child (eid 9) into centroid 2. _promote_orphans only handles the
    # <0 entries (children at indices 1 and 2 in the assignment).
    grouped: dict[int, list[int]] = {2: [2, 9]}
    ream_centroid_ids = [2]
    ream_noncentroid_ids = [9, 1, 4]
    assignment = [0, -1, -1]  # child 0 placed; 1 and 2 orphan.

    with caplog.at_level(logging.WARNING):
        _promote_orphans(grouped, ream_centroid_ids, ream_noncentroid_ids,
                         assignment, layer_idx=11, log=log)

    assert grouped == {2: [2, 9], 1: [1], 4: [4]}
    assert ream_centroid_ids == [2, 1, 4]  # iteration order, not sorted.
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 2
    assert "expert 1" in warnings[0].getMessage()
    assert "expert 4" in warnings[1].getMessage()

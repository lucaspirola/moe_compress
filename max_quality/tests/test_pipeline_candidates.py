"""Unit tests for ``moe_compress.pipeline.candidates``.

Verifies the ``CandidateBag`` data structure's add / by_tag / merge /
to_provenance_dict semantics, including the byte-identical shape match against
the legacy ``_collect_candidates`` return value.
"""

from __future__ import annotations

from moe_compress.pipeline.candidates import CandidateBag


def test_candidate_bag_add_and_tags_for():
    bag = CandidateBag()
    bag.add(0, 1, "a")
    bag.add(0, 1, "b")
    assert bag.tags_for(0, 1) == ("a", "b")


def test_candidate_bag_len_counts_unique_pairs():
    bag = CandidateBag()
    bag.add(0, 1, "a")
    bag.add(0, 1, "b")
    assert len(bag) == 1


def test_candidate_bag_by_tag_inverts_correctly():
    bag = CandidateBag()
    bag.add(0, 1, "x")
    bag.add(0, 2, "x")
    bag.add(0, 1, "y")
    assert bag.by_tag("x") == {0: [1, 2]}
    assert bag.by_tag("y") == {0: [1]}


def test_candidate_bag_by_tag_missing_returns_empty():
    bag = CandidateBag()
    bag.add(0, 1, "x")
    assert bag.by_tag("never_added") == {}


def test_candidate_bag_to_provenance_dict_matches_collect_candidates_shape():
    """Mirror a synthetic Phase-C union over 4 pairs × 3 tags.

    Shape contract = ``dict[tuple[int, int] -> sorted list[str]]`` —
    byte-equivalent to the legacy ``_collect_candidates`` return
    ``{key: sorted(tags) for ...}``.
    """
    bag = CandidateBag()
    # Pair (0, 1) carries tags "magnitude" and "aimer"
    bag.add(0, 1, "magnitude")
    bag.add(0, 1, "aimer")
    # Pair (0, 2) carries only "sink_token"
    bag.add(0, 2, "sink_token")
    # Pair (1, 3) carries "magnitude" and "sink_token"
    bag.add(1, 3, "magnitude")
    bag.add(1, 3, "sink_token")
    # Pair (1, 0) carries only "aimer"
    bag.add(1, 0, "aimer")

    out = bag.to_provenance_dict()

    # Shape: dict keyed by (int, int) -> list[str], strs sorted ascending.
    assert isinstance(out, dict)
    for key, tags in out.items():
        assert isinstance(key, tuple) and len(key) == 2
        assert all(isinstance(k, int) for k in key)
        assert isinstance(tags, list)
        assert all(isinstance(t, str) for t in tags)
        assert tags == sorted(tags)

    # Exact byte-equivalent expected value.
    expected = {
        (0, 1): ["aimer", "magnitude"],
        (0, 2): ["sink_token"],
        (1, 3): ["magnitude", "sink_token"],
        (1, 0): ["aimer"],
    }
    assert out == expected


def test_candidate_bag_merge_unions_tags():
    a = CandidateBag()
    a.add(0, 1, "x")
    a.add(0, 1, "y")
    b = CandidateBag()
    b.add(0, 1, "y")
    b.add(0, 2, "z")

    merged = a.merge(b)

    assert merged.tags_for(0, 1) == ("x", "y")
    assert merged.tags_for(0, 2) == ("z",)
    # Originals unmodified.
    assert a.tags_for(0, 2) == ()
    assert b.tags_for(0, 1) == ("y",)
    # New bag is a separate instance.
    assert merged is not a
    assert merged is not b


def test_candidate_bag_to_blacklist_form_sorts_experts():
    bag = CandidateBag()
    bag.add(0, 5, "a")
    bag.add(0, 2, "a")
    bag.add(0, 9, "b")
    assert bag.to_blacklist_form() == {0: [2, 5, 9]}


def test_candidate_bag_iteration_yields_pairs():
    bag = CandidateBag()
    bag.add(0, 1, "x")
    bag.add(0, 1, "y")
    bag.add(1, 0, "z")

    keys = list(bag)
    assert (0, 1) in keys
    assert (1, 0) in keys
    assert len(keys) == 2

    items = dict(bag.items())
    assert items[(0, 1)] == ("x", "y")
    assert items[(1, 0)] == ("z",)


def test_candidate_bag_contains_pair_tuple():
    bag = CandidateBag()
    bag.add(0, 1, "x")
    assert (0, 1) in bag
    assert (0, 2) not in bag
    # Non-pair keys reject cleanly.
    assert "foo" not in bag
    assert 0 not in bag

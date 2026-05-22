"""``CandidateBag`` — ordered union of (layer, expert) candidates with provenance tags.

Replaces the inline ``dict[(layer_idx, expert_idx) -> list[str]]`` pattern
scattered through Stage 1's candidate-collection phase. Legacy logic
(relocated into the detector plugins in sub-tasks 5-8):

* The legacy ``_collect_candidates`` built a
  ``dict[(layer_idx, expert_idx) -> sorted list[str]]`` by
  ``setdefault((l, e), set()).add(tag)`` and then
  ``{key: sorted(tags) for key, tags in out.items()}`` at return.
* The legacy ``_candidates_by_provenance`` inverted that:
  given a single tag string, returned ``{str(li): sorted([expert_idx, ...])}``
  for all (l, e) whose tag list contains that tag.

``CandidateBag`` reproduces those return shapes byte-identically so the
golden snapshot stays green when later sub-tasks switch to it.
"""

from __future__ import annotations

from typing import Iterator


class CandidateBag:
    """Ordered union of (layer, expert) candidate sets with provenance tags.

    Storage shape::

        self._tags: dict[tuple[int, int], set[str]]

    All public reads materialise sorted ``tuple[str, ...]`` so callers can
    rely on deterministic iteration without sprinkling ``sorted(...)`` calls.

    Design choices
    --------------
    * Internal ``set``, external sorted ``tuple`` — the existing
      ``_collect_candidates`` uses ``dict[(l, e), set[str]]`` and sorts at
      return; mirroring that storage choice means a single dict-of-sets is
      the source of truth and every read materialises sorted output.
    * :meth:`merge` returns a new bag, never mutates — matches the overarching
      plan's "no plugin-to-plugin coupling" principle. Candidate detectors
      must not see each other's intermediates.
    * :meth:`to_provenance_dict` is the byte-equivalence check the unit test
      uses; shape ``dict[tuple[int, int], list[str]]`` with sorted str tag
      lists matches ``_collect_candidates``'s output exactly.
    * No ``remove`` / ``discard`` — candidate sets are write-only in Stage 1's
      lifecycle. Pruning happens via the ablation filter, which produces its
      own blacklist; the candidate bag is never edited after Phase C.
    * :meth:`by_tag` returns int-keyed dict, not str-keyed. The orchestrator
      wraps with ``{str(li): es ...}`` when building the JSON fragment
      (matches the inline pattern). Keeping the bag int-keyed makes
      downstream arithmetic / merging cheaper and reflects the actual key
      type.
    """

    def __init__(self) -> None:
        self._tags: dict[tuple[int, int], set[str]] = {}

    # ----- writes ----------------------------------------------------------
    def add(self, layer_idx: int, expert_idx: int, tag: str) -> None:
        """Add (or re-tag) a (layer, expert) candidate with one provenance tag.

        Multiple ``add`` calls with the same (layer, expert) accumulate tags
        in the bag's set; final :meth:`tags_for` returns them sorted
        lexicographically.
        """
        key = (int(layer_idx), int(expert_idx))
        self._tags.setdefault(key, set()).add(str(tag))

    # ----- per-pair reads --------------------------------------------------
    def tags_for(self, layer_idx: int, expert_idx: int) -> tuple[str, ...]:
        key = (int(layer_idx), int(expert_idx))
        if key not in self._tags:
            return ()
        return tuple(sorted(self._tags[key]))

    # ----- by-tag inversion (replaces _candidates_by_provenance) -----------
    def by_tag(self, tag: str) -> dict[int, list[int]]:
        """Return ``{layer_idx: sorted([expert_idx, ...])}`` for all candidates carrying ``tag``.

        Byte-equivalent semantics to the legacy inline
        ``_candidates_by_provenance`` except keys are ``int`` (not ``str``).
        The current orchestrator wraps the result with ``{str(li): es ...}``;
        this stays the orchestrator's responsibility — the bag emits ints so
        downstream merging stays cheap.
        """
        out: dict[int, list[int]] = {}
        for (li, e), tags in self._tags.items():
            if tag in tags:
                out.setdefault(li, []).append(e)
        return {li: sorted(es) for li, es in out.items()}

    # ----- union / merge ---------------------------------------------------
    def merge(self, other: "CandidateBag") -> "CandidateBag":
        """Return a NEW bag with the union of self's and ``other``'s tags.

        Per-pair tag sets are unioned. Neither input is mutated.
        """
        out = CandidateBag()
        out._tags = {k: set(v) for k, v in self._tags.items()}
        for k, v in other._tags.items():
            out._tags.setdefault(k, set()).update(v)
        return out

    # ----- export forms ----------------------------------------------------
    def to_provenance_dict(self) -> dict[tuple[int, int], list[str]]:
        """Frozen export matching the existing ``_collect_candidates`` return shape.

        Byte-equivalent to the legacy ``_collect_candidates`` return
        ``{key: sorted(tags) for key, tags in out.items()}``. Tests use
        this to verify byte-equivalence.
        """
        return {key: sorted(tags) for key, tags in self._tags.items()}

    def to_blacklist_form(self) -> dict[int, list[int]]:
        """Project to ``{layer_idx: sorted([expert_idx, ...])}`` discarding tags.

        Used as the pre-ablation candidate projection consumed by the
        ablation filter; this is **not** the post-ablation blacklist (that
        comes from the ablation filter's threshold step). Naming reflects
        shape, not semantics.
        """
        out: dict[int, list[int]] = {}
        for (li, e), _tags in self._tags.items():
            out.setdefault(li, []).append(e)
        return {li: sorted(es) for li, es in out.items()}

    # ----- iteration / size ------------------------------------------------
    def items(self) -> Iterator[tuple[tuple[int, int], tuple[str, ...]]]:
        """Yield ``((layer, expert), sorted-tags-tuple)`` pairs in insertion order."""
        for k, v in self._tags.items():
            yield k, tuple(sorted(v))

    def __iter__(self) -> Iterator[tuple[int, int]]:
        return iter(self._tags)

    def __len__(self) -> int:
        return len(self._tags)

    def __contains__(self, key: object) -> bool:
        if isinstance(key, tuple) and len(key) == 2:
            try:
                return (int(key[0]), int(key[1])) in self._tags
            except (TypeError, ValueError):
                return False
        return False

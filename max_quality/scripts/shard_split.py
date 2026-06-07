#!/usr/bin/env python3
"""Split a self-traces JSONL into N contiguous, non-overlapping shards.

Part of the data-parallel calibration-capture machinery (see
``tasks/CALIB_PARALLEL_CAPTURE_DESIGN.md``). The 8000-row corpus is split
into N shards, each replayed by an independent single-GPU process, then the
per-shard sidecars are merged back into one identical to a single full run.

The split is **contiguous + non-overlapping**: shard ``k`` gets rows
``[k*ceil(total/N) : min((k+1)*ceil(total/N), total)]`` written to
``<out_dir>/shard_k/shard_k.jsonl``. The last shard may be shorter; an empty
shard (more shards than rows) is an error.

HARD VERIFICATION (the disjointness guarantee — design §17):
  (a) the sum of per-shard line counts equals the input line count, AND
  (b) the per-row stable key (``_attempt_idx`` if present, else ``seed_idx``,
      else the 0-based line index) forms DISJOINT sets across shards whose
      UNION equals the full set of keys.
If either check fails the script exits non-zero with a clear message; no
partial shard set is left implying success.

Usage
-----
.. code-block:: bash

    python max_quality/scripts/shard_split.py \\
        artifacts/_shared/self_traces.jsonl 4 \\
        --out-dir artifacts/_shared/parallel_capture

Prints the per-shard row counts on success.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


def _row_key(row: dict, line_idx: int) -> "tuple[str, object]":
    """Return a stable per-row key for the disjointness check.

    Preference order matches the design doc: ``_attempt_idx`` (the canonical
    per-trace identity stamped by the build driver), else ``seed_idx`` (the
    v9 duplicate of the same), else the 0-based input line index.

    The key is a ``(kind, value)`` tuple so two rows that happen to share a
    numeric value across different key kinds can never collide silently — in
    practice a single corpus uses one kind uniformly, but tagging the kind
    makes the union/disjoint assertions unambiguous.
    """
    if "_attempt_idx" in row and row["_attempt_idx"] is not None:
        return ("_attempt_idx", int(row["_attempt_idx"]))
    if "seed_idx" in row and row["seed_idx"] is not None:
        return ("seed_idx", int(row["seed_idx"]))
    return ("line_idx", int(line_idx))


def _read_rows(jsonl_path: Path) -> "list[tuple[str, dict]]":
    """Read the input JSONL. Returns a list of ``(raw_line, parsed_row)``.

    Blank lines are skipped (they carry no row and would distort the line
    count contract). Raises ``ValueError`` on malformed JSON with the
    offending line number.
    """
    rows: list[tuple[str, dict]] = []
    with jsonl_path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"shard_split: invalid JSON at line {lineno} of "
                    f"{jsonl_path}: {exc}"
                ) from exc
            rows.append((stripped, parsed))
    return rows


def split_jsonl(
    jsonl_path: Path,
    n_shards: int,
    out_dir: Path,
) -> "list[int]":
    """Split ``jsonl_path`` into ``n_shards`` contiguous shard files.

    Writes ``<out_dir>/shard_k/shard_k.jsonl`` for k in [0, n_shards).
    Performs the HARD disjoint+complete verification before returning.

    Returns the per-shard row counts (length ``n_shards``). Raises
    ``ValueError`` / ``RuntimeError`` on any failure; the caller (CLI) maps
    those to a non-zero exit.
    """
    if n_shards < 1:
        raise ValueError(f"shard_split: n_shards must be >= 1, got {n_shards}")

    rows = _read_rows(jsonl_path)
    total = len(rows)
    if total == 0:
        raise ValueError(f"shard_split: input {jsonl_path} has 0 rows")
    if n_shards > total:
        raise ValueError(
            f"shard_split: n_shards={n_shards} exceeds row count={total}; "
            f"every shard must be non-empty (would create empty shards)."
        )

    per_shard = math.ceil(total / n_shards)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    counts: list[int] = []
    shard_keys: list[set] = []
    written_paths: list[Path] = []

    for k in range(n_shards):
        lo = k * per_shard
        hi = min((k + 1) * per_shard, total)
        chunk = rows[lo:hi]
        if not chunk:
            raise RuntimeError(
                f"shard_split: shard_{k} is empty (lo={lo}, hi={hi}, "
                f"total={total}, per_shard={per_shard}). This should not "
                f"happen after the n_shards<=total guard; aborting."
            )
        shard_dir = out_dir / f"shard_{k}"
        shard_dir.mkdir(parents=True, exist_ok=True)
        shard_file = shard_dir / f"shard_{k}.jsonl"
        keys: set = set()
        with shard_file.open("w", encoding="utf-8") as out:
            for offset, (raw_line, parsed) in enumerate(chunk):
                out.write(raw_line + "\n")
                keys.add(_row_key(parsed, lo + offset))
        counts.append(len(chunk))
        shard_keys.append(keys)
        written_paths.append(shard_file)

    # ------------------------------------------------------------------
    # HARD verification (a): line-count conservation.
    # ------------------------------------------------------------------
    if sum(counts) != total:
        raise RuntimeError(
            f"shard_split DISJOINTNESS FAILURE (count): sum of shard line "
            f"counts ({sum(counts)}) != input line count ({total}). "
            f"Per-shard counts: {counts}."
        )

    # ------------------------------------------------------------------
    # HARD verification (b): keys are disjoint across shards AND their
    # union equals the full key set.
    # ------------------------------------------------------------------
    full_keys: set = set()
    for k, (raw_line, parsed) in enumerate(rows):
        full_keys.add(_row_key(parsed, k))

    # Disjointness: total key count across shards (with multiplicity) must
    # equal the size of the union; any overlap inflates the multiplicity sum.
    keys_with_multiplicity = sum(len(s) for s in shard_keys)
    union_keys: set = set()
    for s in shard_keys:
        union_keys |= s

    if keys_with_multiplicity != len(union_keys):
        # Find an offending overlapping key for the message.
        seen: set = set()
        dupes: set = set()
        for s in shard_keys:
            for key in s:
                if key in seen:
                    dupes.add(key)
                seen.add(key)
        raise RuntimeError(
            f"shard_split DISJOINTNESS FAILURE (overlap): {len(dupes)} "
            f"key(s) appear in more than one shard, e.g. "
            f"{sorted(dupes)[:5]}. Shards are NOT disjoint; aborting."
        )

    if union_keys != full_keys:
        dropped = full_keys - union_keys
        extra = union_keys - full_keys
        raise RuntimeError(
            f"shard_split COMPLETENESS FAILURE: shard key union does not "
            f"equal the input key set. dropped={len(dropped)} "
            f"(e.g. {sorted(dropped)[:5]}), unexpected={len(extra)} "
            f"(e.g. {sorted(extra)[:5]}). aborting."
        )

    # NOTE on key collapse: if the corpus has duplicate stable keys (two
    # rows with the same _attempt_idx), len(full_keys) < total. The
    # count-conservation check (a) still guarantees every LINE is placed
    # exactly once; the union check guarantees no key crosses a shard
    # boundary. We surface the collapse so the operator is aware the corpus
    # is not key-unique (which would be a data-quality bug upstream).
    if len(full_keys) != total:
        raise RuntimeError(
            f"shard_split KEY-UNIQUENESS FAILURE: input has {total} rows but "
            f"only {len(full_keys)} distinct stable keys "
            f"(_attempt_idx/seed_idx/line_idx). Duplicate keys make the "
            f"merge ambiguous; deduplicate the corpus first. aborting."
        )

    return counts


def main(argv: "list[str] | None" = None) -> int:
    p = argparse.ArgumentParser(
        description="Split a self-traces JSONL into N contiguous, "
                    "non-overlapping shard files for data-parallel "
                    "calibration capture.",
    )
    p.add_argument("jsonl", type=str,
                   help="Path to the input self-traces JSONL.")
    p.add_argument("n_shards", type=int,
                   help="Number of shards (== number of GPUs / processes).")
    p.add_argument("--out-dir", type=str, default=None,
                   help="Output directory for shard_k/shard_k.jsonl dirs. "
                        "Default: <jsonl.parent>/parallel_capture.")
    args = p.parse_args(argv)

    jsonl_path = Path(args.jsonl).resolve()
    if not jsonl_path.is_file():
        print(f"shard_split: input file not found: {jsonl_path}",
              file=sys.stderr)
        return 1

    out_dir = (
        Path(args.out_dir).resolve()
        if args.out_dir
        else jsonl_path.parent / "parallel_capture"
    )

    try:
        counts = split_jsonl(jsonl_path, args.n_shards, out_dir)
    except (ValueError, RuntimeError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(f"shard_split: OK — {sum(counts)} rows -> {len(counts)} shards "
          f"under {out_dir}")
    for k, n in enumerate(counts):
        print(f"  shard_{k}/shard_{k}.jsonl : {n} rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())

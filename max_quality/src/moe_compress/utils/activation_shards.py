"""Disk-shard storage for per-layer heal activations.

The Stage-2 merge-heal step needs row-aligned ``(mlp_input, mlp_output, shared_output)``
triples per layer, sampled in minibatches over many steps. Holding the full pool
in RAM caps the pool at ~262K rows on H200 (≈ 12.9 GB per layer at fp32 on-device,
×3 tensors). To expand the pool ~100× we stream it from disk instead, one
safetensors shard at a time.

Lifecycle, one per layer:

1. ``ShardWriter.append(x_in, x_out)`` is called from the capture hook; rows are
   buffered to bf16 CPU memory and flushed as ``acts_layer{L}_shard{S:05d}.safetensors``
   files once the buffer reaches ``shard_rows``.
2. ``ShardWriter.compute_shared_companions(shared_fn)`` runs a second pass after
   capture: for each input shard it computes ``shared_fn(x_in)`` and writes a
   companion ``shared_layer{L}_shard{S:05d}.safetensors``. Decoupling shared
   from capture means the heavy shared-expert module doesn't have to be live
   during the (already memory-tight) corpus forward pass.
3. ``ShardWriter.finalize(split_ratio, seed)`` deterministically partitions the
   shard list into train/holdout (whole-shard split, no row-level mixing),
   writes ``manifest.json``, and returns the ``ShardManifest``.
4. ``HealActivationDataset(manifest)`` is then constructed by the heal loop;
   ``sample_minibatch(mb, generator)`` yields ``(xb, sb, tb)`` on device, in
   fp32, by loading ``K = ceil(mb / shard_rows)`` random train shards per call.
   ``iter_holdout(batch_size)`` does a deterministic full pass over holdout
   shards for patience-eval.

bf16 on disk, fp32 on consume — same dtype contract as the in-memory predecessor.
The dataset is corpus-agnostic: the manifest records the upstream
``CalibrationSpec.cache_key`` only for reproducibility, not for runtime dispatch.
"""
from __future__ import annotations

import json
import logging
import math
import shutil
from collections import OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterator

import torch
from safetensors.torch import safe_open, save_file

log = logging.getLogger(__name__)

_MANIFEST_FILENAME = "manifest.json"
_INPUT_KEY = "input"
_OUTPUT_KEY = "output"
_SHARED_KEY = "shared"
_INPUT_PREFIX = "acts_"
_SHARED_PREFIX = "shared_"


@dataclass(frozen=True)
class ShardEntry:
    """One on-disk activation shard. ``path`` is relative to the manifest dir."""
    path: str
    rows: int


@dataclass
class ShardManifest:
    """All shards for one layer. JSON-serializable.

    The ``corpus_spec_hash`` field is purely informational — it pins the manifest
    to the calibration spec it was captured under, so a stale manifest paired
    with a refreshed corpus is detectable. It is NOT used for dispatch.
    """
    layer_idx: int
    hidden_dim: int
    dtype: str
    corpus_spec_hash: str
    train_shards: list[ShardEntry]
    holdout_shards: list[ShardEntry]
    n_train: int
    n_holdout: int
    shard_rows: int
    schema_version: int = 1

    def to_json(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "layer_idx": self.layer_idx,
            "hidden_dim": self.hidden_dim,
            "dtype": self.dtype,
            "corpus_spec_hash": self.corpus_spec_hash,
            "shard_rows": self.shard_rows,
            "n_train": self.n_train,
            "n_holdout": self.n_holdout,
            "train_shards": [asdict(s) for s in self.train_shards],
            "holdout_shards": [asdict(s) for s in self.holdout_shards],
        }

    @classmethod
    def from_json(cls, payload: dict) -> "ShardManifest":
        if payload.get("schema_version") != 1:
            raise ValueError(
                f"ShardManifest schema_version {payload.get('schema_version')!r} "
                f"not supported (expected 1)"
            )
        train_shards = [ShardEntry(**s) for s in payload["train_shards"]]
        holdout_shards = [ShardEntry(**s) for s in payload["holdout_shards"]]
        # ShardEntry.path is contracted to be a bare filename (no directory
        # components). _shared_name_for relies on this when deriving the
        # companion shared-shard name; surface a clear error here so a
        # malformed or hand-edited manifest fails fast at load time rather
        # than deep inside HealActivationDataset.__init__.
        for entry in (*train_shards, *holdout_shards):
            if "/" in entry.path:
                raise ValueError(
                    f"ShardManifest: entry path {entry.path!r} contains a "
                    f"directory separator; ShardEntry.path must be a bare "
                    f"filename relative to the manifest directory."
                )
        return cls(
            layer_idx=int(payload["layer_idx"]),
            hidden_dim=int(payload["hidden_dim"]),
            dtype=str(payload["dtype"]),
            corpus_spec_hash=str(payload["corpus_spec_hash"]),
            shard_rows=int(payload["shard_rows"]),
            n_train=int(payload["n_train"]),
            n_holdout=int(payload["n_holdout"]),
            train_shards=train_shards,
            holdout_shards=holdout_shards,
        )


def _dtype_str(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


def _dtype_from_str(name: str) -> torch.dtype:
    table = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    if name not in table:
        raise ValueError(f"Unsupported shard dtype {name!r}; valid: {sorted(table)}")
    return table[name]


def _shared_name_for(input_name: str) -> str:
    if "/" in input_name:
        raise ValueError(
            f"Shard name {input_name!r} contains a directory separator; "
            f"ShardEntry.path is contracted to be a bare filename, not a path"
        )
    if not input_name.startswith(_INPUT_PREFIX):
        raise ValueError(
            f"Shard name {input_name!r} does not start with {_INPUT_PREFIX!r}; "
            f"cannot derive companion shared-shard name"
        )
    return _SHARED_PREFIX + input_name[len(_INPUT_PREFIX):]


class ShardWriter:
    """Buffers ``(x_in, x_out)`` rows and flushes them to safetensors shards.

    Append-only. Not thread-safe. One writer per (layer, run) pair.
    """

    def __init__(
        self,
        out_dir: str | Path,
        layer_idx: int,
        hidden_dim: int,
        *,
        shard_rows: int = 4096,
        dtype: torch.dtype = torch.bfloat16,
        corpus_spec_hash: str = "",
    ) -> None:
        """``out_dir`` is created lazily on the first shard flush (not in
        ``__init__``) — a writer that is constructed and immediately torn down
        without any ``append`` leaves no on-disk footprint."""
        if shard_rows <= 0:
            raise ValueError(f"shard_rows must be positive, got {shard_rows}")
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        self.out_dir = Path(out_dir)
        self.layer_idx = int(layer_idx)
        self.hidden_dim = int(hidden_dim)
        self.shard_rows = int(shard_rows)
        self.dtype = dtype
        self.corpus_spec_hash = corpus_spec_hash
        self._buf_in: list[torch.Tensor] = []
        self._buf_out: list[torch.Tensor] = []
        self._buf_rows: int = 0
        self._shards: list[ShardEntry] = []
        self._closed: bool = False

    @property
    def n_captured(self) -> int:
        """Rows written to disk plus rows currently buffered."""
        written = sum(s.rows for s in self._shards)
        return written + self._buf_rows

    @property
    def shard_entries(self) -> list[ShardEntry]:
        """Snapshot of shards written so far (not including buffered rows).

        Returned list is a shallow copy so callers can safely iterate without
        worrying about the writer mutating it underneath them.
        """
        return list(self._shards)

    def append(self, x_in: torch.Tensor, x_out: torch.Tensor) -> None:
        """Buffer one batch of rows; flush a shard when the buffer fills."""
        if self._closed:
            raise RuntimeError("ShardWriter is closed; cannot append")
        if x_in.ndim != 2 or x_in.size(1) != self.hidden_dim:
            raise ValueError(
                f"ShardWriter.append: expected x_in [N, {self.hidden_dim}], "
                f"got shape {tuple(x_in.shape)}"
            )
        if x_in.shape != x_out.shape:
            raise ValueError(
                f"ShardWriter.append: x_in shape {tuple(x_in.shape)} != x_out shape "
                f"{tuple(x_out.shape)} — capture hook returned misaligned rows"
            )
        if x_in.size(0) == 0:
            return
        # Move to CPU bf16 contiguous up front so the buffer dtype is uniform
        # and successive cats are cheap.
        self._buf_in.append(x_in.detach().to(self.dtype).cpu().contiguous())
        self._buf_out.append(x_out.detach().to(self.dtype).cpu().contiguous())
        self._buf_rows += int(x_in.size(0))
        while self._buf_rows >= self.shard_rows:
            self._flush_full_shard()

    def _flush_full_shard(self) -> None:
        cat_in = torch.cat(self._buf_in, dim=0) if len(self._buf_in) > 1 else self._buf_in[0]
        cat_out = torch.cat(self._buf_out, dim=0) if len(self._buf_out) > 1 else self._buf_out[0]
        chunk_in = cat_in[: self.shard_rows].contiguous()
        chunk_out = cat_out[: self.shard_rows].contiguous()
        self._write_input_shard(chunk_in, chunk_out)
        leftover_in = cat_in[self.shard_rows :]
        leftover_out = cat_out[self.shard_rows :]
        if leftover_in.size(0) > 0:
            self._buf_in = [leftover_in.contiguous()]
            self._buf_out = [leftover_out.contiguous()]
            self._buf_rows = int(leftover_in.size(0))
        else:
            self._buf_in = []
            self._buf_out = []
            self._buf_rows = 0

    def _write_input_shard(self, x_in: torch.Tensor, x_out: torch.Tensor) -> None:
        # Lazy out_dir creation: defer until we actually have a shard to flush.
        # A writer that's constructed and torn down with no appends leaves no
        # on-disk footprint.
        self.out_dir.mkdir(parents=True, exist_ok=True)
        idx = len(self._shards)
        name = f"{_INPUT_PREFIX}layer{self.layer_idx}_shard{idx:05d}.safetensors"
        path = self.out_dir / name
        # F-H-1: previously save_file(…, str(path)) wrote in place. A
        # SIGKILL mid-write left a truncated .safetensors at the final
        # name. atomic_safetensors_save does tmp+fsync+os.replace+
        # fsync(parent) so a torn write leaves the previous (good or
        # absent) final-path file untouched and a stale .tmp orphan.
        from .atomic_io import atomic_safetensors_save
        atomic_safetensors_save(path, {_INPUT_KEY: x_in, _OUTPUT_KEY: x_out})
        self._shards.append(ShardEntry(path=name, rows=int(x_in.size(0))))

    def close_pending(self) -> None:
        """Flush any leftover buffered rows as a final (possibly short) shard."""
        if self._buf_rows == 0:
            return
        cat_in = torch.cat(self._buf_in, dim=0).contiguous()
        cat_out = torch.cat(self._buf_out, dim=0).contiguous()
        self._write_input_shard(cat_in, cat_out)
        self._buf_in = []
        self._buf_out = []
        self._buf_rows = 0

    def compute_shared_companions(
        self,
        shared_fn: Callable[[torch.Tensor], torch.Tensor],
        *,
        read_dtype: torch.dtype = torch.float32,
    ) -> None:
        """Second pass: write companion ``shared_*`` shards.

        For each captured input shard, loads it from disk, runs ``shared_fn`` on
        it, and writes the result as ``shared_layer{L}_shard{S:05d}.safetensors``
        with key ``"shared"``. ``shared_fn`` is expected to handle its own
        device placement; it receives an ``fp32`` CPU tensor and may return any
        device/dtype tensor (we cast to the writer dtype before saving).
        """
        if self._closed:
            raise RuntimeError(
                "ShardWriter is closed; cannot compute_shared_companions. "
                "Lifecycle is: append* -> compute_shared_companions -> finalize. "
                "After finalize() the writer is closed and no further state-"
                "mutating calls are permitted."
            )
        if self._buf_rows != 0:
            self.close_pending()
        for entry in self._shards:
            in_path = self.out_dir / entry.path
            with safe_open(str(in_path), framework="pt", device="cpu") as f:
                x_in = f.get_tensor(_INPUT_KEY).to(read_dtype)
            x_shared = shared_fn(x_in)
            x_shared = x_shared.detach().to(self.dtype).cpu().contiguous()
            if x_shared.shape != (entry.rows, self.hidden_dim):
                raise ValueError(
                    f"compute_shared_companions: shared_fn returned shape "
                    f"{tuple(x_shared.shape)} for shard with {entry.rows} rows × "
                    f"{self.hidden_dim} hidden; shape mismatch"
                )
            shared_name = _shared_name_for(entry.path)
            shared_path = self.out_dir / shared_name
            # F-H-1 (companion): same atomic-write protection as the
            # input shard above. See _write_input_shard rationale.
            from .atomic_io import atomic_safetensors_save
            atomic_safetensors_save(shared_path, {_SHARED_KEY: x_shared})

    def finalize(self, *, split_ratio: float = 0.9, seed: int = 0) -> ShardManifest:
        """Whole-shard train/holdout split, write ``manifest.json``, return it.

        The split is deterministic in ``seed``: a permutation of the shard list
        is taken, and the first ``floor(split_ratio * N)`` shards become train,
        the rest holdout. Whole-shard granularity avoids reshuffling rows
        across shards, which would invalidate any per-shard mmap optimisation
        downstream readers might add.
        """
        if self._closed:
            raise RuntimeError(
                "ShardWriter is closed; cannot finalize again. "
                "Lifecycle is: append* -> compute_shared_companions -> finalize. "
                "After finalize() the writer is closed and no further state-"
                "mutating calls are permitted."
            )
        if self._buf_rows != 0:
            self.close_pending()
        if not (0.0 < split_ratio < 1.0):
            raise ValueError(f"split_ratio must be in (0, 1), got {split_ratio}")
        if not self._shards:
            raise RuntimeError("ShardWriter.finalize: no shards were written")
        n = len(self._shards)
        if n < 2:
            plural = "" if n == 1 else "s"
            raise RuntimeError(
                f"ShardWriter.finalize: only {n} shard{plural} written — whole-shard "
                f"train/holdout split requires at least 2. Lower ``shard_rows`` "
                f"or capture more rows. (Captured rows: {self._shards[0].rows}, "
                f"shard_rows: {self.shard_rows}.)"
            )
        gen = torch.Generator().manual_seed(int(seed))
        perm = torch.randperm(n, generator=gen).tolist()
        n_train = max(1, math.floor(split_ratio * n))
        # If split_ratio is close to 1.0 and n is small we could end up with
        # n_train == n; guarantee at least one holdout shard.
        if n_train >= n:
            n_train = n - 1
        train_idx = perm[:n_train]
        holdout_idx = perm[n_train:]
        train_shards = [self._shards[i] for i in train_idx]
        holdout_shards = [self._shards[i] for i in holdout_idx]
        manifest = ShardManifest(
            layer_idx=self.layer_idx,
            hidden_dim=self.hidden_dim,
            dtype=_dtype_str(self.dtype),
            corpus_spec_hash=self.corpus_spec_hash,
            train_shards=train_shards,
            holdout_shards=holdout_shards,
            n_train=sum(s.rows for s in train_shards),
            n_holdout=sum(s.rows for s in holdout_shards),
            shard_rows=self.shard_rows,
        )
        path = self.out_dir / _MANIFEST_FILENAME
        # F-H-2: previously path.write_text(json.dumps(...)) opened-
        # truncated-wrote in place. A SIGKILL mid-flush could leave a
        # JSON file truncated at a "happens-to-be-valid" structural
        # boundary — silently accepted by json.loads but with a
        # truncated shard list → heal trained on a fraction of the
        # captured data without noticing.
        #
        # atomic_json_save does tmp + fsync(fd) + os.replace +
        # fsync(parent_dir) so the manifest is the LAST thing written
        # by finalize(); a torn manifest = no manifest at all,
        # detectable by readers (they require manifest.json to exist).
        # Combined with F-H-1's per-shard atomic writes, the
        # manifest-last invariant for the heal-shards directory is
        # now durable end-to-end (Pattern O).
        from .atomic_io import atomic_json_save
        atomic_json_save(
            path, manifest.to_json(), indent=2, sort_keys=True,
        )
        log.info(
            "Wrote manifest %s: layer=%d n_train=%d n_holdout=%d shards=%d+%d",
            path, manifest.layer_idx, manifest.n_train, manifest.n_holdout,
            len(train_shards), len(holdout_shards),
        )
        self._closed = True
        return manifest

    def cleanup(self) -> None:
        """Delete the shard directory and everything in it.

        Safe to call repeatedly; no-ops if the dir is already gone. Also safe
        when ``out_dir`` was never created (the dir is created lazily on first
        flush — a writer torn down with no appends has no dir to remove).
        """
        if self.out_dir.exists():
            shutil.rmtree(self.out_dir)
        self._shards = []
        self._buf_in = []
        self._buf_out = []
        self._buf_rows = 0


def load_manifest(manifest_dir: str | Path) -> ShardManifest:
    """Read ``<manifest_dir>/manifest.json``."""
    path = Path(manifest_dir) / _MANIFEST_FILENAME
    payload = json.loads(path.read_text())
    return ShardManifest.from_json(payload)


class HealActivationDataset:
    """Streams ``(x_in, x_shared, x_out)`` minibatches from sharded files.

    Maintains a small LRU cache of decoded shards on device so successive
    sampling calls that happen to pick the same shard don't re-read it.
    """

    def __init__(
        self,
        manifest: ShardManifest,
        manifest_dir: str | Path,
        device: torch.device,
        *,
        shard_cache_size: int = 2,
        compute_dtype: torch.dtype = torch.float32,
    ) -> None:
        if shard_cache_size < 1:
            raise ValueError(f"shard_cache_size must be >= 1, got {shard_cache_size}")
        self.manifest = manifest
        self.manifest_dir = Path(manifest_dir)
        self.device = device
        self.compute_dtype = compute_dtype
        self._cache_size = int(shard_cache_size)
        self._cache: "OrderedDict[tuple[bool, int], tuple[torch.Tensor, torch.Tensor, torch.Tensor]]" = OrderedDict()
        # Verify companion `shared_*` shards exist for every input shard.  If
        # `compute_shared_companions` was skipped (or crashed mid-way) the
        # first `sample_minibatch` would otherwise raise a deep
        # FileNotFoundError from inside the heal loop — surface it at
        # construction time with a clear message instead.
        missing: list[str] = []
        for entry in (
            list(self.manifest.train_shards) + list(self.manifest.holdout_shards)
        ):
            shared_path = self.manifest_dir / _shared_name_for(entry.path)
            if not shared_path.exists():
                missing.append(str(shared_path))
        if missing:
            preview = missing[:5]
            more = f" (+{len(missing) - 5} more)" if len(missing) > 5 else ""
            raise RuntimeError(
                "HealActivationDataset: missing companion shared shard(s): "
                f"{preview}{more}. Call ShardWriter.compute_shared_companions(...) "
                "before constructing the dataset."
            )

    @property
    def n_train(self) -> int:
        return self.manifest.n_train

    @property
    def n_holdout(self) -> int:
        return self.manifest.n_holdout

    @property
    def hidden_dim(self) -> int:
        return self.manifest.hidden_dim

    def _load_shard(
        self, idx: int, *, train: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        key = (train, idx)
        cached = self._cache.get(key)
        if cached is not None:
            self._cache.move_to_end(key)
            return cached
        shard_list = self.manifest.train_shards if train else self.manifest.holdout_shards
        entry = shard_list[idx]
        in_path = self.manifest_dir / entry.path
        shared_path = self.manifest_dir / _shared_name_for(entry.path)
        with safe_open(str(in_path), framework="pt", device="cpu") as f:
            x_in_cpu = f.get_tensor(_INPUT_KEY)
            x_out_cpu = f.get_tensor(_OUTPUT_KEY)
        with safe_open(str(shared_path), framework="pt", device="cpu") as f:
            x_shared_cpu = f.get_tensor(_SHARED_KEY)
        x_in = x_in_cpu.to(self.compute_dtype).to(self.device)
        x_out = x_out_cpu.to(self.compute_dtype).to(self.device)
        x_shared = x_shared_cpu.to(self.compute_dtype).to(self.device)
        self._cache[key] = (x_in, x_shared, x_out)
        while len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return x_in, x_shared, x_out

    def sample_minibatch(
        self, mb: int, generator: torch.Generator,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Pick K random train shards, concat, slice mb random rows."""
        n_shards = len(self.manifest.train_shards)
        if n_shards == 0:
            raise RuntimeError("HealActivationDataset.sample_minibatch: no train shards")
        if mb <= 0:
            raise ValueError(f"mb must be positive, got {mb}")
        # Average rows per shard guides how many shards we need to fetch to
        # have at least mb rows in hand. Use the actual minimum to be safe
        # against short final shards.
        min_rows = min(s.rows for s in self.manifest.train_shards)
        k = max(1, math.ceil(mb / max(min_rows, 1)))
        k = min(k, n_shards)
        idx_tensor = torch.randperm(n_shards, generator=generator)[:k]
        # Separate generator for row selection so the row draw doesn't
        # advance the caller's generator state past the shard draw — keeps
        # the two draws independent without making the API harder to use.
        # IMPORTANT: seed `row_gen` from a FRESH draw on the (already
        # advanced) `generator`, NOT from `generator.initial_seed()`.
        # Using initial_seed gives the same row order every step, which
        # destroys within-shard row diversity across the training loop
        # (two consecutive steps that land on the same shard return the
        # same rows). Drawing a fresh seed advances `generator` and gives
        # `row_gen` a new starting state per call.
        row_seed = int(
            torch.randint(0, 2**63 - 1, (1,), generator=generator).item()
        )
        row_gen = torch.Generator().manual_seed(row_seed)
        xs_in, xs_shared, xs_out = [], [], []
        for i in idx_tensor.tolist():
            xi, xs, xo = self._load_shard(int(i), train=True)
            xs_in.append(xi)
            xs_shared.append(xs)
            xs_out.append(xo)
        cat_in = torch.cat(xs_in, dim=0) if len(xs_in) > 1 else xs_in[0]
        cat_shared = torch.cat(xs_shared, dim=0) if len(xs_shared) > 1 else xs_shared[0]
        cat_out = torch.cat(xs_out, dim=0) if len(xs_out) > 1 else xs_out[0]
        total = int(cat_in.size(0))
        if total < mb:
            # Fail loud rather than silently returning an under-sized batch.
            # Upsampling would hide the upstream config bug (token_cap /
            # shard_rows too small) from the caller's batch-size invariants.
            raise RuntimeError(
                f"HealActivationDataset.sample_minibatch: requested mb={mb} but "
                f"only {total} rows available after drawing {k} of {n_shards} "
                f"train shards. Capture more rows or lower the minibatch size."
            )
        if total == mb:
            return cat_in, cat_shared, cat_out
        sel = torch.randperm(total, generator=row_gen)[:mb]
        return cat_in.index_select(0, sel), cat_shared.index_select(0, sel), cat_out.index_select(0, sel)

    def iter_holdout(
        self, batch_size: int,
    ) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Deterministic full pass over holdout shards in manifest order."""
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        for i in range(len(self.manifest.holdout_shards)):
            x_in, x_shared, x_out = self._load_shard(i, train=False)
            n = int(x_in.size(0))
            for s in range(0, n, batch_size):
                e = min(s + batch_size, n)
                yield x_in[s:e], x_shared[s:e], x_out[s:e]

    def iter_all(
        self, batch_size: int,
    ) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Deterministic full pass over train shards followed by holdout shards.

        Used by the cross-domain telemetry path: the XD manifest's train/
        holdout split is structural only (the heal never trains on these rows),
        so telemetry should see the full pool, not just the 10% holdout slice.
        """
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        for i in range(len(self.manifest.train_shards)):
            x_in, x_shared, x_out = self._load_shard(i, train=True)
            n = int(x_in.size(0))
            for s in range(0, n, batch_size):
                e = min(s + batch_size, n)
                yield x_in[s:e], x_shared[s:e], x_out[s:e]
        for i in range(len(self.manifest.holdout_shards)):
            x_in, x_shared, x_out = self._load_shard(i, train=False)
            n = int(x_in.size(0))
            for s in range(0, n, batch_size):
                e = min(s + batch_size, n)
                yield x_in[s:e], x_shared[s:e], x_out[s:e]

    def clear_cache(self) -> None:
        self._cache.clear()

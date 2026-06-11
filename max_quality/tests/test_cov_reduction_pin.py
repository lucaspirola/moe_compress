import torch, pytest
from moe_compress.utils.activation_hooks import InputCovarianceAccumulator

def _gram(acc, key):  # drain the pending GPU/CPU sum for a key
    return acc._pending[key].clone()

def test_grouped_equals_sequential_bytewise():
    torch.manual_seed(0)
    d = 8
    # 3 sequences of differing token counts routed to expert e
    rows = [torch.randn(n, d, dtype=torch.float32) for n in (5, 3, 4)]
    seq_ids = torch.cat([torch.full((n,), s) for s, n in enumerate((5, 3, 4))])
    x = torch.cat(rows, 0)
    key = (0, 0, "gate_proj")

    # reference: feed each sequence as its own update, in ascending seq order
    ref = InputCovarianceAccumulator(); ref.set_storage_dtype(torch.float32)
    for r in rows:
        ref.update(0, 0, "gate_proj", r)

    # pinned grouped: one call with a merged block + seq_ids
    got = InputCovarianceAccumulator(); got.set_storage_dtype(torch.float32)
    got.update_grouped(0, 0, "gate_proj", x, seq_ids)

    assert torch.equal(_gram(got, key), _gram(ref, key))   # BYTE-identical

def test_grouped_is_order_invariant_in_input_block_but_pins_seq_grouping():
    # shuffling the row order WITHIN the merged block (keeping seq_ids aligned)
    # must NOT change the result (each per-seq matmul gets the same row set in
    # the same per-seq relative order via boolean select).
    torch.manual_seed(1); d = 6
    x = torch.randn(12, d, dtype=torch.float32)
    seq_ids = torch.tensor([0,0,1,2,2,2,0,1,1,2,0,1])
    a = InputCovarianceAccumulator(); a.set_storage_dtype(torch.float32)
    a.update_grouped(0,0,"gate_proj", x, seq_ids)
    # a stable-by-seq reference: gather rows per ascending seq in original order
    ref = InputCovarianceAccumulator(); ref.set_storage_dtype(torch.float32)
    for s in sorted(set(seq_ids.tolist())):
        ref.update(0,0,"gate_proj", x[seq_ids == s])
    assert torch.equal(a._pending[(0,0,"gate_proj")], ref._pending[(0,0,"gate_proj")])

def test_update_grouped_single_sequence_equals_plain_update():
    torch.manual_seed(2); d = 4
    x = torch.randn(7, d, dtype=torch.float32)
    seq_ids = torch.zeros(7, dtype=torch.long)
    g = InputCovarianceAccumulator(); g.set_storage_dtype(torch.float32)
    g.update_grouped(0,0,"down_proj", x, seq_ids)
    p = InputCovarianceAccumulator(); p.set_storage_dtype(torch.float32)
    p.update(0,0,"down_proj", x)
    assert torch.equal(g._pending[(0,0,"down_proj")], p._pending[(0,0,"down_proj")])


# ---------------------------------------------------------------------------
# Task 2: callback routing (forward-free). These mirror the EXACT routing the
# cov callbacks in covariance_collection.py perform, so they pin the contracts:
#   * B-path (down_proj): split on the UNPADDED prefix tensor[:tok.shape[0]];
#     trailing zero pad rows make drop-vs-keep byte-equal.
#   * C-path (cross-cov): split the KEPT pre-matmul operands by sel_idx//seq_len
#     (NOT tok//seq_len), accumulating ascending via update_cross.
#   * No-_seq_len / single-seq → plain update/update_cross, byte-identical.
# ---------------------------------------------------------------------------


def _route_b(acc, li, e, name, tensor, ctx, seq_len):
    """Replica of the B-path routing in input_cb / intermediate_cb."""
    tok = ctx.get("token_idx")
    if seq_len and tok is not None:
        rows = tensor[: tok.shape[0]]                 # unpadded prefix (down_proj is padded)
        acc.update_grouped(li, e, name, rows, tok // seq_len)
    else:
        acc.update(li, e, name, tensor)


def _route_c(acc, li, e, name, X_pre, X_post, sel_idx, n_tokens, seq_len):
    """Replica of the C-path operand split in input_cb. sel_idx are the KEPT-row
    token indices (1:1 with X_pre/X_post rows)."""
    if seq_len:
        sids = sel_idx // seq_len
        if torch.unique(sids).numel() > 1:
            for s in torch.unique(sids, sorted=True).tolist():
                m = sids == s
                acc.update_cross(li, e, name, X_pre[m].T @ X_post[m], int(m.sum()))
            return
    acc.update_cross(li, e, name, X_pre.T @ X_post, n_tokens)


def test_factored_padded_down_proj_split_drops_zero_pad_bytewise():
    # Factored down_proj: tensor is [max_tokens, d] with trailing ZERO pad rows;
    # tok length < max_tokens, spanning >=2 sequences. Routed split must equal
    # update_grouped on the unpadded prefix, AND keeping the zero pad rows must
    # be byte-identical to dropping them (silu(0)*0 = 0 contributes nothing).
    torch.manual_seed(3)
    d, seq_len = 8, 10
    # 2 sequences (ids 0 and 1) of token counts 4 and 3 → 7 real tokens.
    tok = torch.tensor([1, 3, 5, 7, 11, 13, 15], dtype=torch.long)  # //10 → [0,0,0,0,1,1,1]
    n_real = tok.shape[0]
    max_tokens = 12                                   # padded length > n_real
    real = torch.randn(n_real, d, dtype=torch.float32)
    tensor = torch.zeros(max_tokens, d, dtype=torch.float32)
    tensor[:n_real] = real                            # trailing rows stay ZERO
    ctx = {"token_idx": tok}
    li, e, name = 0, 0, "down_proj"

    got = InputCovarianceAccumulator(); got.set_storage_dtype(torch.float32)
    _route_b(got, li, e, name, tensor, ctx, seq_len)

    # Reference: explicit update_grouped on the unpadded prefix.
    ref = InputCovarianceAccumulator(); ref.set_storage_dtype(torch.float32)
    ref.update_grouped(li, e, name, tensor[:n_real], tok // seq_len)
    assert torch.equal(got._pending[(li, e, name)], ref._pending[(li, e, name)])

    # Drop-vs-keep: routing on the FULL padded tensor (prefix slice owns the drop)
    # equals splitting the real rows directly — pad rows are zero → byte-equal.
    keep_full = InputCovarianceAccumulator(); keep_full.set_storage_dtype(torch.float32)
    keep_full.update_grouped(li, e, name, real, tok // seq_len)
    assert torch.equal(got._pending[(li, e, name)], keep_full._pending[(li, e, name)])


def test_c_operand_split_on_kept_rows_with_drop():
    # C-path: a keep mask drops >=1 row, so sel_idx = tok[keep] is shorter than
    # tok. The per-seq operand split must use sids = sel_idx // seq_len (1:1 with
    # the kept X_pre/X_post rows). Deriving sids from the full tok would mis-length.
    torch.manual_seed(4)
    d, seq_len = 6, 10
    # 6 candidate tokens over 2 sequences; drop two (positions 1 and 4).
    tok = torch.tensor([2, 4, 6, 12, 14, 16], dtype=torch.long)   # //10 → [0,0,0,1,1,1]
    keep = torch.tensor([True, False, True, True, False, True])
    sel_idx = tok[keep]                               # → [2, 6, 12, 16] → sids [0,0,1,1]
    n_kept = int(keep.sum())
    # Full teacher/student rows; operands are the KEPT rows only.
    X_pre = torch.randn(n_kept, d, dtype=torch.float32)
    X_post = torch.randn(n_kept, d, dtype=torch.float32)
    li, e, name = 0, 0, "gate_proj"

    got = InputCovarianceAccumulator(); got.set_storage_dtype(torch.float32)
    _route_c(got, li, e, name, X_pre, X_post, sel_idx, n_kept, seq_len)

    # Reference: ascending per-seq operand cross fed to update_cross directly.
    ref = InputCovarianceAccumulator(); ref.set_storage_dtype(torch.float32)
    sids = sel_idx // seq_len
    for s in sorted(set(sids.tolist())):
        m = sids == s
        ref.update_cross(li, e, name, X_pre[m].T @ X_post[m], int(m.sum()))
    assert torch.equal(got._pending[(li, e, name)], ref._pending[(li, e, name)])
    assert got._gpu_token_count[(li, e, name)] == n_kept   # token count preserved

    # The full-tok sids (the BUGGY source) has len(tok)=6 != X_pre rows=4 → a
    # boolean mask from it would mis-length the operands. Pin that it is wrong.
    bad_sids = tok // seq_len
    assert bad_sids.shape[0] != X_pre.shape[0]


def test_c_single_sequence_equals_plain_update_cross():
    # All kept rows in ONE sequence → torch.unique(sids)==1 → plain update_cross,
    # byte-identical to the bs=1 path.
    torch.manual_seed(5)
    d, seq_len = 5, 10
    tok = torch.tensor([1, 3, 5, 7], dtype=torch.long)   # all //10 → seq 0
    keep = torch.tensor([True, True, False, True])
    sel_idx = tok[keep]
    n_kept = int(keep.sum())
    X_pre = torch.randn(n_kept, d, dtype=torch.float32)
    X_post = torch.randn(n_kept, d, dtype=torch.float32)
    li, e, name = 0, 0, "gate_proj"

    got = InputCovarianceAccumulator(); got.set_storage_dtype(torch.float32)
    _route_c(got, li, e, name, X_pre, X_post, sel_idx, n_kept, seq_len)
    ref = InputCovarianceAccumulator(); ref.set_storage_dtype(torch.float32)
    ref.update_cross(li, e, name, X_pre.T @ X_post, n_kept)
    assert torch.equal(got._pending[(li, e, name)], ref._pending[(li, e, name)])


def test_no_seq_len_routes_plain_update_and_update_cross():
    # seq_len == 0 / None context → bare update / update_cross, byte-identical.
    torch.manual_seed(6)
    d = 4
    x = torch.randn(9, d, dtype=torch.float32)
    ctx = {"token_idx": torch.arange(9)}
    li, e = 0, 0

    g = InputCovarianceAccumulator(); g.set_storage_dtype(torch.float32)
    _route_b(g, li, e, "down_proj", x, ctx, 0)        # seq_len falsy → plain update
    p = InputCovarianceAccumulator(); p.set_storage_dtype(torch.float32)
    p.update(li, e, "down_proj", x)
    assert torch.equal(g._pending[(li, e, "down_proj")], p._pending[(li, e, "down_proj")])

    X_pre = torch.randn(9, d, dtype=torch.float32)
    X_post = torch.randn(9, d, dtype=torch.float32)
    sel_idx = torch.arange(9)
    gc = InputCovarianceAccumulator(); gc.set_storage_dtype(torch.float32)
    _route_c(gc, li, e, "gate_proj", X_pre, X_post, sel_idx, 9, 0)
    pc = InputCovarianceAccumulator(); pc.set_storage_dtype(torch.float32)
    pc.update_cross(li, e, "gate_proj", X_pre.T @ X_post, 9)
    assert torch.equal(gc._pending[(li, e, "gate_proj")], pc._pending[(li, e, "gate_proj")])

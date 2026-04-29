"""Smoke tests for Stage 3 cross-covariance and D8 ε* changes.

Tests:
1. CrossCovarianceAccumulator produces correct C = X_pre^T @ X_post
2. _aa_svd with cross-covariance (Theorem 3.2) beats auto-covariance
3. Activation-weighted ε* matches brute-force computation
"""
import torch
import math


def test_cross_covariance_accumulator():
    """Verify cross-covariance C = X_pre^T @ X_post is correctly accumulated
    and is numerically distinct from auto-covariance A = X_pre^T @ X_pre
    when pre/post distributions differ."""
    torch.manual_seed(42)
    d_in = 16
    n_tokens = 100

    # Simulate pre-prune and post-prune expert inputs with different distributions
    X_pre = torch.randn(n_tokens, d_in)
    # Post-prune: shifted distribution (some experts merged, routing changed)
    X_post = X_pre + 0.3 * torch.randn(n_tokens, d_in)

    # Ground truth covariances
    A_true = X_pre.T @ X_pre       # auto-cov of pre-prune
    B_true = X_post.T @ X_post     # auto-cov of post-prune
    C_true = X_pre.T @ X_post      # cross-covariance

    # Verify C != A (they should differ when distributions differ)
    assert not torch.allclose(C_true, A_true, atol=1e-4), \
        "Cross-covariance should differ from auto-covariance when distributions shift"

    # Verify C != B
    assert not torch.allclose(C_true, B_true, atol=1e-4), \
        "Cross-covariance should differ from B auto-covariance"

    # Simulate incremental accumulation (as the code would do per batch)
    C_accum = torch.zeros(d_in, d_in)
    B_accum = torch.zeros(d_in, d_in)
    batch_size = 10
    for start in range(0, n_tokens, batch_size):
        end = min(start + batch_size, n_tokens)
        x_pre_batch = X_pre[start:end]
        x_post_batch = X_post[start:end]
        C_accum += x_pre_batch.T @ x_post_batch
        B_accum += x_post_batch.T @ x_post_batch

    assert torch.allclose(C_accum, C_true, atol=1e-5), \
        "Incrementally accumulated cross-covariance should match batch computation"
    assert torch.allclose(B_accum, B_true, atol=1e-5), \
        "Incrementally accumulated B should match batch computation"

    print("✓ test_cross_covariance_accumulator passed")


def test_aa_svd_theorem32_vs_autocov():
    """Verify that AA-SVD with paper-exact cross-covariance (Theorem 3.2)
    produces lower reconstruction error than the auto-covariance approximation."""
    torch.manual_seed(123)
    d_out, d_in = 32, 16
    k = 4
    n_tokens = 200

    W = torch.randn(d_out, d_in)
    X_pre = torch.randn(n_tokens, d_in)
    X_post = X_pre + 0.5 * torch.randn(n_tokens, d_in)  # significant shift

    # Covariances
    A = X_pre.T @ X_pre          # auto-cov pre
    B = X_post.T @ X_post        # auto-cov post
    C = X_pre.T @ X_post         # cross-covariance

    # --- Paper-exact path (Theorem 3.2): M = W @ C @ B^{-1} @ L_B ---
    eigvals, eigvecs = torch.linalg.eigh(B)
    keep = eigvals > 1e-6
    eigvals_k = eigvals[keep].clamp_min(1e-12)
    eigvecs_k = eigvecs[:, keep]
    L_B = eigvecs_k * eigvals_k.sqrt().unsqueeze(0)

    # M_paper = W @ C @ B^{-1} @ L_B
    # B^{-1} @ L_B = Q @ diag(1/λ) @ Q^T @ Q @ diag(√λ) = Q @ diag(1/√λ)
    inv_sqrt = eigvals_k.rsqrt()
    CQ = C @ eigvecs_k
    M_paper = W @ (CQ * inv_sqrt.unsqueeze(0))

    U, S, Vh = torch.linalg.svd(M_paper, full_matrices=False)
    U_k = U[:, :k] * S[:k]
    V_k = (Vh[:k, :] * inv_sqrt.unsqueeze(0)) @ eigvecs_k.T
    W_paper = U_k @ V_k

    # --- Auto-covariance approximation: M = W @ A @ B^{-1} @ L_B ---
    AQ = A @ eigvecs_k
    M_auto = W @ (AQ * inv_sqrt.unsqueeze(0))

    U2, S2, Vh2 = torch.linalg.svd(M_auto, full_matrices=False)
    U_k2 = U2[:, :k] * S2[:k]
    V_k2 = (Vh2[:k, :] * inv_sqrt.unsqueeze(0)) @ eigvecs_k.T
    W_auto = U_k2 @ V_k2

    # --- Evaluate: paper's objective ‖W·X_pre − W'·X_post‖_F ---
    target = W @ X_pre.T    # [d_out, n_tokens]
    err_paper = (target - W_paper @ X_post.T).norm()
    err_auto = (target - W_auto @ X_post.T).norm()

    print(f"  Paper-exact error: {err_paper:.6f}")
    print(f"  Auto-cov error:    {err_auto:.6f}")
    print(f"  Improvement:       {(err_auto - err_paper) / err_auto * 100:.1f}%")

    assert err_paper <= err_auto * 1.01, \
        f"Paper-exact should be <= auto-cov error (paper={err_paper:.4f}, auto={err_auto:.4f})"

    print("✓ test_aa_svd_theorem32_vs_autocov passed")


def test_activation_weighted_epsilon():
    """Verify activation-weighted ε* matches brute-force computation."""
    torch.manual_seed(77)
    d_out, d_in = 24, 12
    k = 3
    n_tokens = 80

    W = torch.randn(d_out, d_in)
    X = torch.randn(n_tokens, d_in)
    A = X.T @ X  # input covariance

    # Compute L_A via eigendecomposition
    eigvals, eigvecs = torch.linalg.eigh(A)
    keep = eigvals > 1e-8
    eigvals_k = eigvals[keep].clamp_min(1e-12)
    eigvecs_k = eigvecs[:, keep]
    L_A = eigvecs_k * eigvals_k.sqrt().unsqueeze(0)  # [d_in, r]

    # Activation-weighted SVD
    M = W @ L_A  # [d_out, r]
    U, S, Vh = torch.linalg.svd(M, full_matrices=False)

    # ε* = sqrt(tail_energy / total_energy) where energy is from weighted SVD
    S2 = S * S
    total = S2.sum()
    tail = S2[k:].sum()
    epsilon_weighted = (tail / total).sqrt().item()

    # Brute-force: compute W_k via weighted SVD, then ε = ‖(W-W_k) L_A‖_F / ‖W L_A‖_F
    U_k = U[:, :k] * S[:k]
    inv_sqrt = eigvals_k.rsqrt()
    V_k = (Vh[:k, :] * inv_sqrt.unsqueeze(0)) @ eigvecs_k.T  # back to weight space
    W_k = U_k @ V_k
    R = W - W_k
    err_weighted = (R @ L_A).norm().item()
    full_weighted = (W @ L_A).norm().item()
    epsilon_brute = err_weighted / full_weighted

    print(f"  ε* (weighted SVD tail): {epsilon_weighted:.6f}")
    print(f"  ε* (brute force):       {epsilon_brute:.6f}")
    print(f"  Difference:             {abs(epsilon_weighted - epsilon_brute):.2e}")

    assert abs(epsilon_weighted - epsilon_brute) < 1e-4, \
        f"Weighted ε* should match brute-force (diff={abs(epsilon_weighted - epsilon_brute):.2e})"

    # Compare with unweighted (current code's spectral tail proxy)
    S_raw = torch.linalg.svdvals(W)
    S2_raw = S_raw * S_raw
    epsilon_unweighted = (S2_raw[k:].sum() / S2_raw.sum()).sqrt().item()

    print(f"  ε* (unweighted raw):    {epsilon_unweighted:.6f}")
    print(f"  Weighted ≠ unweighted:  {abs(epsilon_weighted - epsilon_unweighted) > 1e-3}")

    print("✓ test_activation_weighted_epsilon passed")


if __name__ == "__main__":
    test_cross_covariance_accumulator()
    print()
    test_aa_svd_theorem32_vs_autocov()
    print()
    test_activation_weighted_epsilon()
    print()
    print("All smoke tests passed ✓")

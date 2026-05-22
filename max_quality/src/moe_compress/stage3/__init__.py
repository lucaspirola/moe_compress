"""Stage 3 — Non-uniform SVD factorization (plugin architecture).

Scaffold stage (S3-1): run delegates to the legacy stage3_svd.run monolith.
Plugin extraction lands in S3-2..S3-6; the real orchestrator + monolith
deletion land in S3-7.
"""
from .orchestrator import run

__all__ = ["run"]

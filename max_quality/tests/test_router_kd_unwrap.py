"""Task 2 — single ``unwrap_student`` helper.

Peels DDP (``.module``) and torch.compile (``._orig_mod``) wrappers in any
nesting order, idempotent on a bare module.
"""
from __future__ import annotations

import torch.nn as nn

from moe_compress.router_kd._unwrap import unwrap_student


class _Wrap:
    def __init__(self, inner, attr):
        setattr(self, attr, inner)


def test_unwrap_plain():
    m = nn.Linear(2, 2)
    assert unwrap_student(m) is m


def test_unwrap_compile_only():
    m = nn.Linear(2, 2)
    w = _Wrap(m, "_orig_mod")
    assert unwrap_student(w) is m


def test_unwrap_ddp_only():
    m = nn.Linear(2, 2)
    w = _Wrap(m, "module")
    assert unwrap_student(w) is m


def test_unwrap_ddp_over_compile():
    # DDP(compile(m)): outer has .module → compile wrapper, which has _orig_mod → m
    m = nn.Linear(2, 2)
    compiled = _Wrap(m, "_orig_mod")
    ddp = _Wrap(compiled, "module")
    assert unwrap_student(ddp) is m


def test_unwrap_compile_over_ddp():
    # compile(DDP(m)): outer _orig_mod → DDP wrapper, which has .module → m
    m = nn.Linear(2, 2)
    ddp = _Wrap(m, "module")
    compiled = _Wrap(ddp, "_orig_mod")
    assert unwrap_student(compiled) is m

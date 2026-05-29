"""Strace-based syscall observer for Pattern O fsync ordering tests.

NOTE on fsync predicates: strace records ``fsync(N)`` with a bare numeric
file descriptor — there is NO path argument to match against. The
canonical Pattern O trace is::

    openat(..., "/.../target.tmp",  O_RDONLY|O_CLOEXEC) = 7
    fsync(7) = 0
    rename("/.../target.tmp", "/.../target") = 0
    openat(..., "/.../parent_dir",  O_RDONLY|O_DIRECTORY|...) = 7
    fsync(7) = 0

so the file vs. parent-dir fsync is disambiguated by the most-recent
preceding ``openat`` (path-bearing) — NOT by anything in the fsync's
own args. After the production-side ``_fsync_dir`` change (commit #2
of the H-1/H-2 patch) the parent-dir openat decodes as
``O_RDONLY|O_DIRECTORY|O_CLOEXEC`` and the ``'O_DIRECTORY' in args``
predicate is a reliable disambiguator.

Usage::

    def test_periodic_ckpt_pattern_o(strace_syscalls):
        body = '''
        import torch
        from vllm import calibration_X as writer
        writer.dump_X_checkpoint(r"...")
        '''
        with strace_syscalls(body) as recorder:
            pass  # body ran inside the strace subprocess
        recorder.assert_order(
            ("openat", lambda a: ".tmp" in a),       # opens payload tmp
            ("fsync",  lambda a: True),               # payload fsync
            ("rename", lambda a: ".tmp" in a),        # tmp -> final
            ("openat", lambda a: "O_DIRECTORY" in a), # opens parent dir
            ("fsync",  lambda a: True),               # parent-dir fsync
        )

Linux-only: the fixture issues ``pytest.skip`` on systems without
``strace`` on ``PATH``.
"""
from __future__ import annotations

import contextlib
import re
import subprocess
import sys

import pytest


# Each strace line we care about looks like one of:
#   fsync(7)                                       = 0
#   rename("/tmp/xxx.tmp", "/tmp/xxx")             = 0
#   renameat2(AT_FDCWD, "/.../a.tmp", ...)         = 0
#   openat(AT_FDCWD, "/some/dir", O_RDONLY|...)    = 7
# strace's `-f` follow-forks emits an optional pid prefix in one of two
# forms depending on the strace version: ``[pid N] syscall(...)`` or just
# ``N syscall(...)``.
_SYSCALL_RE = re.compile(
    r"^(?:\[pid\s+\d+\]\s+|\d+\s+)?"
    r"(?P<name>fsync|rename|renameat|renameat2|openat)"
    r"\((?P<args>[^)]*)\)\s*=\s*(?P<ret>-?\d+|0x[0-9a-fA-F]+)"
)


class _StraceRecorder:
    """Holds the parsed syscall sequence and asserts ordering."""

    def __init__(self, syscalls: list[tuple[str, str]]) -> None:
        # list of (name, args_str), in observed wall-clock order.
        self.syscalls = syscalls

    def assert_order(self, *predicates) -> None:
        """Assert the recorded syscalls match the given ordered predicates.

        Each predicate is a ``(syscall_name, args_predicate_callable)``
        tuple. The recorded syscalls are scanned in observed order; each
        predicate must match SOME subsequent syscall (gaps are allowed
        — e.g. ``close`` between a matched ``fsync`` and the next
        matched ``rename`` does not break the order check).
        """
        i = 0
        for syscall_name, args_pred in predicates:
            matched = False
            while i < len(self.syscalls):
                name, args = self.syscalls[i]
                i += 1
                if name == syscall_name and args_pred(args):
                    matched = True
                    break
            if not matched:
                raise AssertionError(
                    f"strace order check failed: did not find {syscall_name} "
                    f"matching predicate after position {i}. "
                    f"Recorded syscalls (last 20): {self.syscalls[-20:]}"
                )


@pytest.fixture
def strace_syscalls(tmp_path):
    """Yield a factory: ``strace_syscalls(python_body)`` -> recorder.

    The factory runs the ``python_body`` source string inside a
    ``strace -f -e trace=fsync,rename,renameat,renameat2,openat``
    subprocess. The trace log is parsed into a list of
    ``(syscall_name, args)`` tuples that the test asserts ordering
    against.

    Skipped if ``strace`` is not on PATH (e.g. macOS, BSD CI).
    """
    if subprocess.run(
        ["which", "strace"], capture_output=True
    ).returncode != 0:
        pytest.skip(
            "strace not available; Pattern-O fsync-order tests require "
            "Linux + strace"
        )

    @contextlib.contextmanager
    def _runner(python_code: str):
        trace_log = tmp_path / "strace.log"
        cmd = [
            "strace",
            "-f",                                          # follow forks
            "-e", "trace=fsync,rename,renameat,renameat2,openat",
            "-o", str(trace_log),
            sys.executable, "-c", python_code,
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            raise AssertionError(
                f"strace subprocess failed (rc={res.returncode}):\n"
                f"stdout:\n{res.stdout}\nstderr:\n{res.stderr}"
            )
        syscalls: list[tuple[str, str]] = []
        for line in trace_log.read_text().splitlines():
            m = _SYSCALL_RE.search(line)
            if m:
                syscalls.append((m.group("name"), m.group("args")))
        yield _StraceRecorder(syscalls)

    yield _runner

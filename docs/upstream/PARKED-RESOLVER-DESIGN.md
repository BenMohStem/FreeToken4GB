# PARKED: CPU-MoE Explicit Cache-Size Fix — Full Design & Revival Kit

**Status: PARKED 2026-09-13** (user decision). The remote branch
`fix/cpu-moe-explicit-cache-size` (`81fed19`) was deleted; the full tested
change is preserved below and in local branch `archive/cpu-moe-fix-81fed19`
(clone at `/home/benmohstem/FreeToken`). This doc alone is enough to revive it.

## What it fixes (bug present on upstream main @ 9535656)

`python/freetoken/engine/engine.py`, `_adjust_config`, cpu-strategy block
(~L1535–1548): `override("moe_cache_size", 2 * num_experts)` +
`override("moe_prefill_overlap", True)` unconditionally — an explicit
`--moe-cache-size 256` silently became 512 slots (~34 MiB extra VRAM, the
boot-vs-OOM margin on 4-GiB cards) and `--disable-moe-prefill-overlap` was
ignored.

## The change (full diff vs upstream main 9535656)

```diff
diff --git a/python/freetoken/engine/engine.py b/python/freetoken/engine/engine.py
index 006c190..1bced63 100644
--- a/python/freetoken/engine/engine.py
+++ b/python/freetoken/engine/engine.py
@@ -41,9 +41,7 @@ def _require_offload_cache_size(cache_size: int, num_experts: int) -> None:
         raise ValueError(
             f"moe_cache_size={cache_size} is too small: need at least num_experts={num_experts} "
             f"slots. Pass --moe-cache-size/--moe-cache-rate, or use --moe-cache-auto "
-            f"(the default for offload/hybrid backends when no cache-sizing flag is given; "
-            f"--moe-strategy cpu always sizes its own fixed two-layer buffer and ignores "
-            f"cache-sizing flags)."
+            f"(the default for offload/hybrid backends when no cache-sizing flag is given)."
         )
 
 
@@ -1111,6 +1109,27 @@ def _parse_cpu_layers_spec(spec: str, num_moe_layers: int) -> frozenset[int]:
     return frozenset(round(i * num_moe_layers / k) for i in range(k))
 
 
+def _resolve_cpu_moe_cache(
+    moe_cache_size: int, moe_prefill_overlap: bool, num_experts: int
+) -> tuple[int, bool]:
+    """Resolve the GPU prefill slot cache for ``--moe-strategy cpu``.
+
+    An explicit ``--moe-cache-size`` is authoritative -- the pre-0.1.3 code
+    unconditionally replaced it with ``2 * num_experts``, silently doubling
+    a user-requested cache (256 -> 512 slots on a 256-expert model), VRAM that
+    small GPUs cannot spare. A cache below ``2 * num_experts`` cannot feed the
+    two-buffer prefill overlap, so it disables overlap (synchronous
+    single-buffer prefill). With no explicit size, the historical default
+    stays: the two-layer double buffer (``2 * num_experts``, overlap on).
+
+    Pure function so the config resolution stays CPU-testable.
+    """
+    if moe_cache_size > 0:
+        overlap = moe_prefill_overlap and moe_cache_size >= 2 * num_experts
+        return moe_cache_size, overlap
+    return 2 * num_experts, True
+
+
 def _resolve_cpu_layers(config: EngineConfig, num_moe_layers: int, *, reserved: int = 0, method=None) -> frozenset[int]:
     """MoE layer ids whose decode runs on the CPU executor.
 
@@ -1533,18 +1552,32 @@ def _adjust_config(config: EngineConfig):
             override("moe_cache_auto", False)
 
     if is_moe and config.moe_strategy == "cpu":
-        # CPU-compute decode keeps experts in host RAM and computes them on the CPU;
-        # the GPU only holds the two-layer prefill double buffer. So the slot cache is
-        # fixed at exactly two expert layers (prefill overlap requires >= 2*num_experts)
-        # and --moe-cache-size / --moe-cache-auto / --moe-cache-rate do not apply.
+        # CPU-compute decode keeps experts in host RAM and computes them on the CPU.
+        # See _resolve_cpu_moe_cache: an explicit --moe-cache-size is authoritative
+        # (the old code silently replaced it with 2*num_experts); the default keeps
+        # the historical two-layer prefill double buffer.
         num_experts = config.model_config.num_experts
+
         if getattr(config, "moe_cache_auto", False):
             override("moe_cache_auto", False)
-        override("moe_cache_size", 2 * num_experts)
-        override("moe_prefill_overlap", True)
+
+        size, overlap = _resolve_cpu_moe_cache(
+            config.moe_cache_size, config.moe_prefill_overlap, num_experts
+        )
+        if not overlap and config.moe_prefill_overlap:
+            # Explicit size below 2*num_experts cannot feed the two-buffer
+            # prefill overlap; say so instead of silently flipping it off.
+            logger.warning_rank0(
+                f"MoE strategy 'cpu': explicit moe_cache_size={size} < "
+                f"2*num_experts={2 * num_experts}; disabling prefill overlap "
+                f"(synchronous single-buffer prefill)"
+            )
+        override("moe_cache_size", size)
+        override("moe_prefill_overlap", overlap)
         logger.info_rank0(
-            f"MoE backend 'cpu': decode computes experts on CPU; GPU keeps a "
-            f"two-layer prefill buffer (moe_cache_size={2 * num_experts})"
+            f"MoE strategy 'cpu': decode computes experts on CPU; "
+            f"GPU prefill cache size={config.moe_cache_size}, "
+            f"overlap={config.moe_prefill_overlap}"
         )
 
     if (
```

## The test (tests/engine/test_cpu_moe_cache_size.py — full source)

```python
"""Resolver for the cpu MoE strategy's GPU prefill slot cache (--moe-cache-size).

CPU-only: exercises _resolve_cpu_moe_cache without a GPU.
"""

from __future__ import annotations

import pytest

from freetoken.engine.engine import _resolve_cpu_moe_cache as resolve

E = 256  # num_experts of Qwen3.6-35B-A3B


def test_explicit_size_is_authoritative():
    # The pre-0.1.3 behavior doubled this to 2*E = 512.
    size, overlap = resolve(256, True, E)
    assert size == 256
    assert overlap is False  # 256 < 2*E cannot feed the two-buffer overlap


def test_explicit_size_at_or_above_double_keeps_overlap():
    size, overlap = resolve(512, True, E)
    assert size == 512
    assert overlap is True
    size, overlap = resolve(1024, True, E)
    assert size == 1024
    assert overlap is True


def test_explicit_size_respects_disabled_overlap():
    size, overlap = resolve(256, False, E)
    assert size == 256
    assert overlap is False


def test_default_is_two_layer_double_buffer():
    size, overlap = resolve(0, True, E)
    assert size == 2 * E
    assert overlap is True
    # even if the user passed --disable-moe-prefill-overlap without a size,
    # the default two-layer buffer keeps overlap semantics
    size, overlap = resolve(0, False, E)
    assert size == 2 * E
    assert overlap is True


def test_negative_size_treated_as_default():
    # argparse-level guard aside, be robust: anything not > 0 means "not explicit"
    size, overlap = resolve(-1, True, E)
    assert size == 2 * E
    assert overlap is True


def test_small_expert_models():
    # 8-expert model: explicit 16 == 2*E boundary keeps overlap
    size, overlap = resolve(16, True, 8)
    assert size == 16
    assert overlap is True
    # explicit 15 < 2*E disables it
    size, overlap = resolve(15, True, 8)
    assert size == 15
    assert overlap is False


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-q"]))
```

## Semantic difference vs the fork's inline fix

| Input | Fork inline fix (shipped on lowvram-4gb) | Resolver (this, for upstream) |
|---|---|---|
| explicit size, overlap ON, size ≥ 2E | size kept, overlap ON | size kept, overlap ON (same) |
| explicit size, overlap ON, size < 2E | size kept, overlap left ON (inconsistent state, user asked for both) | size kept, overlap OFF + warning |
| explicit size, overlap OFF | size kept, overlap OFF (same) | size kept, overlap OFF (same) |
| no size, overlap ON | 2E slots, overlap ON (same) | 2E slots, overlap ON (same) |
| no size, overlap OFF | E slots, overlap OFF | 2E slots, overlap ON (upstream historical default) |

The user's proven recipe (`--moe-cache-size 256 --disable-moe-prefill-overlap`)
resolves identically under both. See AI-CONTEXT-4GB.md §5 for why the fork
keeps the inline variant (byte-identity with the daily-driver install).

## Evidence (verified 2026-09-13)

- 6/6 test cases passed on the parked branch (`81fed19`), run via:
  `PYTHONPATH=$PWD/python:/opt/scilab/.venv/lib/python3.13/site-packages python3 -m pytest tests/engine/test_cpu_moe_cache_size.py -v`
- On stock upstream main the test FAILS (ImportError: resolver absent; and
  the size silently doubles 256→512)
- Real-hardware proof of the bug + fix effect on the RTX 3050 4-GiB testbed
  with `nvidia/Qwen3.6-35B-A3B-NVFP4`: `[VRAM-DIAG] after_moe_cache` lines
  (897.8 MiB free with the honored 256-slot cache; the stock doubling cost
  ~34 MiB more), 64K pool served a 65,349-token request.

## Revival steps (when the user approves + supplies a classic PAT)

1. Re-verify upstream main still has the bug: fetch
   `upstream/main`'s engine.py cpu block (rename to `moe_strategy` happened
   long ago; the override lines are the tell).
2. `git checkout -b fix/cpu-moe-explicit-cache-size upstream/main` in the
   clone; apply the diff above; add the test file; `python3 -m py_compile`.
3. Run the test (command above); verify fail-before/pass-after by
   temporarily checking out stock upstream main's engine.py.
4. Push to the fork (fine-grained PAT Contents:RW is enough for the fork).
5. Open the issue (ISSUE-TEXT.md), get its number, then open the PR
   (PR-TEXT.md) with `Fixes #N` substituted.
6. Human gate: the user reviews the final texts before posting (upstream
   AI policy: human ownership of AI-assisted PRs).

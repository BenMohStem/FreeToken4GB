# Upstream Pull Request — ready-to-paste

> **REVIVAL REQUIRED FIRST**: the original branch
> `fix/cpu-moe-explicit-cache-size` (`81fed19`) was deleted from the fork on
> 2026-09-13 (user decision; fine-grained PAT cannot post upstream anyway).
> Recreate it exactly per `docs/upstream/PARKED-RESOLVER-DESIGN.md` before
> opening this PR — the head ref below only exists once that branch is pushed.
> Open the ISSUE (see ISSUE-TEXT.md) first, get its number N, and substitute
> `#<N>` below.

Post as a PR: head `BenMohStem:fix/cpu-moe-explicit-cache-size` →
base `FlashML-org:main`. Prefill via:
`https://github.com/FlashML-org/FreeToken/compare/main...BenMohStem:fix%2Fcpu-moe-explicit-cache-size?expand=1`

---

# Title (Conventional Commits; used as squash subject)

fix(engine): respect explicit --moe-cache-size with the cpu moe strategy

# Body

Fixes #<N>.

The cpu strategy block in `_adjust_config` unconditionally overrode
`moe_cache_size` to `2 * num_experts` and forced `moe_prefill_overlap=True`,
silently ignoring an explicit `--moe-cache-size` (256 became 512 on a
256-expert model) and `--disable-moe-prefill-overlap`.

Change: resolution moves into a pure, CPU-testable helper
`_resolve_cpu_moe_cache(size, overlap, num_experts)`:
- an explicit size > 0 is authoritative;
- overlap survives only when size >= 2*num_experts — a smaller cache cannot
  feed the two-buffer prefill overlap, so it disables with a warning
  (synchronous single-buffer prefill) instead of silently doubling the size;
- no explicit size keeps the historical default (2*num_experts, overlap on).
Also updates the `_require_offload_cache_size` docstring that documented the
old overriding behavior.

Testing:
- new `tests/engine/test_cpu_moe_cache_size.py` (CPU-only, 6 cases) — fails
  on unpatched `main` (resolver absent; size silently doubled), passes here;
- real hardware: RTX 3050 4 GiB laptop, `nvidia/Qwen3.6-35B-A3B-NVFP4`,
  `ft serve --moe-strategy cpu --moe-cache-size 256 --disable-moe-prefill-overlap ...`
  now resolves `moe_cache_size=256, overlap=False`; a 64K-token pool served
  a 65,349-token request on this config.

Per the AI policy: this change was developed with AI assistance; I reviewed
the diff, ran the tests, and run the engine daily on this hardware.

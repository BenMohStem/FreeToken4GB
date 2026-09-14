# Upstream Bug Report — ready-to-paste

Post as a NEW ISSUE on https://github.com/FlashML-org/FreeToken/issues/new
(fine-grained PATs cannot post upstream — 403; use the web UI or a classic PAT
with `repo` scope). The `# Title` goes in the title field; `# Body` in the body.

## How to prefill via URL

Encode title+body and open:
`https://github.com/FlashML-org/FreeToken/issues/new?title=<urlencoded title>&body=<urlencoded body>`

---

# Title

--moe-cache-size silently doubled to 2*num_experts with the cpu moe backend (and --disable-moe-prefill-overlap ignored)

# Body

### Environment
- GPU: NVIDIA RTX 3050 Laptop, 4096 MiB; CPU: Ryzen 5 5600H; RAM: 32 GiB
- OS: Debian 13 (trixie), driver 610.43.02, CUDA 13.3
- FreeToken: 0.1.2 wheel (reproduced); the same code path exists on current
  `main` where the flag is renamed `--moe-backend` → `--moe-strategy`
- Checkpoint: `nvidia/Qwen3.6-35B-A3B-NVFP4` (256 experts)

### Command
ft serve --model-path nvidia/Qwen3.6-35B-A3B-NVFP4 --moe-backend cpu --moe-cache-size 256 --disable-moe-prefill-overlap ...

### Actual
The resolved-config log shows the explicit size doubled and the disabled
overlap re-enabled:
`MoE backend 'cpu': ... GPU keeps a two-layer prefill buffer (moe_cache_size=512)`
`_adjust_config`'s cpu block overrides both unconditionally
(`override("moe_cache_size", 2 * num_experts)`; `override("moe_prefill_overlap", True)`).

### Expected
- An explicit `--moe-cache-size` is authoritative.
- `--disable-moe-prefill-overlap` is honored (or, for an explicit size below
  2*num_experts, overlap auto-disables with a warning — a smaller cache
  cannot feed the two-buffer overlap anyway).

### Impact
On small GPUs the doubled prefill expert buffer (~34 MiB extra with NVFP4
experts on a 256-expert model) is the margin between booting and OOM, and the
forced overlap re-enables double-buffering the user explicitly turned off.

I have a minimal fix + CPU-only unit test ready and will open a PR linking
this issue.

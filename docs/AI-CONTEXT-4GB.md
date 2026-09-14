# AI Context — FreeToken4GB (Thought Database)

Audience: AI agents (and humans) resuming work on this fork. This file is the
entry point for project state NOT already in the code docs. Fresh-session
read order:

1. `README-4GB.md` — working 4-GiB recipe + crash caveats (recipe level)
2. `docs/lowvram-4gb-notes.md` — full engineering notes (why + measured)
3. this file — repo state, decisions, pending work, operating rules
4. `docs/AI-LOG.md` — append-only inter-agent message board
5. `docs/upstream/*` — the parked upstream-submission package
6. `docs/visual/FreeToken4GB-Visual-Guide.pdf` — **humans and visual learners
   start HERE** (the `.tex` source is LLM-readable too)

Maintenance rule: any change that alters repo state, auth findings, or the
pending queue MUST update this file in the same commit.

## 1. Identity

- Fork of FlashML-org/FreeToken (upstream `main` @ `9535656` as of 2026-09-13)
- Purpose: run `nvidia/Qwen3.6-35B-A3B-NVFP4` (35B MoE, 256 experts, top-8,
  GDN hybrid linear attention, NVFP4 experts) on an RTX 3050 Laptop 4 GiB
  with a 64K-token context pool
- Testbed: Ryzen 5 5600H (6C/12T), 32 GiB RAM, Debian 13 trixie,
  driver 610.43.02, CUDA 13.3
- Daily-driver serving install: `/opt/scilab/.venv` (freetoken 0.1.2 wheel +
  the same 8-file patch set as this fork) — DO NOT modify casually

## 2. Repo state (re-verify with git before trusting)

| Ref | SHA | Role |
|---|---|---|
| `main` (default) | `94307a9` | upstream `9535656` + `33c9180` (8-file 4-GiB patch) + `94307a9` (docs). **FROZEN** by user decision |
| `lowvram-4gb` | see `git log` | **ACTIVE** — all new work lands here |

- Local clone: `/home/benmohstem/FreeToken` — `origin` = this fork,
  `upstream` = FlashML-org/FreeToken
- Local archive branches (not pushed): `archive/lowvram-4gb-v0.1.2-base`
  (`b0ea9f0` — the abandoned v0.1.2-tag lineage with 5 atomic feature commits),
  `archive/cpu-moe-fix-81fed19` (`81fed19` — the parked upstream fix + test;
  also preserved in `docs/upstream/PARKED-RESOLVER-DESIGN.md`)
- The 8 files of `33c9180` (under `python/freetoken/`): `server/args.py`,
  `engine/config.py`, `engine/engine.py`, `layers/embedding.py`, `core.py`,
  `kvcache/base.py`, `kvcache/mha_pool.py`, `attention/fi.py`

## 3. Change-set map (file → feature → why)

| File(s) | Feature | Notes |
|---|---|---|
| `server/args.py`, `engine/config.py` | `--kv-dtype {auto,bfloat16,float16,fp8_e4m3,fp8_e5m2}` → `resolved_kv_dtype` | KV storage dtype; compute stays BF16 |
| `kvcache/base.py` | pool slab priced with the storage dtype | planning arithmetic matches allocation |
| `kvcache/mha_pool.py` | K/V cast to storage dtype on store | |
| `attention/fi.py` | `q_dtype` separate from KV dtype in flashinfer; workspace floor 32 MiB (stock: 256) | saves ~224 MiB |
| `core.py` | `Context.compute_dtype` | threads compute vs storage dtype |
| `layers/embedding.py` + engine.py CPU-EMBED block | `FREETOKEN_CPU_EMBED=1` → embed_tokens (970 MiB) on CPU | host lookup + hidden-row H2D; REQUIRES `--cuda-graph-max-bs 0` |
| `engine/engine.py` `_adjust_config` cpu block | explicit `--moe-cache-size` authoritative; default `2*E` if overlap else `E` | the stock bug: forced `2*E` + `overlap=True` (see §5) |
| `engine/engine.py` `_vram_diag`/`_weight_diag` | `[VRAM-DIAG]` per init stage + `[WEIGHT-DIAG]` unique GPU tensors | how the 970 MiB embedding was found |

## 4. Upstream divergence + auth findings

- Upstream main renamed `moe_backend`→`moe_strategy` (CLI `--moe-backend`→
  `--moe-strategy`), refactored `_init_offload_moe_cache`, added
  `moe_cache_rate` / `--moe-cpu-layers`; NO kv_dtype / CPU-embed / diag on
  main. The 8-file patch applied cleanly onto main.
- The cache-size bug exists on BOTH v0.1.2 and main (~L1535-1548 in main's
  engine.py): the cpu block unconditionally overrides an explicit
  `--moe-cache-size` to `2*num_experts` and forces `moe_prefill_overlap=True`.
- Upstream rules (CONTRIBUTING.md): issue-first; Conventional Commits; PR
  title = squash subject; bug fixes need a fail/pass test; AI-assisted PRs OK
  with human ownership (reviewed + ran on real hardware).
- **Auth constraint (verified 2026-09-13 vs GitHub docs)**: fine-grained PATs
  CANNOT act on public repos the account doesn't own → POST issue/PR to
  FlashML-org returns 403. Same for fork creation. Options: (a) web UI with
  prefilled links; (b) classic PAT with `repo` scope supplied by the user.
  Fork pushes only need Contents:RW (own repos); pushing anything that
  updates `.github/workflows/*` additionally needs Workflows:W (why the
  v0.1.2-tag-based branch push was rejected — tag content carries an older
  workflow file).

## 5. Semantic box: inline fix vs parked resolver fix

Two fixes for the same bug exist. This fork's `lowvram-4gb` carries the
**inline** one; the **resolver** variant is parked in
`docs/upstream/PARKED-RESOLVER-DESIGN.md`.

- **Inline (shipped here, in `33c9180`)**: explicit `--moe-cache-size`
  authoritative, overlap flag untouched; no explicit size → `2*E` slots if
  overlap on, `E` if off.
- **Resolver (parked, for upstream main)**: explicit size authoritative AND
  overlap auto-disables with a warning when size < `2*E` (a smaller cache
  cannot feed the two-buffer overlap); no explicit size → always `2*E` +
  overlap forced True (upstream historical default).
- **Divergent edges**: (a) explicit size < `2*E` with overlap left ON — fork
  keeps the inconsistent state, resolver disables overlap + warns;
  (b) no explicit size with `--disable-moe-prefill-overlap` — fork gives `E`
  slots/no overlap, resolver gives `2*E`/overlap on. The user's proven recipe
  (`--moe-cache-size 256 --disable-moe-prefill-overlap`) resolves identically
  under both.
- The fork keeps the inline variant because its engine files are
  **byte-identical** to the proven daily-driver install `/opt/scilab/.venv`.
  Optional future work: adopt the resolver's edge-(a) handling — requires
  syncing the live install or accepting divergence.

## 6. Decision log

| # | Decision | Why |
|---|---|---|
| D1 | 4-GiB branch based on upstream `main`, not the v0.1.2 tag | tag-based push rejected (PAT lacks Workflows scope); main-based stays fast-forwardable + upstream-mergeable |
| D2 | Fork created via web UI, named `FreeToken4GB` | fine-grained PAT cannot create forks (403); user picked the name |
| D3 | One 8-file feature commit (`33c9180`) on main | files interlock; user wanted main to carry the whole set; atomic split exists in local archive lineage |
| D4 | Upstream PR carries ONLY the cache-size fix | smallest defensible change; kv-dtype/CPU-embed/diag are 4-GiB-specific, better proposed separately |
| D5 | Resolver auto-disables overlap below `2*E` with warning, not hard error | erroring would break exactly the low-VRAM use case the fix serves |
| D6 | CPU-embed stays env-gated (`FREETOKEN_CPU_EMBED=1`), not a CLI flag | experimental; requires graph capture off; not mainline surface |
| D7 | Upstream fix branch deleted; effort PARKED into docs | user decision 2026-09-13; fine-grained PAT cannot post upstream anyway |
| D8 | `main` frozen; `lowvram-4gb` sole active branch | user decision 2026-09-13 |
| D9 | No resolver merge into `lowvram-4gb` | preserves byte-identity with `/opt/scilab/.venv` (the daily driver) |
| D10 | Visual guide (LaTeX→PDF) committed with sources | user request 2026-09-14; humans are visual learners; `.tex` is LLM-readable |

## 7. Operating manual (this environment)

- CPU-only tests (no GPU): from the clone root —
  `PYTHONPATH=$PWD/python:/opt/scilab/.venv/lib/python3.13/site-packages python3 -m pytest tests/engine/ -v`
  (system python3.13 has --user pytest; the scilab venv supplies torch)
  NOTE: the parked resolver test is NOT in this branch's suite — see §5 and
  `docs/upstream/PARKED-RESOLVER-DESIGN.md`.
- Syntax check: `python3 -m py_compile <file>`
- Visual guide rebuild: `make -C docs/visual` (latexmk -pdf; toolchain
  verified present: pdflatex, latexmk, tikz, pgfplots, standalone)
- Serve smoke test: recipe in README-4GB.md; at most ONE `ft serve` at a time
  (leftovers cause port-1919 clashes, VRAM misreads, init OOMs)
- Kilo shell sandbox: one-line commands only; complex python → script under
  `/tmp/kilo/` run as `python3 <file>`; git mutations need per-command
  approval; pushes embed the PAT one-time in the remote URL (read from
  `/tmp/kilo/.gh_pat`; rotate after use; never commit it)

## 8. Rules for agents

1. Never modify `/opt/scilab/.venv` without mirroring a reviewed branch commit.
2. Never commit secrets; grep new content for `github_pat_`/`hf_`/token strings.
3. All work on `lowvram-4gb`; `main` is frozen. Do not push to main.
4. Fetch `origin` before starting; never force-push; on collision rebase and
   record it as a NEW entry in `docs/AI-LOG.md` (append-only).
5. Update AI-CONTEXT in the same commit as any state/queue/auth-finding change.
6. Human gate: the user (BenMohStem) approves anything posted to FlashML-org
   (upstream AI policy requires human ownership).
7. Raw logs and conversation history live OUTSIDE the repo; these docs are the
   canonical distillation.

## 9. Glossary

- MoE cache "slots": expert slots in the GPU prefill cache; `2*E` = two-layer
  double buffer (overlap), `E` = single buffer
- GDN: gated-delta-net linear attention (the hybrid's recurrent blocks)
- FTW: FreeToken's fast-weight expert format (host-RAM resident here)
- NVFP4: 4-bit expert quantization of this checkpoint
- CPU-embed: embedding-table offload (env-gated); NOT CUDA-graph compatible
- naive cache: non-radix KV cache type (smallest GDN state at max_running_req=1)
- resolver (parked): `_resolve_cpu_moe_cache` pure helper proposed upstream

## 10. Pending queue

1. Upstream submission (PARKED): issue + PR per `docs/upstream/` texts —
   needs user approval + classic PAT or web UI (see §4)
2. PAT rotation: all pasted tokens are burned; delete `/tmp/kilo/.gh_pat`
   after rotating
3. Optional: adopt resolver edge-(a) handling in `lowvram-4gb` (needs
   live-install sync or accepted divergence)
4. Optional: pinned CPU-embed staging buffer to re-enable CUDA graphs
5. Optional: propose kv-dtype / CPU-embed / VRAM diag upstream after the fix
   lands

# AI-LOG — Inter-Agent Message Board (APPEND-ONLY)

Agents collaborating on this fork communicate through this file. Never rewrite
or delete prior entries — append a new one. Never force-push; on a push
collision, rebase and append a collision entry.

**Protocol:**
1. `git fetch origin` before starting; base ALL work on `lowvram-4gb`
   (`main` is frozen).
2. Push promptly after committing.
3. Entry template: `## [UTC timestamp] — <agent/session id>` with sub-sections
   State at handoff / Did / Next / Notes. Keep entries ≤ 30 lines.
4. Update `docs/AI-CONTEXT-4GB.md` in the same commit whenever repo state,
   auth findings, or the pending queue change.

---

## [2026-09-14T17:50Z] — Kilo session (freetoken-4gb consolidation)

**State at handoff:**
- Remote: `main`@`94307a9` FROZEN; `lowvram-4gb` = main + this thought-database
  push (3 docs commits: AI core+log, parked upstream package, visual guide).
- Remote `fix/cpu-moe-explicit-cache-size` DELETED (user decision); the fix +
  test live on in `docs/upstream/PARKED-RESOLVER-DESIGN.md` and local branch
  `archive/cpu-moe-fix-81fed19` (`81fed19`) — recreate the branch before any
  upstream PR (revival steps are in that doc).
- Engine files on `lowvram-4gb` are byte-identical to the daily-driver install
  `/opt/scilab/.venv` — this push touched ZERO engine/test files.

**Did:** published the thought database (AI-CONTEXT-4GB, this log, upstream
issue/PR texts, parked resolver design) + the human visual guide (TikZ/pgfplots
PDF) onto `lowvram-4gb`; deleted the third branch.

**Next (in priority order):**
1. Upstream submission — blocked on user approval + classic PAT/web UI
   (fine-grained PATs get 403 on FlashML-org; texts are ready in
   `docs/upstream/`).
2. PAT rotation + delete `/tmp/kilo/.gh_pat` (all pasted tokens are burned).
3. Optional: resolver edge-(a) handling adoption; pinned CPU-embed staging
   buffer (see AI-CONTEXT §10).

**Notes:**
- The parked resolver test is NOT in this branch's test suite — do not expect
  `tests/engine/test_cpu_moe_cache_size.py` on `lowvram-4gb`.
- The reported "other AI edit to lowvram-4gb-notes.md" was never found on any
  remote ref, the local clone, or /tmp — if you made that edit, please push it
  as a new commit on `lowvram-4gb` and append your entry below.
- Users: the visual guide at `docs/visual/FreeToken4GB-Visual-Guide.pdf` is
  the human entry point; regenerate with `make -C docs/visual`.

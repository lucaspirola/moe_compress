# Heavy Spec-Compliance Audit — Three Regimes
# Target spec: `max_quality/ALGORITHM_REFERENCE.md`

## Goal
Produce a durable, evidence-grounded audit of the spec against THREE
concurrent compliance regimes:

1. **Paper regime** — fidelity to the 10 cited papers (§13), modulo
   sanctioned deviations in Chapter 12.
2. **H200 regime** — correctness and soundness of the pipeline's
   hardware-aware execution choices (VRAM budgets, FP precision,
   kernel choices, torch.compile, batching, host-RAM budgets,
   FlashAttention) against the actual H200 hardware spec (141 GB
   HBM3e, Hopper SM_90, FP8 E4M3/E5M2 supported) and against the
   constraints of the HF Jobs runtime the pipeline targets (default
   30-min timeout, bucket FUSE non-durable on cancel, Hub commits
   durable).
3. **Resume regime** — soundness of the crash-resume model in §11
   (durability boundaries, atomic writes, idempotence, format-version
   guards, .pt vs .json ordering, HF Hub commit semantics, bucket
   non-durability).

A spec behavior is COMPLIANT only if every regime that touches it
audits clean. A paper deviation justified "for H200 reasons" passes
only if the H200 claim is audited and stands; a resume mechanism
justified "to satisfy durability" passes only if the durability claim
audits clean.

## Execution context
- **The audit runs locally** on a WSL2 machine with normal Claude
  Code tooling: `Read`, `Bash`, `Edit`, `Task` for subagent dispatch,
  WebFetch, etc. No HF Jobs, no Hub uploads required for the audit
  itself.
- **The target pipeline** (the thing being audited) runs on HF Jobs
  H200. The H200 and Resume regimes audit the pipeline's claims about
  THAT runtime, not the audit's own runtime.
- Auto-memory at `~/.claude/projects/-home-lucas-ai-moe-compress/memory/`
  is read directly as one of several truth sources for the H200 and
  Resume regimes. Auto-memory facts about HF Jobs (timeouts, bucket
  non-durability, FP8 hardware) are authoritative inputs.
- Working dir: `/home/lucas/ai/moe_compress`. The audit produces
  artifacts under `audit/spec_compliance/`. Local commits with prefix
  `audit(spec):` are durable (normal git repo).

## Hard rules
- **DO NOT modify the spec or Chapter 12.** Findings only. Spec edits
  are a separate task downstream of this audit.
- **DO NOT delete artifacts.** Append-only JSONL; sibling `*.review.md`
  for corrections.
- **DO NOT trust your memory of prior conversations.** Disk is truth.
  Read `00_manifest.json` and the artifact tree at every fresh start.
- **DO NOT use WebFetch as the primary paper-reading tool.** Always
  download PDFs locally and parse with `pdftotext` / `marker`. WebFetch
  is reserved for hardware datasheets and HF docs.
- **DO consult `~/.claude/projects/-home-lucas-ai-moe-compress/memory/`**
  as a citable truth source for the H200 and Resume regimes.
- All commits local-only. Prefix: `audit(spec):`.

## Artifact tree
All under `audit/spec_compliance/` (relative to repo root). JSONL
throughout. Schemas mandatory — downstream phases parse these files
programmatically.

```
audit/spec_compliance/
├── 00_manifest.json
├── 01_papers/<arxiv_id>/{source.pdf, source.md, extraction_log.txt, claims.jsonl}
├── 02_spec_index/
│   ├── citations.jsonl          # Eq./§/Algorithm/Theorem refs in spec
│   ├── formulas.jsonl           # code-fenced math blocks
│   ├── h200_claims.jsonl        # every hardware/perf/HF-Jobs claim
│   └── resume_claims.jsonl      # every durability/idempotence claim
├── 03_truth_refs/
│   ├── h200_truth.md            # canonical H200 + HF Jobs facts
│   └── resume_truth.md          # canonical durability facts
├── 04_crossref/
│   ├── paper_<id>.crossref.jsonl
│   ├── h200.crossref.jsonl
│   └── resume.crossref.jsonl
├── 05_chapter12_audit/ch12_rows.jsonl
├── 06_findings/{findings.jsonl, findings.md}
└── logs/phase_<N>_<timestamp>.log
```

### `00_manifest.json` schema
```json
{
  "phases": {
    "1_paper_acquisition":   {"status": "pending|in_progress|done", "papers_done": []},
    "2_paper_extraction":    {"status": "...", "papers_done": []},
    "3_truth_refs":          {"status": "...", "h200_done": false, "resume_done": false},
    "4_spec_indexing":       {"status": "...", "subindexes_done": []},
    "5_paper_crossref":      {"status": "...", "papers_done": []},
    "6_h200_crossref":       {"status": "..."},
    "7_resume_crossref":     {"status": "..."},
    "8_chapter12_audit":     {"status": "..."},
    "9_findings_merge":      {"status": "..."}
  },
  "spec_path": "max_quality/ALGORITHM_REFERENCE.md",
  "spec_sha256": "<hash at audit start>",
  "ch12_sha256": "<hash of just the §12 region at audit start>",
  "started_at": "<iso8601>"
}
```

### Schemas

#### `01_papers/<id>/claims.jsonl`
```json
{"id": "2604.06542#eq11", "paper_id": "2604.06542", "kind": "equation|algorithm|theorem|lemma|hyperparameter|claim|table_value", "label": "Eq. 11", "section": "§3.2", "page": 5, "verbatim": "...", "paraphrase": "...", "preconditions": "...", "consequences": "..."}
```
**Target ~30–80 claims per paper, cap at 100.** Skip purely
background/related-work mentions; capture anything the spec might rely on.

#### `02_spec_index/citations.jsonl`
```json
{"spec_line": 175, "spec_section": "§4 Stage 1", "raw_text": "(Eq. 11 — sum, not mean)", "cited_paper": "2604.06542", "cited_label": "Eq. 11", "context": "..."}
```

#### `02_spec_index/h200_claims.jsonl`
Every assertion the spec makes about hardware behavior, compute time,
VRAM, host RAM, FP precision, kernel choice, torch.compile, batching,
HF Jobs timeouts/flavors, or numerical-equivalence-under-optimization.
```json
{
  "id": "H200-0017",
  "spec_line": 390,
  "spec_section": "§6 Stage 3 Phase B.2",
  "claim_kind": "vram_budget|host_ram_budget|fp_precision|kernel_choice|torch_compile|batch_scheduling|wall_time|numerical_equivalence|hardware_capability|throughput|hf_jobs_runtime",
  "verbatim": "originals are snapshotted to CPU RAM (~50 GB; H200 has 256 GB host RAM)",
  "asserted_value": {"host_ram_used_gb": 50, "host_ram_available_gb": 256},
  "depends_on_paper_deviation": false,
  "ch12_row_if_any": null,
  "claim_summary": "Originals snapshot needs 50 GB; H200 host has 256 GB headroom"
}
```

#### `02_spec_index/resume_claims.jsonl`
Every assertion about crash-resume, durability boundaries, idempotence,
atomic writes, checkpoint format, .pt/.json invariants, HF Hub commits,
or bucket usage.
```json
{
  "id": "RES-0009",
  "spec_line": 710,
  "spec_section": "§11 Within-Stage Resume",
  "claim_kind": "atomic_write|durability_boundary|idempotence|invariant|format_version|hub_commit|bucket_usage|recovery_action",
  "verbatim": "If `.pt` exists without `.json` (orphaned by crash between snapshot and JSON write), the `.pt` is deleted and the layer is reprocessed from scratch.",
  "invariant_implied": "Layer-completion is defined by both .pt and .json existing simultaneously",
  "claim_summary": "Crash between snapshot and JSON write must not leave layer in partially-completed state"
}
```

#### `03_truth_refs/h200_truth.md`
A markdown reference of canonical H200 + HF Jobs facts. **The audit
consults this as ground truth; subagents must not improvise hardware
facts.** Populate (and treat as authoritative) the following fields,
citing sources:

- HBM3e capacity: **141 GB** per GPU
- HBM3e bandwidth: 4.8 TB/s
- Compute: SM_90 (Hopper); FP8 E4M3 + E5M2 supported (auto-memory:
  `feedback_fp8_a100_hardware.md` — FP8 inference is Hopper-only,
  A100 has no FP8)
- Host RAM on standard HF Jobs H200 instance: confirm via current docs
- TensorFloat-32 / BF16 / FP16 native; FP32 reduced throughput
- FlashAttention-2/3 supported
- torch.compile: works for prefill-dominant graphs; care with dynamic
  shapes; cudagraphs caveats
- HF Jobs default timeout: 30 min (auto-memory:
  `feedback_hf_jobs_timeout.md`); timeout-kill = cancel
- HF Jobs bucket FUSE non-durable on cancel/timeout (auto-memory:
  `reference_hf_jobs_durability.md`); Hub commits durable

Each fact line in this file must include a `<!-- source: ... -->`
HTML comment with the citation (NVIDIA H200 datasheet URL, HF Jobs
docs URL, or absolute path to the auto-memory file).

#### `03_truth_refs/resume_truth.md`
Canonical durability facts for the resume regime. Populate from:
- HF Hub commits are durable (atomic on the Hub side)
- HF Bucket FUSE: NOT durable on cancel/timeout (auto-memory:
  `reference_hf_jobs_durability.md`)
- Local FS atomic write idiom: write to `*.tmp` then `os.rename`
- POSIX rename guarantees within same filesystem; not across mounts
- JSON ordering rule: write payload file (`.pt`) BEFORE manifest
  (`.json`) so `.json` existence implies payload existence
- Per-stage policy: spec §11 mandates Hub commit per stage —
  duplicate that as canonical durability boundary

Each fact must cite source.

#### `04_crossref/paper_<id>.crossref.jsonl`
```json
{
  "claim_id": "2604.06542#eq11",
  "spec_uses_it": true,
  "spec_locations": [{"line": 175}],
  "verdict": "match|deviation_in_ch12|deviation_uncovered|paraphrase_drift|spec_omits",
  "ch12_row": "D4",
  "spec_paraphrase": "...",
  "paper_verbatim": "...",
  "evidence_pages": [5],
  "sanction_justification": "ch12|h200|resume|none",
  "severity": "critical|high|medium|low|minor|info|none",
  "notes": "..."
}
```
`sanction_justification` records WHICH regime sanctions a paper
deviation; cross-checked in phases 6/7.

#### `04_crossref/h200.crossref.jsonl`
```json
{
  "claim_id": "H200-0017",
  "spec_location": {"line": 390},
  "truth_evidence": [{"truth_ref": "h200_truth.md#L8", "fact": "host RAM 256 GB on standard HF Jobs H200"}],
  "verdict": "consistent|inconsistent|unverifiable|stale|paper_compliance_break",
  "verdict_detail": "...",
  "implies_paper_deviation": false,
  "linked_paper_claim_id": null,
  "severity": "...",
  "notes": "..."
}
```
- `consistent` — spec claim matches truth
- `inconsistent` — spec claim contradicts truth (FINDING)
- `unverifiable` — truth ref couldn't confirm; needs human (FINDING info)
- `stale` — truth has moved on (e.g. driver/kernel update changes the
  claim); spec needs an update (FINDING)
- `paper_compliance_break` — H200 optimization breaks numerical
  equivalence with the paper (e.g. FP8 teacher when paper uses BF16);
  must have Ch. 12 row, else FINDING

#### `04_crossref/resume.crossref.jsonl`
```json
{
  "claim_id": "RES-0009",
  "spec_location": {"line": 710},
  "truth_evidence": [{"truth_ref": "resume_truth.md#L18", "fact": "JSON ordering rule"}],
  "verdict": "consistent|inconsistent|incomplete|unverifiable",
  "verdict_detail": "...",
  "missing_invariant": null,
  "severity": "...",
  "notes": "..."
}
```
`incomplete` — claim is right but needs additional invariants
(e.g. mentions atomic .pt write but not the rename order).

#### `05_chapter12_audit/ch12_rows.jsonl`
```json
{
  "row_id": "D4",
  "stage": 1,
  "paper_says_column": "...",
  "implementation_does_column": "...",
  "justification_column": "...",
  "paper_says_verified": true,
  "paper_says_evidence": {"paper_id": "...", "page": ..., "verbatim": "..."},
  "implementation_does_verified": true,
  "implementation_does_evidence": {"spec_line": 187},
  "justification_regime": "paper|h200|resume|mixed",
  "justification_audit": "stands|fails|empirical_pending",
  "verdict": "accurate|inaccurate_paper|inaccurate_impl|orphan|missing_row|justification_unsupported",
  "notes": "..."
}
```
A row that says "for H200 reasons" must reference an H200 claim that
audits clean. A row whose justification is `empirical_pending`
(e.g. D5b, D7a with TODO ablations) is flagged informationally — not
a finding, but tracked.

#### `06_findings/findings.jsonl`
```json
{
  "id": "F-0001",
  "regime": "paper|h200|resume|chapter12|cross",
  "severity": "...",
  "category": "uncovered_deviation|misdescribed_ch12|stale_reference|missing_citation|paraphrase_drift|silent_omission|hardware_inconsistency|durability_gap|unsupported_justification|empirical_pending",
  "spec_location": {"line": ..., "section": "..."},
  "evidence": {"paper": {...}, "h200_truth": {...}, "resume_truth": {...}, "ch12": {...}},
  "summary": "...",
  "recommended_fix": "...",
  "source_phase": "..."
}
```

## Phase mechanics

### Resume protocol on a fresh agent turn
1. Read `00_manifest.json`. Missing → init from scratch (record both
   `spec_sha256` and `ch12_sha256`).
2. Validate `spec_sha256` against the current spec file. If changed,
   abort with a warning — the user must restart or accept partial.
3. Find the first non-`done` phase, resume there.
4. Within a phase, use the per-artifact lists in the manifest to skip
   completed units.
5. **Update the manifest after EACH unit, not at phase end.** A
   killed agent leaves a partial valid JSONL plus a manifest that
   correctly reflects what was finished.

### Dispatch shape
- **Phase 1 acquisition**: serial, fast (~10 min).
- **Phase 2 paper extraction**: parallelizable, **one subagent per
  paper** (10 papers in parallel where the harness allows).
- **Phase 3 truth refs**: TWO subagents in parallel (h200, resume).
- **Phase 4 spec indexing**: serial single agent (or split into 4
  parallel subagents writing to disjoint files).
- **Phase 5 paper crossref**: parallelizable, one subagent per paper.
- **Phase 6 h200 crossref**: serial single agent (whole-spec view).
- **Phase 7 resume crossref**: serial single agent.
- **Phase 8 ch12 audit**: serial; reads outputs of 5/6/7.
- **Phase 9 findings merge**: serial.

### Subagent prompt skeleton
```
You are running phase [N] of the heavy spec-compliance audit.
Read `audit/spec_compliance/00_manifest.json` for context.
Your unit of work: [paper_id | spec section | Ch.12 row range | regime].
Inputs you may read: [paths].
Outputs to produce: [artifact paths + schema reference].
Truth references (if applicable): [h200_truth.md | resume_truth.md].
Auto-memory (read-only): ~/.claude/projects/-home-lucas-ai-moe-compress/memory/
You MUST NOT write outside `audit/spec_compliance/`.
Definition of done: [exact file existence + minimum-row condition].
On exit: update the manifest's relevant array, then stop. The
orchestrator dispatches the next unit.
Hard token budget: write JSONL incrementally so a kill mid-run leaves
a valid partial file.
```

## Phase-by-phase recipe

### Phase 1 — Paper acquisition
For each arXiv ID in spec §13:
1. `hf paper download <id>` or `arxiv-downloader <id>`; fallback to
   `curl https://arxiv.org/pdf/<id>.pdf` if those fail.
2. Record sha256 in `extraction_log.txt`.

### Phase 2 — Paper extraction → claims
Per paper subagent:
1. Run `pdftotext -layout source.pdf source.md` (fast, low fidelity)
   AND `marker_single source.pdf` if marker is available (high fidelity
   for tables/equations). Pick whichever produces parseable equations;
   log the choice in `extraction_log.txt`.
2. Read `source.md` and emit `claims.jsonl` per the schema.
   **Cap at 100 claims; minimum 30.** Capture every numbered
   Eq./Algorithm/Theorem/Lemma in methodology and experiments
   sections, plus any hyperparameter table.
3. Update manifest: append `paper_id` to
   `phases.2_paper_extraction.papers_done`.

### Phase 3 — Truth references
Two subagents in parallel:

**H200 truth**:
1. Read every relevant memo in
   `~/.claude/projects/-home-lucas-ai-moe-compress/memory/`
   (start with `MEMORY.md`, follow links). Cite each fact you
   reproduce in `h200_truth.md` with a `<!-- source: ... -->` comment
   pointing to the exact memo file.
2. Cross-check NVIDIA H200 datasheet via WebFetch for HBM capacity,
   bandwidth, FP8/FP16/BF16 support. Cite URL and fetch date.
3. Cross-check HF Jobs docs via WebFetch for default timeout, host
   RAM per flavor, bucket semantics.
4. Output `h200_truth.md` as a flat list of `- fact <!-- source: X -->`
   lines (one fact per line so subsequent crossref can grep it).

**Resume truth**:
1. Read auto-memory entries about HF Jobs durability, bucket
   non-durability, Hub upload semantics. Cite each.
2. Document the canonical local-FS atomic-write idiom (write to .tmp,
   rename, fsync) and POSIX rename guarantees (cite man page or
   POSIX.1 spec).
3. Output `resume_truth.md` in the same flat-fact format.

Mark phase done in manifest.

### Phase 4 — Spec indexing
Single agent (or split 4 ways):
1. **citations.jsonl** — `grep -nE "Eq\.|§|Algorithm|Theorem|Lemma|paper [0-9]"` on the spec; parse each match.
2. **formulas.jsonl** — every code-fenced block in the spec.
3. **h200_claims.jsonl** —
   `grep -inE "VRAM|GB|H200|Hopper|FP8|BF16|FP16|FP32|torch\.compile|FlashAttention|batch_size|seq_len|wall.?time|min(ute)?s|hour|kernel|cudnn|tf32|HBM|HF Jobs|timeout|bucket"`
   plus contextual scan (read each match's surrounding paragraph).
   Emit one row per hardware-or-perf assertion.
4. **resume_claims.jsonl** —
   `grep -inE "atomic|resume|crash|idempot|rename|\.tmp|sha256|format_version|durab|bucket|Hub commit|snapshot|spill|reentry"`
   plus a full read of §11 and §10.
5. **Sanity checks** (each emits findings of category=`missing_citation`
   or `dangling_reference`):
   - Every cited `<paper_id>` appears in §13.
   - Every Ch. 12 row's deviation type matches the regime its
     justification implies (paper-justified → must cite paper section;
     h200-justified → must cite H200 fact; resume-justified → must
     cite durability fact).

### Phase 5 — Paper crossref
Per paper subagent (parallel):
1. Read `claims.jsonl` and the spec.
2. For each claim, search the spec (grep) for any reference to it.
3. Verdict per the schema. When verdict is `deviation_in_ch12`, also
   record the sanction regime by reading the Ch. 12 row's
   justification text:
   - Justification cites paper-internal reasoning →
     `sanction_justification: "ch12"` (self-contained)
   - Justification cites VRAM/wall-time/H200/HF-Jobs →
     `sanction_justification: "h200"` (must be confirmed in phase 6)
   - Justification cites durability/resume →
     `sanction_justification: "resume"` (must be confirmed in phase 7)
4. Append non-`match` rows to `findings.jsonl` with regime=`paper`.

### Phase 6 — H200 crossref
Serial agent:
1. For each row in `h200_claims.jsonl`, look up the claim's facts
   against `h200_truth.md` (grep facts; quote inline).
2. Verdict per the schema.
3. **Cross-cutting check**: for every paper crossref row whose
   `sanction_justification` was `h200`, find the H200 claim that
   backs it. If absent or `inconsistent` → emit a `cross` finding
   with category=`unsupported_justification`.
4. Append non-`consistent` rows to findings with regime=`h200`.

### Phase 7 — Resume crossref
Serial agent:
1. For each row in `resume_claims.jsonl`, look up the implied
   invariant against `resume_truth.md`.
2. Verdict per the schema.
3. **Cross-cutting check**: same as phase 6 but for
   `sanction_justification: "resume"`.
4. Additional integrity checks:
   - Every stage's "complete" definition references both payload
     and manifest existence (or equivalent).
   - Every "atomic write" claim is backed by a `.tmp + rename` truth
     fact OR a Hub-commit truth fact.
   - Bucket-write claims must be flagged as non-durable per
     auto-memory unless followed by a Hub-commit claim.
5. Append non-`consistent` rows to findings with regime=`resume`.

### Phase 8 — Chapter 12 audit
Serial agent:
1. Parse Ch. 12 from spec.
2. For each row:
   - Verify "Paper Says" verbatim against the cited paper's
     `source.md`.
   - Verify "Implementation Does" against the spec body section.
   - **Classify the justification** into `paper|h200|resume|mixed` by
     keyword scan (VRAM/wall-time/HF-Jobs → h200, durability/resume
     → resume).
   - **Audit the justification** against the appropriate truth ref:
     - `paper` → already verified at row-level
     - `h200` → consult `h200_truth.md`
     - `resume` → consult `resume_truth.md`
     - `mixed` → both
   - Verdict per the schema.
3. **Detect missing rows**: any spec behavior in §4–§9 marked as a
   deviation in phases 5/6/7 but with no corresponding Ch. 12 row →
   emit `missing_row` finding.

### Phase 9 — Findings merge
Serial agent:
1. Concatenate all findings.
2. Deduplicate by `(spec_location, category, regime)`.
3. Sort by severity, then by spec line.
4. Render `findings.md` with four sections: Paper-regime, H200-regime,
   Resume-regime, plus a Cross-cutting section for `regime=cross`.
5. Each finding embeds the verbatim evidence quotes.
6. Mark all phases `done` in manifest.

## Definition of overall done
- All phases `done` in `00_manifest.json`.
- For each paper in §13: `claims.jsonl` exists with ≥30 entries and
  `<id>.crossref.jsonl` exists.
- Both truth refs exist with ≥10 cited facts each.
- `h200_claims.jsonl` and `resume_claims.jsonl` each have ≥10 entries
  (the spec is rich enough that fewer indicates incomplete indexing).
- `ch12_rows.jsonl` covers every Ch. 12 row plus any `missing_row`
  entries.
- `findings.md` exists with the four sections above (sections may be
  empty but must be present).

When all true, print:
`AUDIT COMPLETE — see audit/spec_compliance/06_findings/findings.md`

## Known failure modes and defenses
- **Context exhaustion mid-paper**: per-line JSONL writes; manifest
  updates after each unit. Killed agent leaves a partial valid JSONL.
- **PDF extractor disagreement**: prefer marker for equations,
  pdftotext for prose; log choice in `extraction_log.txt`.
- **Auto-memory drift**: phase 3 records each cited memo's sha256
  alongside its facts; later phases can detect mid-audit drift.
- **Spec changes during audit**: phase 1 records `spec_sha256`. Every
  fresh-turn resume validates it. Mismatch → abort with warning.
- **Hardware truth ambiguity**: if an H200 fact can't be sourced,
  emit verdict=`unverifiable` (not `consistent`); defer to human.
- **Justification cycles**: if a paper deviation cites H200 and the
  H200 claim cites the paper deviation as motivation, flag as
  cross-cutting circular justification (FINDING, severity=medium).

## What to do if you find something genuinely ambiguous
Emit a finding with category=`info`, severity=`minor`, document both
readings in `notes`. The audit produces evidence; humans relitigate.

## Final commit
At end of phase 9, commit everything under `audit/spec_compliance/`
locally:
```
audit(spec): heavy three-regime compliance audit, phases 1–9 artifacts
```
No spec edits in this commit. Fix-up is a separate task.

---

**Start by reading `audit/spec_compliance/00_manifest.json`. If it
doesn't exist, you're starting from scratch: create it, initialize all
phases to `pending`, record both `spec_sha256` and `ch12_sha256`, and
begin phase 1. If it exists, find the first non-`done` phase and
resume there. Do not re-do completed phases.**

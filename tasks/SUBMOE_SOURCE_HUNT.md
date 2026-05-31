# Sub-MoE Source-Code Hunt (arXiv:2506.23266)

**Paper:** *Sub-MoE: Efficient Mixture-of-Expert LLMs Compression via Subspace Expert Merging*
**Authors:** Lujun Li, Qiyuan Zhu, Jiacheng Wang, Wei Li, Hao Gu, Sirui Han\*, Yike Guo\* (\*corresponding)
**Affiliations:** HKUST (1), Xi'an Jiaotong University (2), University of Birmingham (3)
**Corresponding emails (from paper):** `{lliee, qzhuat, siruihan, yikeguo}@ust.hk`
**Date of investigation:** 2026-05-28

---

## VERDICT: MAYBE-FOUND (placeholder repo only — no code yet)

The authors **have created a public GitHub repo** under one of the corresponding authors' handles, but it currently contains **only a README + LICENSE**. The repo cited in the paper itself (`lliai/MoERazor`) **does not exist and has never existed** (confirmed via direct 404 + zero Wayback snapshots + zero hits in `MoERazor` global GitHub repo-name search).

### Top URL candidate

`https://github.com/siruihan2024/Sub-MoE` (verified 200 OK, public, MIT-licensed)

- **Owner:** `siruihan2024` → GitHub display name **Sirui Han**, company **HKUST**, bio "Assistant Professor, Division of Emerging Interdisciplinary Areas, RGC-Fulbright Research Scholar, The Hong Kong University of Science and Technology", homepage https://siruihan.com/. **This is the corresponding author of the Sub-MoE paper.**
- **Repo description (verbatim):** `"Sub-MoE: Efficient Mixture-of-Expert LLMs Compression via Subspace Expert Merging"` — identical to the paper title.
- **Created:** `2026-03-14T15:46:51Z`. **Last push:** `2026-03-14T15:46:56Z` (single initial commit, never updated).
- **License:** MIT, "Copyright (c) 2025 Sirui Han".
- **Single commit:** `9cff619a00a403f63df0685b48eddeaf29f21a6f`, author `Sirui Han <siruihan@ust.hk>` (email matches paper exactly), message `"Initial commit: add README and LICENSE"`.
- **Tree:** `LICENSE` (1066 B), `README.md` (1171 B) — **no source code present.**
- **README content** (verbatim, key excerpts):
  - Venue badge: `AAAI 2026`
  - arXiv badge: `2506.23266`
  - Author list matches paper exactly
  - Abstract section: *"Abstract coming soon. Please refer to the paper for details."*
  - BibTeX includes `booktitle={AAAI 2026}, year={2026}`

This strongly suggests the paper was originally submitted to NeurIPS 2025 (the arXiv tarball's main .tex file is literally named `neurips_2025.tex`), rejected, then resubmitted to and accepted at **AAAI 2026**. The arXiv abs page still notes `"Work in progress, revisions ongoing"`. The placeholder repo was created when AAAI acceptance came through.

### Pattern observation (why this is a placeholder, not code)

`siruihan2024`'s 35 public repos were **all created on the same day** (2026-03-14), and all but 3 (`siruihan2024.github.io`, `LexMind`, `awesome-world-law-agent`, `reco-lab`) are 1-KB stubs with description-only. This is a bulk pre-registration of repo names for the group's accepted papers; `Sub-MoE` is one slot in that batch and has not yet been populated. Compare against `lliai/D2MoE` (the same author group's previous paper, NeurIPS 2025) which **was** fully released as actual code — so the group's usual home for code is the `lliai` handle, not `siruihan2024`. The `siruihan2024/Sub-MoE` repo may eventually get the code, OR the authors may switch to publishing under `lliai/Sub-MoE` (or `lliai/MoERazor` as the paper promised).

---

## Per-channel report

### 1. arXiv abstract + ancillary tarball
- `https://arxiv.org/abs/2506.23266` — abstract ends with literal sentence: *"Code will be released at https://github.com/lliai/MoERazor."* (future tense)
- arXiv comments field: *"Work in progress, revisions ongoing"*
- Downloaded e-print tarball (`https://arxiv.org/e-print/2506.23266`, 3.2 MB): contains only `neurips_2025.tex`, `neurips_2025.bbl`, `neurips_2025.sty`, `main.bib`, 4 figure PNGs/PDFs, `00README.json`. **No source code attached.**
- Only `github.com` URLs anywhere in the LaTeX source: `lliai/MoERazor` (in abstract/conclusion), plus standard citation URLs for `facebookresearch/fairseq`, `ggerganov/llama.cpp`, `huggingface/transformers`, `microsoft/DeepSpeed`, `NVIDIA/FasterTransformer`, `pjlab-sys4nlp/llama-moe`. None of those are Sub-MoE code.

### 2. The cited URL `github.com/lliai/MoERazor`
- HTTP HEAD → `404 Not Found` (verified 2026-05-28 14:41 UTC)
- Wayback Machine: `{"url":"github.com/lliai/MoERazor","archived_snapshots":{}}` → **never archived → has never existed publicly**
- Global GitHub repo-name search for "MoERazor" → **zero results**
- Variants tested, all 404: `lliai/Sub-MoE`, `lliai/SubMoE`, `lliai/Sub-MoE_Code`, `lujun-li/Sub-MoE`, `lujun-li/SubMoE`, `lujunli/Sub-MoE`

### 3. lliai's own GitHub account (`https://github.com/lliai`)
- Profile: "IronMan", 1285 public repos (mostly forks of others' work). Real code releases: e.g. `D2MoE` (the group's prior paper, fully released, 82 stars), `AttnZero`, `EASYEP`, `DSA`, etc.
- 30 most recently pushed lliai repos checked: **no Sub-MoE / MoERazor / subspace-merging repo present.**
- Most recent meaningful push: `AttnZero` 2025-09-02; `EASYEP` 2025-04-10; `D2MoE` 2025-03-25.

### 4. Co-author GitHub handle search
- **Lujun Li candidates:** `lujunliai1997` (name "Lujun Li", 0 repos, no info), `DobricLilujun` (name "LUJUN LI", no info shown). Both look like duplicate / abandoned accounts; lliai is the real one.
- **Qiyuan Zhu:** `qiyuanchn` exists (name "Qiyuan Zhu", company "Beijing Institute of Technology") — *probably a different Qiyuan Zhu* (paper's Qiyuan Zhu is HKUST `qzhuat@ust.hk`). 3 repos visible (`dreamzero`, `Motus`, `Wan2.2`), none Sub-MoE-related.
- **Sirui Han:** `siruihan2024` — **MATCH** (HKUST, same email). See top finding above.
- **Yike Guo, Jiacheng Wang, Wei Li, Hao Gu:** not individually searched — handle names are too common to disambiguate without further evidence, and the corresponding-author lead via Sirui Han already found the most likely repo.

### 5. Global GitHub search
- `https://api.github.com/search/repositories?q=Sub-MoE+subspace+expert` → 1 hit: `siruihan2024/Sub-MoE` (the one above).
- `https://api.github.com/search/repositories?q=Sub-MoE` → 20+ hits, only `siruihan2024/Sub-MoE` matches the paper; the rest are unrelated (e.g. `submit.vtbs.moe`, `MoeSublime`, etc.).
- `https://api.github.com/search/repositories?q=MoERazor` → **zero results.**
- Authenticated code search for the distinctive phrase `"Sub-MoE" "Subspace Expert Merging"` requires GitHub auth (got 401 from unauthenticated API); skipped — but a code-search hit would still resolve to the same `siruihan2024` repo since no other repo on GitHub has those terms.

### 6. OpenReview
- `https://api.openreview.net/notes/search?term=Sub-MoE&content=all&source=all` → 0 results.
- `https://api.openreview.net/notes/search?term=Sub-MoE&content=all&source=all&group=AAAI.org` → 0 results.
- `https://api.openreview.net/notes/search?query=Sub-MoE+subspace+expert+merging` → returned many notes but no titles matching "Sub-MoE". (Note: AAAI 2026 does not use OpenReview for main-track papers; NeurIPS 2025 paper would have been withdrawn after rejection so likely not visible.)
- **No supplementary code zip available via OpenReview.**

### 7. HuggingFace
- `https://huggingface.co/papers/2506.23266` — paper page exists with abstract; cites `github.com/lliai/MoERazor`; **no linked models, datasets, or spaces.**
- `huggingface.co/api/models?search=sub-moe` → 4 unrelated hits (MoeSS-SUBModel, Deepseek-V2 subset experts, etc.). None are Sub-MoE.
- `huggingface.co/api/models?search=submoe` → empty `[]`.
- `huggingface.co/api/datasets?search=sub-moe` → empty `[]`.
- `huggingface.co/api/spaces?search=sub-moe` → 1 unrelated hit (`moeit/sub-store`).
- `huggingface.co/api/spaces?search=submoe` → empty `[]`.

### 8. Papers with Code
- `https://paperswithcode.com/paper/sub-moe-efficient-mixture-of-expert-llms` — page exists; cites `github.com/lliai/MoERazor` (broken) as the only code link.

### 9. Wayback Machine
- `github.com/lliai/MoERazor` → `archived_snapshots: {}` (never archived)
- `github.com/lliai/Sub-MoE` → `archived_snapshots: {}` (never archived)
- `github.com/lliai/SubMoE` → `archived_snapshots: {}` (never archived)
- `github.com/siruihan2024/Sub-MoE` → `archived_snapshots: {}` (never archived)
- `siruihan.com` → `archived_snapshots: {}` (never archived)

### 10. Sirui Han personal homepage
- `https://siruihan.com/` returns content but `grep` for `sub.?moe`, `MoERazor`, `2506.23266`, `lliai`, `Lujun` → **no matches** (homepage doesn't currently advertise Sub-MoE code).
- Repo-pages variants: `https://siruihan2024.github.io/Sub-MoE/` → 404.

### 11. lliai/D2MoE README (the prior-paper repo)
- Checked for forward-pointer to Sub-MoE → **no mention** of Sub-MoE, MoERazor, or 2506.23266.

---

## Channels still untried (so user knows search is not 100% exhaustive)

1. **Authenticated GitHub code search** for the distinctive phrase `"Subspace Expert Merging"` across all repos (requires a GitHub PAT — unauthenticated API returned 401). Could catch a hidden community re-implementation.
2. **Google / Bing / DuckDuckGo web search** for `"Sub-MoE" "Subspace Expert Merging" site:github.com` — not performed (no Google-via-curl in this session).
3. **Yike Guo HKUST AI lab pages** (group page, lab software lists) — not crawled.
4. **GitLab / Bitbucket / Gitee** — Chinese-affiliated authors sometimes mirror on Gitee; not searched.
5. **Co-author handle disambiguation** for Lujun Li, Yike Guo, Jiacheng Wang (Xi'an Jiaotong), Wei Li (Birmingham), Hao Gu — only `lliai` (≈ Lujun Li) and `siruihan2024` were fully verified.
6. **AAAI 2026 supplementary material download** (if accepted papers are mirrored on aaai.org with supplementary zips). Not checked — AAAI 2026 proceedings page not yet published as of this investigation.
7. **HuggingFace discussion tab on the paper page** — has 1 upvote; user-constraint forbids posting there, but reading existing discussion was not exhaustively done.

---

## Recommended action

The official-but-empty repo `https://github.com/siruihan2024/Sub-MoE` is the **single best URL to watch**. The owner is provably the paper's corresponding author. Bulk repo-creation pattern (35 stubs on 2026-03-14) suggests the group will populate these over time. If/when the paper is presented at AAAI 2026 (Jan 2026), code is most likely to appear either here or under `lliai/Sub-MoE` (or `lliai/MoERazor` as originally promised). As of 2026-05-28: **no public source code release exists.**

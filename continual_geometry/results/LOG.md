# Build / decision log

Append-only. Newest entries at the bottom of each day.

---

## 2026-08-10

### Step 0 — Chou v2 PDF integrity check (per 05 brief, pre-reconciliation)

`papers/Feature Learning beyond the Lazy-Rich Dichotomy - Insights from
Representational Geometry.pdf` (re-downloaded arXiv:2503.18114v2):

- size = 10,494,734 bytes — **not** the truncated 524,288 ✓
- trailer = `%%EOF` present ✓
- extracts via `gs -sDEVICE=txtwrite` (233,239 chars) ✓
- algorithm captions confirmed from the complete PDF:
  - Algorithm 1 = "Estimate simulated manifold capacity" (α_sim bisection)
  - Algorithm 2 = "Estimate manifold capacity and effective geometric measures"
    (the geometric estimator; α, R, D, ρ_c, ρ_a, ψ)

**PASS.** D1 relabel (Algorithm 1 → Algorithm 2 for the geometric estimator) is
now confirmed against the primary PDF, not only the arXiv HTML. Reconciliation
may proceed.

### Reconciliation pass (05 §1) — files changed

Single logical change-set (the "one commit" of 05 §1.2). No code, docs only.

- **04-spec-corrections-iclr2026.md** — dated erratum header (Algorithm 1→2 for
  the geometric estimator); body left unchanged as a correction-log record (§1.1).
- **00-math-spec.md**
  - §6 heading → Algorithm 2; NB that Algorithm 1 = α_sim.
  - §6.1 → rewritten in stacked-matrix (P-space) form: anchors as rows of
    `S∈R^{P×N}`, `S_y=diag(y)S`, exact `a/b/c` scalars with **P×P** pseudo-inverses;
    measures α, D_eff, R_eff, Ψ_eff; ρ_c/ρ_a/ψ (abs, unnormalized, cross-index, C3);
    added `rho_c_signed` H1d instrument and `rho_convention`.
  - §6.2 → **exact** three-factor identity `α=Ψ_eff·(1+R_eff⁻²)/D_eff`; UNCERTAIN
    marker removed; algebraic closure shown.
  - §6.3 → `raw`/`standardized` → `raw`/`gaussianized` (Wakhloo 2023).
  - §7 → generic / **retained** / **tilted(β)**; "aligned" retired; D2 staging.
  - §8 → **log-space additive** attribution, zero residual (3 factors).
  - §13 → cost model re-derived for P×P linear algebra; pairwise vs full_P note.
- **02-validation-suite.md** — §2 collapsed to the single §2a gate (mean-field
  validity) with the **β axis** added (P×N×β; β=∞ ≡ retained); §2b deleted
  (exact identity); §2c Ψ_eff identity check (GLUE-dependent) added; noise-floor
  list updated (α_retained, Ψ_eff, R_eff/D_eff, rho_c_glue/rho_c_signed).
- **01-experiments.md** — config: `estimation_mode`, `rho_convention`, `beta`,
  `ccgp_enabled`, `capacity_modes: [raw, gaussianized]`; Phase 0 pairwise-vs-full_P
  row; deleted-§2b reference fixed; pre-reg H1a (log-space shares), H1b, H2b/H2c,
  Fig 3 → retained; §3.3 P=2 restated as task-structure.
- **AGENTS.md** — §3 terminology table: retained/tilted, psi_eff vs
  center_axis_alignment, rho_c_glue/rho_c_signed; "aligned" retired; §6 approximation
  bullet → exact-identity resolved.
- **PROJECT.md** — P1 restated (task-structure, not estimability); ICML entry →
  Algorithm 2; ICLR-2026 added as load-bearing (§5.1) and as a §15 objection (C7
  distinctions); §6.6/§13/§8 estimability + approximation wording updated;
  "aligned"→retained, "standardized"→gaussianized throughout.
- **03-references.md** — version-drift note corrected; ICML entry → Algorithm 2;
  ICLR-2026 row added to §A/§H; §E.4 pseudocode col → Algorithm 2 and convention
  **reversed to C3** (rho_c_glue primary); §E.5 marked resolved by stacked-matrix
  form; §E.6 downgraded (retained is legitimate; tilt is enhancement); §D/§F add
  ood-generalization-geometry (reference-only) and glue-decomposition.md; §G
  resolved-numbering corrected; §B Montanari phrasing.
- **reference/README.md** — `Version:` field added to the provenance template;
  status table updated (glue-decomposition added, glue-algorithm→Alg2,
  glue-refinements status).
- **reference/{manifold-generator, validation-settings, parameterization,
  cl-metrics, glue-refinements}.md** — `Version:` field added; manifold-generator
  naming flag marked RESOLVED; glue-refinements A2↔C4 non-conflict cross-ref line.
- **reference/glue-decomposition.md** — NEW agent-drafted sheet (ICLR 2026 §B.3,
  exact 3-factor identity + a/b/c), awaiting human verify.

Verified vs inferred: all scientific substitutions trace to sources verified
2026-08-10 (Chou v2 PDF Step-0 check; ICLR 2026 §B.3 for the identity/a-b-c). New
reference sheets carry blank `Verified by (human)` and may not be cited by code
until signed off.

### Git — initial import + reconciliation (two commits, into the parent monorepo)

Repo root is the **parent monorepo**
`/home/kat/workspace/Structure-Function-Analysis-of-Network-Topologies` (holds
unrelated projects). `continual_geometry/` was **entirely untracked**; verified
no nested `.git`, no leftover `.git` in `third_party/`, subproject not ignored.
Committed as two objects for legible history (per Kati's steer):

- `32b8b65e` — **import vendored deps + gitignore**: `third_party/` source
  (replicaMFT, correlated_capacity, VENDORED.md, patches/), `papers/.gitignore`,
  and a new `continual_geometry/.gitignore`.
- `2099b0af` — **spec reconciliation (05 §1)**: docs-only + `src/glue/adapters/README.md`
  + this log.

**Gitignore conflict resolved:** the parent `.gitignore` has a blanket `results`
rule that would drop `continual_geometry/results/`. Added an **anchored**
`!/results/` in `continual_geometry/.gitignore` so this project's audit trail
(LOG.md, future *.json evidence) is tracked while vendored
`third_party/**/results/` artifacts (~78 .npy/CSV) stay ignored. `papers/` PDFs
(48 MB) remain ignored.

**Deferred (not part of §1):** vendoring `ood-generalization-geometry` (Gaussianization)
moved to §2.2 (its own commit, when `preprocessing.py` is the active task) — it is
figure-repro code needed only at preprocessing, and keeps the reconciliation diff clean.

*(This entry post-dates the two commits above; fold it into the next daily
checkpoint commit — §8.)*

---

## 2026-08-11

### Amendment before §2 — ρ_c two-role rule + H1d flag-and-stop

Per Kati ratification: **"primary" is two jobs**. `rho_c_glue` = reporting /
comparability with published GLUE; `rho_c_signed` = **required H1d instrument**
(under `|·|`, H1d is untestable). `rho_convention: both` is **mandatory** — no
path may reduce to one. Applied in `03` §E.4 (already had the split), `00` §6.1,
`01` config + H1d pre-reg (already named `rho_c_signed`), `AGENTS` §3, `05` §2.2
and §6 item 1, `glue-decomposition.md`.

**Escalation live:** if `manifold_analysis_corr` does not expose anchor centers
`s⁰_μ`, that is flag-and-stop (adapter extraction vs `third_party/patches/` vs
drop H1d) — **not** degradation to abs-only.

### Spot-check (load-bearing reconciliation)

1. `00` §6.2 — exact identity present; UNCERTAIN gone from capacity approx;
   two-factor only mentioned historically.
2. `02` — §2b deleted; §2a has β axis; no live cross-ref treating §2b as a gate.
3. "aligned capacity/margin" retired in AGENTS/00/01 pre-reg; remaining hits in
   `04` (correction log) and `05` (historical apply-order) only; PROJECT leftovers
   cleaned.
4. `00` §8 — log-space additive, zero residual; Wakhloo caveat (report ρ_a with R)
   intact.
5. `glue-decomposition.md` — `Verified by (human)` still blank.
6. `04` erratum header present; body unchanged.

### Gitignore confirm

`git check-ignore -v continual_geometry/results/LOG.md` → exit 1 (not ignored).
Probe JSON `results/_gitignore_probe.json` appeared in `git status` (then removed).

### §2 start — `src/manifolds/`

Scaffolded: `generator.py`, `labeling.py`, `dichotomies.py`, `streams.py`,
tests under `tests/`, `pyproject.toml` + `requirements.txt`. Local `.venv` via
`uv` (gitignored). **`pytest`: 15 passed** (`02` §5 stream/dichotomy suite +
unit-norm + ρ_C monotonicity).

### replicaMFT capability audit — three findings, one structural

Triggered by the §2.2 verify-first anchor-center check. Read
`mftma/manifold_analysis_correlation.py` end to end. Framing per Kati: these are
**properties of the replica mean-field theory that GLUE refines**, not bugs.

- **F2 (structural, decisive).** `manifold_analysis_corr` takes no `y` because the
  dichotomy average is integrated out *analytically* in the replica derivation —
  `y` was eliminated before the code was written. GLUE samples `(y, t)` and
  averages *numerically*, which is precisely why it supports an analyst-chosen
  ensemble `Y`. ⇒ `α_mf` gives **generic capacity only, permanently**. Retained,
  tilted, and per-task attribution are not extractable at any effort. This is
  assumption A2 of `glue-refinements.md`, seen in code.
- **F1 (spec error we were carrying).** `res_coeff0` = mean over `μ≠ν` of the
  **absolute cosine** between global-mean-subtracted **manifold** centers (raw
  class means) in a `P−1` basis. Abs *and* normalized, on manifold centers not
  anchor centers ⇒ matches **neither** `rho_c_glue` nor `rho_c_signed`.
  Renamed `center_cos_abs`; `00` §6.1 corrected.
- **F4 (follows from F2).** `R_M`/`D_M` are the PRX-2018 eq-28/29 functionals, not
  `R_eff = √(E[c]/E[b−c])` / `D_eff = E[b]/P`. The D3 `Ψ_eff` identity is therefore
  **expected to fail**; reclassified gate → diagnostic (`02` §2c).
- Also: anchor centers are formed (`s_all.mean(axis=1)`) then discarded, in a
  per-manifold frame (centering, norm division, per-manifold QR re-basis when
  `N > M`) ⇒ cross-manifold `⟨s⁰_μ, s⁰_ν⟩` unrecoverable from returns.

**Decision (Kati): build the GLUE core ourselves.** New `src/glue/core.py`,
estimator string `glue_core@<sha>`, firewalled like any other estimator. Scope:
α, D_eff, R_eff, Ψ_eff, anchor ρ_c in **both** conventions. Not ρ_a / ψ_{μν}.
Nothing in `third_party/` modified — clears the firewall. Hard deadline: validated
against B.5 recovery by **end of Day 14**, else Figure 2 falls back to the
rotation/expansion decomposition (principal angles vs ΔPR).

### Spec corrections 1–5 applied (pre-code commit)

1. `00` §6.1 — replicaMFT capability block: label-invariance, `res_coeff0` ≠ ρ_c,
   `R_M`/`D_M` ≠ `R_eff`/`D_eff`, anchor centers discarded.
2. `00` §7 estimator-routing table + `01` Phase 0 — `α_mf` generic-only; retained
   and tilted route through `α_sim` and `α_core`.
3. `02` §2a — β axis is `α_sim` vs `α_core`; `α_mf` participates at β = 0 only.
   §2c reclassified to diagnostic.
4. Harmonic-mean α aggregation pinned in `00` §6.1, `01`, `AGENTS` §4.
5. `third_party/VENDORED.md` — capability limit recorded as a verified finding;
   replicaMFT downgraded to generic-α cross-check; `glue_core` added to the
   version-string table.

### FLAG — technical correction to the build route (needs Kati's eye, not a stop)

The instruction says anchor points "via `maxproj` composed **per-manifold** in the
ambient frame". Composing per-manifold reproduces exactly the structure that makes
`α_mf` label-invariant, so it would silently re-inherit the limitation we are
escaping. Reason: in ICLR B.3, `S_y = diag(y)·S`, and `diag(y)` **cancels
algebraically** in all three quadratic forms —

```
a = (S_y t)ᵀ (S_y S_yᵀ)† (S_y t) = (S t)ᵀ (S Sᵀ)† (S t)     since diag(y)² = I
```

so `a`, `b`, `c` are invariant to `y` *given* `S`. **All** `y`-dependence must
therefore enter through the anchor points themselves, which it does only if they
come from the **joint** QP over all `P·M` constraints with rows `y_μ z^μ_j`
(Algorithm 2 / `00` §6.1) — where the shared separating direction couples the
manifolds. Implementing joint; `maxproj`/`minimize_vt_sq` still usable for the
`P = 2` cross-check. No vendored code touched either way.

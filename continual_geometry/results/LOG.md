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

### `src/glue/core.py` built and **validated** — both targets pass

Joint anchor QP at κ=0: `min ½‖v−t‖²  s.t. Gv ≤ 0`, rows of `G` = `y_μ z^μ_i`.
KKT gives `v = t − Gᵀλ`, so the duals are the NNLS problem
`λ = argmin_{λ≥0} ‖Gᵀλ − t‖²` — solved with `scipy.optimize.nnls`, no cvxopt and
no vendored code in the path. Anchors are the unsigned dual-weighted averages
`Σλ z / Σλ` per `00` §6.1.

**Convention resolved by a hard check, not by choice.** A manifold with zero dual
mass gets a **zero row**. That reproduces the rectification in the replica
expression `α = 1/E[(⟨t,ŝ⟩)₊²]`: for `P` orthonormal point manifolds ~half are
active, `E[a] = P/2`, so `α = 2` — the known κ=0 point-manifold capacity.
`test_point_manifold_capacity_is_two` measures **α = 2.00 ± 0.12**, active
fraction 0.50. This is the load-bearing test for the whole formulation.

**Also found and fixed:** ensembles must use an RNG stream for `y` independent of
the one for `t` (`rng.spawn(2)`), or comparisons across β differ by Monte-Carlo
noise on `t` as well as by `Y`. At `n_t = 150` that noise is ~10% of α — larger
than the effects being compared. Tilt-β=50 and retained now agree to <1%.

**Gate 1 — §B.5 ground-truth recovery** (P=2, M=200, N=1000, n_t=200, 3 seeds;
`results/glue_core_recovery.json`, 162 s). Recovery is close to identity on all
three axes, with clean cross-axis separation:

| ground truth | 2 / 0.8 / 0.0 | 4 / 1.0 / 0.2 | 6 / 1.4 / 0.4 | 8 / 1.7 / 0.6 | 10 / 2.0 / 0.8 |
|---|---|---|---|---|---|
| `D_eff` vs D | 2.25 | 4.07 | 5.60 | 6.98 | 8.17 |
| `R_eff` vs R | 0.87 | 1.02 | 1.34 | 1.56 | 1.77 |
| `rho_c_glue` vs ρ_C | 0.04 | 0.22 | 0.41 | 0.61 | 0.81 |

`D_eff` is pinned at 4.07–4.08 across **both** the R and ρ_C sweeps — the axes are
separable. `α` falls monotonically on all three. `Ψ_eff ∈ [0.85, 0.91] ⊂ [0,1]` ✓.
`identity_residual < 1e-8` everywhere (exact by construction, so this is a
transcription check on the file).

**F4 resolved with a number.** The cross-sheet ambiguity — does axis/center
correlation land in `R` or in `D`? — is now measured at our parameters: sweeping
ρ_C from 0 → 0.8 moves `R_eff` 1.02 → 1.82 and leaves `D_eff` at 4.07 → 4.08.
**Correlation is absorbed entirely by the radius**, as Wakhloo's duality predicts,
not by dimension. §8 attribution must keep reporting ρ_c alongside R.

**Gate 2 — α channel vs `α_sim`.** P=8, d=200, D=4, R=1, M=60:
`α_sim = 0.471` vs `α_core = 0.433` (8%, and `α_sim` is coarse here — `N_c = 17`).

**Deadline met on Day 3, not Day 14.** The rotation/expansion fallback for
Figure 2 is not needed.

### `src/glue/adapters/simcap.py` — and a correction to the α_sim assumption

**`manifold_simcap_analysis` is generic-only as shipped.** `compute_sep_Nc_general`
(`:180`) draws its *own* random balanced labels internally, so the vendored entry
point cannot produce retained or tilted capacity either — the "α_sim runs a real
SVM with real labels" claim holds for `check_data_separability_general` (which
takes explicit labels and *is* public), not for the entry point above it. The
adapter therefore owns the bisection: injected `Ensemble`, injected
`numpy.random.Generator` (which also removes the global `np.random.seed` violation
of `AGENTS.md` §4), vendored SVM call unchanged. Nothing in `third_party/` touched.

Figure 3 is still safe — but via this adapter, not via the vendored function.

### `02` §2c Ψ_eff diagnostic — ran, mismatched as predicted

`results/psi_eff_diagnostic.json`, n_t = 200:

| | P=8 | P=16 |
|---|---|---|
| `Ψ_eff` via replicaMFT identity | 0.694 | 0.665 |
| `Ψ_eff` direct `E[c]/E[a]` (core) | 0.772 | 0.768 |
| relative gap | **10.1%** | **13.4%** |
| `D_M` vs `D_eff` | 3.26 / 3.70 | 3.32 / 3.58 |
| `R_M` vs `R_eff` | 1.069 / 1.001 | 0.967 / 0.992 |
| `α_mf` vs `α_core` | 0.399 / 0.417 | 0.415 / 0.433 |
| `center_cos_abs` vs `rho_c_glue` / `rho_c_signed` | 0.143 / 0.044 / 0.025 | 0.074 / 0.040 / 0.000 |

Confirms F4 (the PRX functionals are not the GLUE ones) and F1 (`res_coeff0` is a
third quantity, ~3× `rho_c_glue` here). Note `α_mf` and `α_core` agree to ~4% on
the **generic** channel — exactly the cross-check role replicaMFT now has.

### Environment

Added `scipy` (core dependency). Vendored estimators need `cvxopt`, `autograd`,
and **`pymanopt==0.2.5`** — replicaMFT imports `pymanopt.solvers`, removed
upstream in pymanopt 1.0, so that pin must not be raised. Recorded in
`requirements.txt` and as the `vendored` extra in `pyproject.toml`.
**`pytest`: 30 passed.**

### Still open

- `preprocessing.py` and ood-geometry vendoring remain deferred, per instruction.
- `center_policy` (`"all"` vs `"active"` when averaging anchors over samples where
  a manifold is inactive) is a flagged convention; `"all"` is the literal §B.3
  reading and passes recovery, so it stays default. Recorded in the result key.
- ρ_a and ψ_{μν} deliberately out of scope.

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

---

### P1 — cost model: Phase 1 fits in **2.8 h wall**, no cuts needed

`scripts/run_cost_model.py` → `results/cost_model.json`. Machine: 256 logical
cores (2 × AMD EPYC 7763), 503 GB. Workers pinned to 1 BLAS thread each.

**The 18 s/eval figure was a multithreaded-BLAS number.** Pinned to one thread the
base point (P=16, M=150, N=300, n_t=200) costs **52.4 s/eval**. That is the right
number to plan with, because throughput is set by total core-seconds, not by
per-eval latency, and one process per core beats few processes with threaded BLAS.

Grid as specified — T=16, Tier-2 every 4th boundary → 40 evals/run;
6 γ × 4 conditions × 10 streams × 5 seeds = 1,200 runs = **48,000 evaluations**:

| `n_t` | s/eval | serial | wall @248 workers | α CV |
|---|---|---|---|---|
| 50 | 13.1 | 175 h | 0.70 h | 3.23% |
| 100 | 26.2 | 349 h | 1.41 h | 2.82% |
| **200** | **52.4** | **699 h** | **2.82 h** | **1.87%** |
| 400 | 104.8 | 1397 h | 5.63 h | 1.03% |
| 800 | 209.6 | 2795 h | 11.27 h | 0.72% |

**Decision: keep `n_t = 200`. The cut order is not invoked** — no reduction to
Tier-2 interval, retained-tasks-evaluated, seeds, or streams. Even `n_t = 400`
(CV 1.0%) is affordable at 5.6 h if a measure turns out to need it.

Noise floors at the base point (12 replicates, `n_t = 200`), for `02` §6:
α 1.87%, D_eff 1.26%, R_eff 0.50%, Ψ_eff 1.39%, ρ_c 0.97%.

Scaling, for future sizing: cost is superlinear in `P` (≈ P^2.1: 70 / 288 /
1285 ms per sample at P = 8/16/32, M=150, N=300) and ≈ M^1.6, and grows with `N`.
`P = 32` would have cost ~4.5× — worth noting given that §2a came out against
needing it. `α_sim` is cheap by comparison (1.4–35 s per full bisection).

### P2a — `center_policy`: `"all"` wins, on evidence

Full B.5 protocol under both policies, at `n_t ∈ {200, 1000}`
(`results/glue_core_recovery.json`, 180 estimations, 69 s). Mean absolute relative
recovery error:

| policy | `D → D_eff` | `R → R_eff` | `ρ_C → rho_c_glue` |
|---|---|---|---|
| `"all"` | 10.3% | 7.0% | **3.7%** |
| `"active"` | **9.6%** | **6.4%** | 8.6% |

`"active"` wins by under a percentage point on `D` and `R` — comparable to seed
noise — while `"all"` wins by more than 2× on `ρ_C`, consistently at both `n_t`.
`ρ_C` is also the most direct of the three tests since the generated correlation
is exactly known, and `"active"` systematically over-estimates `ρ_c`
(0.043/0.239/0.434 vs truth 0/0.2/0.4) — the measure H1d turns on.
**Default stays `"all"`, now with evidence rather than by default.** Conclusions
about D and R are robust to the choice; conclusions about ρ_c are not. Kept in the
result key either way.

### P2b — scale compression is finite-`M`, and we found the mechanism

Three candidate causes, all tested (`results/scale_compression.json`):

1. **Not `(y,t)` Monte-Carlo.** `n_t` 200 → 1000 moves `D_eff` at `D=10` from 8.17
   to 8.19, `R_eff` at `R=2` from 1.773 to 1.785. Gap does not close.
2. **Not an artifact of B.5's `P = 2`.** Mean |rel err| on `D`: 13.1% (P=2),
   14.4% (P=8), 16.5% (P=16). Nearly flat.
3. **It is finite sampling of each manifold.** `D_eff/D` at P=2, N=1000:

| `M` | 50 | 100 | 200 | 400 | 800 |
|---|---|---|---|---|---|
| D=2 | 1.112 | 1.118 | 1.123 | 1.127 | 1.130 |
| D=6 | 0.846 | 0.898 | 0.934 | 0.955 | 0.969 |
| D=10 | 0.671 | 0.750 | 0.817 | 0.865 | 0.897 |

The deficit falls as ≈ `M^-0.45` and converges to 1. Anchors are extreme points of
a finite sample, and a finite sample under-represents a high-dimensional
manifold's extent; higher `D` needs more points for the same coverage. The ~12%
over-report at `D = 2` is a separate, `M`-independent effect.

**Consequence — a caveat, not a fix.** 5% deficit at `D = 10` would need
`M ≈ 4000`, unaffordable at Phase-1 scale. But the bias is monotone (signs and
rank orders preserved), a fixed transform at fixed `M` (which we hold constant),
and it **compresses** dynamic range — so estimated `Δ log D_eff` is biased *toward
zero* and the §8 dimension channel is **conservative**: a real contribution can be
understated, never manufactured. Stated in `00` §6.1 and in the paper.

### P2c — B.5 presentation corrected

The earlier table read as a confounded diagonal because three independent sweeps
were compressed into one row-per-measure. The script and the sheet now report each
sweep separately with the off-axis knobs stated (`D` and `R` each swept at
`ρ_C = 0`; `D = 4, R = 1` held while `ρ_C` sweeps). The underlying runs were always
independent — this was a reporting defect, not a design one.

### P3 — §2a at β = 0: **`P = 16, N = 300` confirmed, escalation not triggered**

`scripts/run_gate_2a.py` → `results/gate_2a.json`. 81 estimations, 411 s.

| P | N | α_sim | α_core | α_mf | core–sim | mf–sim |
|---|---|---|---|---|---|---|
| 8 | 300 | 0.483 ± 0.052 | 0.440 ± 0.006 | 0.388 | 8.9% | 19.8% |
| 8 | 600 | 0.449 ± 0.046 | 0.439 ± 0.013 | 0.387 | 2.2% | 13.9% |
| 8 | 1200 | 0.436 ± 0.043 | 0.430 ± 0.016 | 0.387 | 1.5% | 11.3% |
| **16** | **300** | 0.420 ± 0.017 | 0.434 ± 0.002 | 0.413 | **3.4%** | **1.6%** |
| 16 | 600 | 0.440 ± 0.009 | 0.440 ± 0.004 | 0.414 | 0.0% | 6.0% |
| 16 | 1200 | 0.460 ± 0.013 | 0.437 ± 0.006 | 0.413 | 4.9% | 10.1% |
| 32 | 300 | 0.437 ± 0.014 | 0.440 ± 0.005 | 0.423 | 0.7% | 3.1% |
| 32 | 600 | 0.451 ± 0.007 | 0.433 ± 0.001 | 0.421 | 4.0% | 6.6% |
| 32 | 1200 | 0.440 ± 0.019 | 0.439 ± 0.004 | 0.422 | 0.1% | 3.9% |

`α_core` vs `α_sim`: 2.9% mean, ≤5% everywhere except (P=8, N=300), no systematic
sign. At the design point both `α_core` (3.4%) and `α_mf` (1.6%) sit inside
`α_sim`'s own seed scatter. **The `N`→600→1200 and `P`→32 escalation ladder is not
invoked; the five-factor redesign is off the table.**

**Caveat recorded rather than buried:** the clean `O(1/N)` trend appears only at
`P = 8` for `α_mf` (19.8 → 13.9 → 11.3%). Elsewhere there is no trend, because
`α_core` is flat in `N` (< 2% variation at fixed geometry, seed SD ~0.004) while
**`α_sim` itself drifts upward with ambient `N`** (0.420 → 0.440 → 0.460 at P=16)
with 3–10× larger seed SD. The generated geometry is `N`-independent, so that
drift is a property of the simulation estimator — its bisection range and
projection statistics both scale with `N`. Part of the residual at large `N` is
therefore `α_sim`'s bias, not `α_core`'s error. Neither is asserted correct; both
are reported. This does not affect the design-point conclusion, which is at the
smallest `N`.

### Documentation

`docs/reference/glue-core-validation.md` created — the paper's methods paragraph
and the reviewer answer to "you implemented the estimator yourself". Records, with
numbers: the α=2 exact limit, B.5 recovery per axis, the α_sim/α_mf cross-check,
the center_policy decision, and the compression limitation with its mechanism.
`Verified by (human)` blank.

### Follow-ups closing the sheet (`results/estimator_followups.json`)

**1. `α_sim`'s N-drift is an artifact — confirmed, not inferred.** Re-ran with `N`
raised by **zero-padding one point cloud**: identical geometry, isometric
embedding, capacity invariant by construction.

| P=16, M=60, 5 seeds | N=300 | N=600 | N=1200 | drift |
|---|---|---|---|---|
| `α_sim` | 0.434 ± 0.025 | 0.435 ± 0.016 | 0.461 ± 0.014 | **+6.2%** |
| `α_core` | 0.439 ± 0.006 | 0.434 ± 0.008 | 0.437 ± 0.004 | **−0.3%** |

`α_core` is invariant as it must be; `α_sim` is not. `02` §2a amended: `α_sim` is
the reference standard **at the design point** (smallest N, best agreement,
tightest scatter), not unconditional ground truth — large-`N` disagreement must
not be charged to `α_core` by default.

**2. Cost of `M` measured at project scale: ≈ `M^1.68`** (242 / 1173 / 4067
ms/sample at M = 150 / 400 / 800, P=16, N=300). Full grid: 2.8 h / 13.7 h / 47.4 h.
`M = 150` justified as the main-grid point; **a high-`M` robustness arm at `M = 800`
costs 2.1 h over 4.5% of the grid** and is now planned, so the compression is
bounded empirically rather than argued. (An earlier log line estimated `M^1.6` by
extrapolating from the 50–150 range; the direct measurement at project scale
confirms it.)

**3. Compression reframed from caveat to methods finding.** The mechanism is in
the *definition* — anchors are dual-weighted extreme points of the sampled cloud —
so every GLUE-family estimator at modest `M` inherits it. Chou et al. subsample to
**M = 50**, where our curve puts `D_eff/D ≈ 0.67` at `D = 10`. The sheet now states
the finite-sample behaviour of anchor-based geometry estimation as a deficit
scaling ≈ `M^-0.45`, uniform in `D`, independent of `P` and `n_t`. Also added: a
**caption requirement** that every figure reporting absolute `D_eff` carries the
lower-bound note, not just the methods section.

**4. MDE at 5 vs 8 seeds** — uniformly **25.4% tighter for +1.7 h wall**
(α 3.78 → 2.82%, `D_eff` 2.55 → 1.90%, `R_eff` 1.01 → 0.75%, `Ψ_eff` 2.81 → 2.09%,
`ρ_c` 1.96 → 1.46%). 8 seeds adopted. **Stated as a lower bound**: these floors are
estimator Monte-Carlo variance at fixed manifolds and exclude network-init and
stream variability; recompute from the Phase 0 pilot before the Day-14 table.

**5. Grid discrepancy found and resolved.** The `05` brief's figure (10 streams ×
5 seeds = 1,200 runs, 48,000 evals, 2.8 h) disagrees with `01` Phase 1 (20 streams
× 8 seeds = 3,840 runs, 153,600 evals, **9.0 h**). Both fit; the larger stands.

**Compute is no longer the binding constraint — engineering time is.** Next:
`src/models/` → training loop → Phase 0, no further validation detours. Phase 0's
own runs are cheap, so Gate-2 alignment and the time-reparameterization test get
run properly rather than minimally even though both feed cut hypotheses.

### `src/models/` and `src/train/` built — 55 tests pass

**Framework decision: numpy with hand-derived gradients, no torch/jax.** The model
is two layers with an MSE loss, so the gradients are four lines, and this buys
float64 throughout (`AGENTS.md` §4 — torch defaults to float32 and would need
dtype management at every op), bitwise reproducibility from `(seed, config)`, and
no dependency. Phase 1 is thousands of tiny independent CPU runs, which suits many
single-threaded processes rather than one accelerator. Gradients are verified
against central finite differences to `rtol=1e-5`
(`test_gradients_match_finite_differences`) — the whole choice rests on that test.

**Base-width constant derived (`00` §4.2 UNCERTAIN, `02` §3).** Requiring
`μP(γ₀=1, N=N_base) ≡ NTP(N=N_base)` fixes it uniquely: the width factors are
*relative* to the base width, `N^{1/2} → (N/64)^{1/2}` in the output scale and
`N → N/64` in the learning rate. At `γ₀=1, N=64` both give `γ_eff = 1`, `η = η₀`.
`test_base_width_equivalence` asserts bitwise equality on the forward pass **and**
the first gradient step, plus a companion test that the equivalence is specific to
`N = N_base`. **AGENT-DERIVED, not human-verified** —
`docs/reference/parameterization-derivation.md` is human-owned and unwritten; the
test pins the behaviour but the justification should be checked against Graldi §3.

Invariants under test: I5 `f(x;θ₀) = 0` exactly for every γ (zero-init readout);
I6 hidden weights **bitwise identical across γ** (dedicated γ-independent `shape`
RNG stream) — this is what makes γ and `a` orthogonal; I8 single shared readout.
`paired_init(seed)` exposes four named streams (`shape`/`data`/`stream`/`probe`)
so conditions are paired rather than merely seeded alike.

Alignment (`00` §5) is built despite the a-axis being cut from the Phase 1
factorial, because **Gate 2 is still to be run and recorded** and needs a correct
geodesic to be interpretable. Grassmann geodesic, norm- and rank-preserving
reassembly, endpoints and monotone principal angles all tested — including an
explicit test that the geodesic differs from linear-interpolate-then-reorthonormalize,
which is the failure the AGENTS Grassmann rule exists to prevent. Endpoint checks
use the projector distance `‖P_A − P_B‖_F` rather than the largest principal
angle: `arccos` is ill-conditioned near 1, so identical subspaces read ~1.5e-8
(√eps) no matter how exact the geodesic is, whereas the projector distance reaches
1e-14 and actually tests the code.

**Design hazard found while testing — under-training masquerades as forgetting.**
A step budget tuned on task 0 leaves later tasks unlearned, because later tasks
start from a solution to a *different* dichotomy and must overwrite it. Measured
at test scale (P=8, M=20, N=64), per-task final train accuracy:

| lr₀ | steps | task 0 | 1 | 2 | 3 |
|---|---|---|---|---|---|
| 0.05 | 400 | 0.93 | 0.78 | 0.62 | 0.69 |
| 0.05 | 2000 | 0.95 | 0.88 | 0.75 | 0.76 |
| 0.2 | 2000 | 0.99 | 1.00 | 0.98 | 0.99 |
| 0.5 | 2000 | 1.00 | 1.00 | 1.00 | 1.00 |

A task the network never learned cannot be forgotten, so its "forgetting" is
under-training and would have entered H1/H2 silently. `TrainConfig` now carries
a convergence target with early stop, and `TaskRecord.converged` /
`.usable_for_forgetting` record the outcome per task. **Phase 0 must verify
`all(t.converged)` across the γ grid before Phase 1 commits compute.**

---

## 2026-08-11 — Phase 0 (`scripts/run_phase0.py` → `results/phase0.json`)

Project config: `P=16, N=300, M=150, d=150, D=4, R=1.0, n_t=200`, γ ∈ {0.03, 0.1,
0.3, 1, 3, 10}, 3 seeds.

### GATE 2 PASSES — and this reopens a closed decision

The check `05` called "the most likely single failure in the project", and for
which D4 pre-authorized dropping the `a` axis, **passed decisively**.
Capacity-at-init against alignment `a`:

| `a` | 0.00 | 0.17 | 0.33 | 0.50 | 0.67 | 0.83 | 1.00 |
|---|---|---|---|---|---|---|---|
| `α` | 0.3091 | 0.3212 | 0.3383 | 0.3557 | 0.3736 | 0.3874 | 0.3940 |

Strictly monotone, rank correlation **+1.00**, range **23.97%** against a required
2× noise floor of 3.74% — **6.4× the threshold**. Direction is the expected one:
aligning `W(0)`'s row space with the manifold-center subspace raises capacity at
init, which is what "wealth" is supposed to mean.

**Decision returning to the human.** The `a` axis was cut by pre-authorization
anticipating this failure. It did not fail, the knob works, and compute is no
longer binding (`01` Phase 1: 9.0 h wall at 20 streams × 8 seeds). Reinstating `a`
is now a *scope* question rather than a feasibility one — it is what makes `C3`
(wealthy-lazy + poor-rich) and `C4` (anti-diagonal) meaningful, and those are the
architecture conditions the modularity story rests on. **Not reinstating it
unilaterally**; flagging that the grounds for the cut are gone.

### Capacity-at-init flat in γ — exact, by construction

Not a statistical pass. `h_m = ReLU(β₀ W_m x)` contains no γ, and the readout is
zero-init, so the representation at init is **bitwise identical across γ** and
`α` is equal to the last bit (0.310595 at γ = 0.03 and γ = 10). Verified both ways.
This is I6 holding, and it is what makes γ and `a` orthogonal.

### `pairwise` vs `full_P`: capacity agrees, geometry does not

| measure | `full_P` | `pairwise` | rel. diff | floor | |
|---|---|---|---|---|---|
| `α` | 0.3078 | 0.3109 | **1.02%** | 1.87% | agree |
| `R_eff` | 0.8718 | 0.8664 | **0.62%** | 0.50% | ~agree |
| `D_eff` | 4.548 | 5.734 | **26.1%** | 1.26% | **diverge** |
| `Ψ_eff` | 0.6041 | 0.7657 | **26.8%** | 1.39% | **diverge** |
| `ρ_c_glue` | 0.5980 | 0.6272 | 4.88% | 0.97% | diverge |
| `ρ_c_signed` | 0.5219 | 0.5348 | 2.49% | 0.97% | diverge |

The structure is the interesting part. `D_eff` and `Ψ_eff` inflate by **the same
factor** under pairwise — 1.2609 and 1.2676, a ratio of **1.0053** — so they
cancel in `α = Ψ_eff(1 + R_eff⁻²)/D_eff` to within 0.5%, and `R_eff` barely moves.
The estimation mode is close to a **pure common rescaling of the `D`/`Ψ` pair**.
Mechanistically this is what a joint QP should do: at `P = 2` only two manifolds
compete for each `t`, anchors are less constrained, and the apparent dimension
rises.

Consequences:

1. **Published GLUE capacities are comparable to ours; published GLUE
   *geometries* are not.** A 26% offset in `D_eff` is a mode difference, not a
   finding. Worth stating — the literature reports pairwise geometry.
2. **The attribution is in the affected channels.** `α` is safe either way, but
   Figure 2 decomposes into `D_eff` and `Ψ_eff`. If the ~1.26 factor is *constant
   across conditions* it cancels in `Δlog D_eff` and the attribution is untouched;
   if it drifts with the representation, it does not. **This must be checked at
   two or more conditions before Phase 1 commits** — one condition cannot
   distinguish a constant offset from a varying one. Added to the gate list.
3. `full_P` remains primary, `pairwise` reported alongside, per `01` §4.

### Stopping criterion was wrong, found by its own diagnostic

The convergence check flagged non-convergence at low γ, and the detail disproved
the obvious reading. Per-task train accuracy was **identical across all six γ**
(0.991 / 0.994 / 0.982 / 0.972) with stopping at step 1 on the first task. Reason:
from `u = 0`, one gradient step gives `u ∝ Σ_b y_b h(x_b)` — the kernel/Hebbian
readout — and

    f(x) ∝ Σ_b y_b h(x_b)ᵀ h(x)

whose **sign does not depend on the learning rate or on γ**. So train accuracy
jumps to ~0.99 at step 1 in every arm, and an accuracy-based stopping rule halts
training before any feature learning — precisely the thing under study. `01` §1
specifies `stopping: "matched_loss"`; the implementation used accuracy. Fixed:
`TrainConfig` now takes `stopping ∈ {"matched_loss", "fixed_steps"}` with
`target_loss`, and `test_accuracy_saturates_at_step_one_so_stopping_must_use_loss`
pins the reason so it cannot be undone. Check 3 rerun under matched loss
(`scripts/run_phase0_richness.py`).

Note this also means **train accuracy is near-useless as a progress measure in
this model** — it saturates immediately at every γ. Loss, `‖ΔW‖/‖W‖`, and the
geometry are the informative axes.

### Richness separation PASSES at matched loss — after fixing `lr0`

The first matched-loss rerun failed at γ ≤ 1: loss plateaued near 0.34 and never
reached the 0.05 target within 5,000 steps/task. Two candidate explanations, and
the obvious one was wrong.

**Ruled out — an expressivity floor.** `2N = 600` readout parameters against
`P·M = 2400` samples looks underparameterized, which would put a hard floor under
the lazy arm. Solving the readout exactly by least squares at fixed `W` gives loss
**0.0002**: no floor. The targets are constant within each manifold, so there are
only `P = 16` distinct values to fit and 600 features fit them trivially.

**Actual cause — `lr0 = 0.2` was simply too small.** μP's `η ∝ γ₀²` makes the
*function-space* rate γ-independent (`Δf ∝ η·s²` with `s = β_L/γ_eff`; the product
is 6.66e-4 at both γ = 0.03 and γ = 10, identical as designed). What differs is
that feature learning speeds the fit at large γ, so a global rate too small to
converge leaves only the rich arms finishing. Raising `lr0` fixes every arm at
once, and the system is stable well beyond the value chosen:

| γ | `lr0`=0.2 | `lr0`=5.0 | `lr0`=50 |
|---|---|---|---|
| 0.03 | 0.426 (no conv.) | **0.050** @1196 steps | 0.050 @120 |
| 1.0 | 0.415 (no conv.) | **0.049** @264 | 0.045 @28 |
| 10 | 0.050 @1006 | **0.049** @42 | 0.032 @6 |

**`lr0 = 5.0` adopted** for Phase 1. Re-run at 3 seeds, `T = 2`, matched loss 0.05:

| γ | 0.03 | 0.1 | 0.3 | 1.0 | 3.0 | 10 |
|---|---|---|---|---|---|---|
| `‖ΔW‖/‖W‖` | 0.0015 | 0.0125 | 0.0486 | 0.1274 | 0.2525 | 0.5025 |
| steps to target | 2883 | 2330 | 1305 | 515 | 203 | 73 |

**PASS: 2.53 decades** of `ΔW` separation at matched loss, against the 1-decade
requirement, and **every arm reaches the target at every seed**. The separation is
therefore richness, not differential training progress.

**Carry forward for H3.** Reaching the *same loss* takes **39.5× more steps** at
γ = 0.03 than at γ = 10. That spread is the raw material for the
time-reparameterization test (`00` §12): if warping step count by this factor
collapses the geometry trajectories onto each other, H3 is dead. The factor is now
measured, so the test has a principled warp to try first rather than a fitted one.

---

## 2026-08-11 — mode constancy: the `pairwise` offset is NOT constant

`scripts/run_mode_constancy.py` → `results/mode_constancy.json`. Nine geometries ×
3 seeds × both modes: six synthetic points moving one axis at a time from the
Phase 1 base (`D` ∈ {2,4,8}, `R` ∈ {0.5,1,1.5}, `ρ_C` ∈ {0,0.4}) and three network
representations (init, trained lazy γ=0.03, trained rich γ=10), since Phase 1
measures representations rather than generated manifolds.

**A first pass was wrong and is worth recording.** It placed the synthetic
manifolds in the *input* dimension `d = 150`, where `P(D+1) = 144` of 150
dimensions at `D = 8`: the arrangement is nearly degenerate, the manifolds are no
longer in general position, and `D_eff` collapses for reasons unrelated to
estimation mode. That inflated the effect (`D_eff` ratio reached 1.52). Redone in
the *measurement* ambient dimension `N = 300`. **General rule: keep
`P(D+1) ≪ ambient`** — this is why B.5 uses `N = 1000` — and it is now a guard
worth having in any arrangement built for estimation.

**Verdict: not constant.** The `pairwise/full_P` ratio on `D_eff` ranges
1.038 → 1.253 (CV 7.05%) and on `Ψ_eff` 1.050 → 1.283 (CV 7.74%).

The decisive statistic is not the ratio's spread but whether `Δlog D_eff` — the
quantity the attribution actually uses — survives the mode change:

| comparison | `Δlog D_eff` full_P | pairwise | error | vs floor |
|---|---|---|---|---|
| `rep_init → rep_rich` | −0.2273 | −0.3483 | 0.1210 | **9.7×** |
| `rep_lazy → rep_rich` | −0.2247 | −0.3441 | 0.1194 | **9.5×** |
| `rep_init → rep_lazy` | −0.0026 | −0.0042 | 0.0015 | 0.1× |
| worst overall | +0.7199 | +0.9075 | 0.1876 | 15× |

The third row is the internal control: two representations that are geometrically
almost identical, where the two modes agree to a tenth of the noise floor. So the
disagreement is not noise — **it grows with the size of the geometry change**,
which is exactly the pathology that would contaminate Figure 2. The first two rows
are the comparison Phase 1 actually makes (same network, different training
states), and there the error is ~10× the floor.

**Consequence, per the ratified rule: `full_P` is mandatory for all Tier-2
measurement.** `pairwise` is computed and reported only where comparability with
published values is the point. The appendix gets the observation that published
GLUE *geometries* are not comparable across estimation modes while *capacities*
are — `α` agrees to ~1% in every geometry tested, because the `D_eff` and `Ψ_eff`
distortions are co-directional and largely cancel in `α = Ψ_eff(1+R_eff⁻²)/D_eff`.

### Bonus, and it sharpens the compression caveat

With ground truth known at these points, both modes can be scored directly. In
log space, against the generated values:

| | `d log D_eff / d log D` | `d log R_eff / d log R` |
|---|---|---|
| `full_P` (P=16) | **0.663** | **0.581** |
| `pairwise` (P=2) | 0.772 | 0.624 |

(1.0 would be faithful.) This upgrades the earlier qualitative statement — "the
bias is monotone and compresses dynamic range, so the dimension channel
understates rather than manufactures" — into a **measured attenuation factor**:
at Phase 1 settings, a true `Δlog D` registers as ≈ 0.66 of itself and a true
`Δlog R` as ≈ 0.58 of itself. Figure 2's dimension and radius channels are
attenuated by roughly a third and two fifths respectively. Two consequences:

1. The conservatism claim now has a number attached instead of a direction.
2. It is *not* being used as a correction. Dividing by 0.663 would import the
   synthetic-manifold calibration into representation measurements, which the
   `rep_*` points show are a different regime (their mode ratios sit at 1.25 while
   the synthetic points sit at 1.04–1.21). Reported as an attenuation bound, not
   applied as a factor.

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

---

## 2026-08-11 — `src/analysis/` and the time-reparameterization test: **H3 survives**

`src/analysis/{trajectories,timewarp}.py`, `scripts/run_timewarp.py` →
`results/timewarp.json`. Phase 0's fifth and last check. Geometry
`(α, D_eff, R_eff, ρ_c)` recorded at 9 logarithmically spaced checkpoints to 4000
steps, γ ∈ {0.03, 1, 3, 10}, 3 seeds, `full_P`, `n_t = 200`. The measurement RNG is
held fixed across checkpoints so successive points differ because the
representation moved, not because the estimator resampled `(y,t)`.

Three warps of increasing generosity, residuals in units of the noise floor so
1.0 is the decision boundary: the **measured** warp (the steps-to-target-loss
ratio, no free parameters), the **best rate** (one free parameter), and **DTW**
(the most generous monotone warp there is; fixed start, which the paired-init
design justifies, and free end).

### First: the lazy arm's geometry does not move at all

Total geometric excursion over the whole training window:

| γ | 0.03 | 1.0 | 3.0 | 10.0 |
|---|---|---|---|---|
| excursion (noise floors) | **0.2** | 9.8 | 27.2 | 65.2 |

At γ = 0.03 the loss falls 0.499 → 0.010 while the representation geometry moves
**two tenths of one noise floor** — `D_eff` goes 4.436 → 4.431. All of the learning
is in the readout. This is the lazy limit confirmed by direct geometric
measurement rather than by a weight-norm proxy, and it makes the γ ≪ 1 vs γ ~ 1
contrast a contrast between *no* geometric change and geometric change, which is
what H1/H2 need.

It also makes every comparison involving γ = 0.03 **vacuous** for this test: a
trajectory that does not move is trivially the slowed-down opening of any other
trajectory, so "they coincide" says nothing about shape. Those pairs are now
labelled vacuous rather than counted as coincidence.

### Verdict: H3 is testable

| pair | measured | best rate | best monotone (DTW) | verdict |
|---|---|---|---|---|
| 0.03 vs 1 | 4.10 | 0.04 | 0.09 | vacuous |
| 0.03 vs 3 | 11.81 | 0.05 | 1.24 | vacuous |
| 0.03 vs 10 | 30.67 | 0.10 | 12.86 | vacuous |
| **1 vs 3** | 10.27 | 5.54 | **1.05** | coincide (marginal, ±0.29) |
| **1 vs 10** | 35.06 | 6.92 | **11.54** | **DIFFER** |
| **3 vs 10** | 26.24 | 21.32 | **7.06** | **DIFFER** |

Two of the three informative large-γ pairs differ by 7–12 noise floors under the
most generous monotone warp available. **Trajectories at different large γ are not
one trajectory at two speeds**, the division-of-labour hypothesis survives, and
heterogeneity contrasts are unrestricted — `00` §12's fallback does not fire.

Caveat worth carrying: `1 vs 3` is marginal at 1.05 ± 0.29, and those are the two
arms that move least (9.8 and 27.2 floors). So the honest rule is **use
well-separated γ with γ = 10 as one endpoint**; a 1-vs-3 contrast is not
demonstrably a shape difference.

### Two substantive by-products

**Matching loss does not match geometry.** The parameter-free measured warp fails
everywhere (4–35 floors). And the rate that best aligns *geometry* is 10–20× the
rate that aligns *loss*: for 1 vs 10, best-rate scale 161.6 against a loss-based
8.0. Geometry and loss evolve on different clocks, and even the best geometric rate
leaves 6.92 floors. This is worth a sentence in the paper: matched-loss stopping
equalizes training progress, not representational change.

**ρ_c is by far the most dynamic channel.** Movement per channel at γ = 10, in
floors: `ρ_c` 122, `R_eff` 22, `α` 19, `D_eff` 18. Good news for H1d, which turns
on ρ_c: the measure with the tightest noise floor is also the one that moves most.

### Two bugs, one of which nearly produced the wrong headline

1. `measured_warp_scale` returned the inverse of the convention `residual` uses.
   Caught by a unit test asserting a known rate is recovered.
2. DTW required coverage in *either* series rather than **both**. That let a nearly
   static trajectory be matched against the other's flat opening segment and score
   a near-zero residual — a degenerate warp reported as coincidence. Under the
   buggy version `1 vs 10` scored 1.18 and would have been called marginal
   coincidence; corrected it is 11.54. Both directions are now pinned by controls:
   a synthetic pure-rate-rescaling must be recovered and called coincident, and a
   synthetic shape inversion must survive every warp.

---

## 2026-08-11 — Phase 1 pipeline, and the artifact check becomes a rule

Phase 0 is closed. `AGENTS.md` §8.2 now makes the pattern that closed it mandatory:
**before interpreting any result, state what would have to be true for it to be an
artifact, and check that.** Both of the last two bugs were caught by diagnostics
rather than by tests, and the DTW one was a false negative that would have killed H3
and looked like a finding. The section carries the four cases so far as a table, plus
the four checks that have earned default status: compare against an exactly solvable
limit; vary what should not matter; confirm the method *fails* where it should;
and ask whether the quantity moved at all before interpreting agreement about how it
moved.

It earned its keep within the hour — see the probe finding below.

### Built

- `src/analysis/attribution.py` — the exact three-factor log-space decomposition, plus
  the `ρ_c → R_eff` conversion.
- `src/pipeline.py` — one arm end to end: stream → sequential training → Tier-2
  geometry on a schedule → attribution, accuracy matrix, `CF`/`CFr`, and the `00` §11
  manipulation checks.
- `scripts/run_phase1.py` — the grid, resumable, one JSON per arm.
- `scripts/run_probe_check.py` → `results/probe_check.json`.
- `scripts/fig_gamma_excursion.py` → `results/fig_gamma_excursion.csv` and
  `results/figures/gamma_excursion.{pdf,png}`.
- 26 new tests; 90 pass.

**Schedule and budget.** Tracked tasks every 4th, measured at their own boundary
(the attribution baseline) and at every later measurement boundary, so retention is
followed over a widening lag rather than at one fixed lag: `{0:[0], 4:[0,4],
8:[0,4,8], 12:[0,4,8,12], 15:[0,4,8,12]}`. That is **38 evaluations per arm against
the cost model's 40**, asserted by a test. The measurement RNG is fixed across
boundaries and shared between the generic and retained ensembles, so a change between
two boundaries is the representation moving, and the generic/retained crossing is
paired at every boundary.

### The probe measure was dead, and its first replacement was noise

Figure 3 overlays a probe metric on generic capacity. The obvious choice — accuracy
of a refit linear readout on the held-out dichotomy `y*` — is **exactly 1.0**, at
initialization and after training, in both modules, at every γ. The load is
`P/N = 16/300 = 0.053` against a critical capacity near 0.3, so every balanced
dichotomy is separable with room to spare, and separability at fixed sub-critical
load cannot track capacity. It would have plotted as a flat line at 1.0 and read as
"generic capacity is preserved".

Three candidates, scored on whether the γ signal beats the measure's own noise:

| measure | γ=10 | γ=1 | γ=0.03 | signal | noise | SNR | verdict |
|---|---|---|---|---|---|---|---|
| `accuracy` | 1.0000 | 1.0000 | 1.0000 | 0 | 0 | — | **saturated** |
| `margin` | 0.0500 | 0.0632 | 0.0598 | 0.0132 | 0.0033 | **4.0** | use |
| `margin_p05` | 0.0462 | 0.0565 | 0.0521 | 0.0102 | 0.0032 | 3.2 | use |
| `heldout_manifold_accuracy` | 0.4608 | 0.4635 | 0.4142 | 0.0494 | 0.1206 | 0.4 | **noise** |

**`margin` is the probe measure**, and it is the principled one rather than a
fallback: capacity *is* the load at which the margin reaches zero, so the margin is
the graded quantity underneath the thresholded one. Its change over training is
sign-consistent across seeds in every arm — rich training reduces the probe margin,
γ=0.03 leaves it unchanged to four decimals, which is the geometrically static lazy
arm showing up again on an independent measure.

**Held-out-manifold accuracy is a control, not an overlay.** It sits at chance with a
within-run split sd of 0.06–0.16, which is *correct* rather than broken: `y*` is a
random balanced dichotomy, so there is no shared structure for a readout fitted on 12
manifolds to extend to 4 unseen ones. Its apparent γ=0.03 dip is one seed's
initialization (0.329 vs 0.495 at init, and γ=0.03 barely moves), not an effect of γ.
Kept because if it ever rises above chance, the probe is leaking factor structure.

The artifact rule is what produced this: the first question asked of a 1.0 was "did
this quantity move at all", and the second, asked of the 0.414, was "is this
difference bigger than the seed difference". It was not.

### `ρ_c → R_eff`: the conversion is measured, and it does not transfer unchanged

Fitted on the B.5 center-correlation sweep, where ground-truth `ρ_C` 0 → 0.8 drives
`R_eff` 1.02 → 1.82 while `D_eff` stays flat to **0.38%** — the radius absorbs center
correlation essentially alone, as Wakhloo's duality says:

    log R_eff = 0.0126 + 0.3548 · (−log(1 − ρ_c))      R² = 0.99986

`R_eff ∝ (1 − ρ_c)^(−0.355)`, five points, near-exact. Attribution now reports what
fraction of each observed `Δ log R_eff` the observed `Δ ρ_c` accounts for, which makes
`00` §8's joint-reporting requirement quantitative.

Two guards, both from the first smoke grid. The denominator is **floored at the
`R_eff` noise floor** (CV 0.50%): fractions of −134 were coming from radius changes of
order 1e-3, and a ratio whose denominator is unresolvable is not a measurement.
And fractions **≫ 1 survive legitimately** — one group reports 42 — which is
informative rather than broken: it means `ρ_c` collapsed far more than the radius
followed, so the synthetic calibration (fitted where `ρ_C` is the *only* thing
varying) over-predicts on representations, where radius and centers move together.
**Open, to settle at the design point:** whether that over-prediction is systematic
enough to quote a transfer factor, or whether the conversion should be reported only
as a bound. Not answerable at smoke settings (`n_t = 25`).

### Smoke grid runs; the shape is right and is not yet evidence

Reduced settings (`T=4, P=8, M=40, N=80, n_t=25`) — a wiring check, not a
measurement. Retained capacity falls at every lag, far more at γ=10 (Δlog α ≈ −0.95
to −1.28 at lag 1) than at γ=0.3 (−0.11), and the dominant factor differs between
them: utility at γ=10, radius at γ=0.3. That is the H1 shape. It is also 25 `(y,t)`
samples and one seed, so it goes in this log and nowhere near a figure.

Identity residual across every evaluation in the suite: **≤ 3.5e-16**.

---

## 2026-08-11 — **the Phase 1 cost model was wrong by 9×**, and why

Launched the full grid (1280 arms, 48,640 evals) on 254 workers. After 79 minutes,
**zero arms had completed.** Load was a steady 267 and every worker had accumulated
79 minutes of CPU, so nothing was blocked — it was simply far slower than projected.
Timed one evaluation against the live grid: **775.8 s, against the 52.4 s the cost
model assumed.** Killed it (no arm was near completion, so nothing was lost) and
measured the throughput curve properly — `results/scaling.json`,
`scripts/run_scaling.py`, `n_t = 20` to keep it short, which is legitimate because
cost is linear in `n_t` while the memory footprint driving contention is set by `P·M`.

| workers | eval latency | throughput | effective cores | efficiency |
|---|---|---|---|---|
| 1 | 5.71 s | 0.166 eval/s | 0.9 | 0.94 |
| 32 | 6.77 s | 4.078 eval/s | 27.6 | 0.86 |
| 64 | 11.27 s | 4.633 eval/s | 52.2 | 0.82 |
| **128** | 20.44 s | **5.169 eval/s** | 105.7 | 0.83 |
| 192 | 40.26 s | 4.150 eval/s | 167.1 | 0.87 |
| 254 | 60.10 s | 3.749 eval/s | 225.3 | 0.89 |

**Throughput peaks at 128 workers and then falls. Running 254 was worse than running
32.** Two structural causes:

1. **128 physical cores, not 256.** `nproc` reports 256 on this 2×64-core EPYC 7763
   because of SMT, so 254 workers was already 2× oversubscribed before anything else.
2. **The anchor QP is not small.** It has `P·M = 2400` variables, so its Gram is
   2400² × 8 B ≈ **46 MB against a 32 MB L3**. Every worker streams that from DRAM on
   every one of `n_t` samples, so the workload is memory-bandwidth-bound and cores past
   the bandwidth limit contribute nothing.

**The error in `results/cost_model.json` was methodological, not arithmetic.** It
measured 52.4 s/eval in isolation and projected wall time by dividing total
core-seconds by 248, i.e. it assumed perfect scaling and never measured it. The
`01` §4 line "throughput is set by total core-seconds, so one thread per worker is the
correct configuration for an embarrassingly parallel grid" was the load-bearing wrong
sentence: the grid is embarrassingly parallel in its *control flow* and
bandwidth-coupled in its *memory access*, and only the first was checked.

`_par.n_workers` now caps at 128 (was `nproc − 2` = 254), which alone is a 1.4×
throughput gain over what was running.

**Revised cost: the full grid is 26.1 h at 128 workers, against the 3.0 h projected.**
Training adds ~2 h. This invokes the `01` §4 cut order (`n_t` → Tier-2 interval →
retained-tasks-evaluated) for the first time — pre-authorized there, and the decision
of how far to cut is pending.

Worth noting for the paper's reproducibility section, since it generalizes: **the cost
of anchor-based GLUE estimation is bandwidth-bound in `P·M`, not compute-bound.** At
`M = 150` and `P = 16` the QP working set exceeds a 32 MB L3, so the estimator does not
scale with core count on a shared-memory machine past the bandwidth limit. Anyone
sizing a GLUE run from a single-process timing will over-project by roughly an order of
magnitude. Chou et al.'s `M = 50` puts the working set at ~5 MB, inside L3 — their
subsampling has a performance rationale as well as a statistical one.

---

## Where the 26 h actually came from, and a calibration bug the pilot caught

### The 9.3× decomposed

The projection was 2.82 h: 48,000 evals × 52.4 s/eval ÷ **248 workers**. The measured
figure was 26.1 h. Multiplying out:

| term | factor | note |
|---|---|---|
| grid size 48,000 → 48,640 | **1.01×** | essentially nothing |
| workers 248 → 128 | **1.94×** | the cap imposed after the scaling measurement |
| per-eval cost 52.4 s → ~247 s | **4.72×** | contention at 128 workers |

1.01 × 1.94 × 4.72 = 9.24, and 2.82 h × 9.24 = 26.1 h. ✓

**The grid did not grow — it is 1.3% larger than planned.** The entire overrun is
memory-bandwidth contention, and the worker cap is not a separate cause but a symptom
of the same one: throughput peaked at 128 because beyond that the memory system
saturated, so the cap was the best available response to contention, not an independent
choice. The original model's error was to measure per-eval cost single-threaded in
isolation and assume linear scaling to 248 workers. For a memory-bandwidth-bound
estimator that assumption is worth a factor of ~9.

The practical consequence inverts the obvious reading: **cutting the grid is a weak
lever and the solver is a strong one.** Halving the grid halves the cost, once. But
per-eval cost is inflated 4.7× by bandwidth, so shrinking the estimator's working set
attacks the dominant term — and should pay more in parallel than it does alone.

### Solver change: speed only, exact to machine precision

`_nnls_colgen` replaces the dense active-set solve with column generation over the
anchor matrix. On real Phase 1 data, single-threaded, clean process, current code:

| solver | s/eval | α | D_eff | R_eff |
|---|---|---|---|---|
| `nnls` (previous) | 54.4 | 0.316013 | 5.055807 | 0.960511 |
| `colgen` (current) | **21.3** | 0.316013 | 5.055807 | 0.960511 |

Identical to every digit reported, and to 3.7e-15 under test. **2.55× faster.** The
54.4 s also confirms the 52.4 s baseline held — per-eval cost never regressed. So the
pilot's geometry stands (it ran the reference solver); only its *timings* describe
superseded code. Contended throughput at 128 workers is the number that resizes the
grid and is still to be measured.

### The stale fork, made structural

The pilot's workers ran the pre-change solver: the parent imported the module, the edit
landed, and `fork` handed every worker the parent's already-imported copy. Two changes,
neither relying on vigilance:

1. **`spawn`, not `fork`** (`scripts/_par.py`). Workers re-import from disk. Costs about
   a second of startup against arms that run tens of minutes.
2. **Version stamps** (`src/provenance.py`). Every result record carries the git SHA,
   the dirty flag, and a per-module source hash; `assert_current()` at worker start
   refuses to run stale.

The part that is easy to get wrong: **the hash must be taken at import time.** Hashing
the file at worker start cannot detect this failure, because the forked child re-reads
the *new* bytes from disk while executing the *old* bytes in memory, and would report
agreement. Only a value frozen at import travels with the fork.
`tests/test_provenance.py` asserts this by actually forking.

### The pilot's real find: the ρ_c calibration was fitted on the wrong convention

The first draft figure reported center-collapse shares of **15–38**, where 1.0 means the
radius change is entirely center collapse. That is not a finding, it is an artifact, and
the artifact check (`AGENTS.md` §8.2) found it:

- `radius_from_rho` uses R_eff ∝ (1 − ρ_c)^−0.355, which needs ρ ∈ [0, 1).
- It was fitted on **`rho_c_glue`**, which is **unnormalized** (`00` §6.1 C3) and so is
  not a correlation. On Phase 1 representations it runs to **1.487**, median 1.235, with
  **71% of measurements above the fitted maximum of 0.809**.
- A `np.clip(rho, 0, 0.995)` then turned every out-of-domain input into a large finite
  number. The domain violation produced no error, just plausible-looking results.

Why it survived the B.5 validation: on synthetic manifolds the two conventions almost
coincide (0.038/0.043, 0.220/0.236, 0.409/0.426, 0.609/0.614, 0.809/0.803). They
separate only on representations. **A calibration validated on synthetic data was
carrying an assumption the synthetic data could not test.**

Fixed by refitting on `rho_c_signed`, the normalized convention, which on the same
representations sits at **0.331–0.481 — inside the fitted range**:

    log R_eff = 0.002911 + 0.36483 · (−log(1 − ρ_c_signed)),  R² = 0.9997

and by replacing the clip with a hard refusal outside `RHO_FIT_RANGE`. Corrected
center-collapse shares are **0.28–0.76**: between a third and three quarters of the
radius change is accounted for by center collapse, the rest being anisotropy. That is
an interpretable result where 15–38 was not.

Also structural: `summarize` now **re-derives attribution from stored geometry** rather
than reusing what the worker computed. Geometry is the measurement, attribution is
analysis over it, so a calibration correction is now a re-summarize rather than a 26 h
re-run — which is exactly what this fix needed.

### Draft Figure 2 (γ=10, S-HL/S-LL, 4 of 8 pilot arms)

`results/figures/attribution_pilot.png`. Plumbing confirmed end to end: Tier-2 output
reaches the attribution code in the right shape, the identity closes (max residual
0.0), and the figure script runs.

At γ₀=10, retained capacity falls by Δlog α ≈ −1.10 to −1.52 (a 3.0–4.6× drop), and the
split is stable across lag: **utility ≈ 48%, dimension ≈ 44%, radius ≈ 8%**. Most of the
loss is present by lag 3 and deepens slowly. The lazy arm is still running.

On the encoding problem: signed contributions cannot use a naive stacked area, since
`−Δ log D_eff` is negative by construction. The figure stacks positives up from zero and
negatives down from zero and overlays `Δ log α` as a line at the signed total, so the
line reads correctly whether or not the terms share a sign. In this arm all three happen
to be negative; mixed signs are handled and will appear in the lazy arm.

### Pilot sizing

8 arms and 304 evals was not the single-cell pilot intended — one γ, one condition, one
seed, ~38 evals. The de-risking value was reached at the first completed arm. Size
pilots to the smallest unit that exercises the whole path.

### Grid resized on measured throughput: 26.1 h → 4.0 h, no cuts needed

Measured on the **real** workload at `n_t = 200`, spawned pool, clean processes
(`results/scaling_colgen.json`):

| solver | workers | latency/eval | throughput | full grid (48,640) |
|---|---|---|---|---|
| `nnls` | 128 | 218.3 s | 0.547 eval/s | **24.7 h** |
| `colgen` | 128 | 33.4 s | 2.829 eval/s | 4.8 h |
| `colgen` | 192 | 45.3 s | 3.390 eval/s | **4.0 h** |
| `colgen` | 254 | 66.9 s | 3.470 eval/s | 3.9 h |

The `nnls@128` row predicts 24.7 h against the 26.1 h actually observed — 5% — which is
the check that this measurement describes the same thing the grid did.

**colgen is 5.17× faster in parallel against 2.55× single-threaded.** The gap between
those two numbers is the point: the win is not arithmetic, it is the working set. A
solver that fits in cache stops competing for memory bandwidth, so the benefit compounds
with the worker count rather than being independent of it. This is why the earlier
reading — that grid size drove the overrun and the solver was not the lever — had it
backwards: the grid was 1.3% over plan, and the solver was worth 6.5×.

Two consequences. **No cuts to the grid are required**; the scope conversation is moot at
4 h. And the worker cap moves 128 → 192, because the peak is a property of the workload
and the workload changed: colgen is no longer bandwidth-bound, so SMT past 128 physical
cores now helps where it previously hurt. 254 buys 2% over 192 for 48% more latency.

A related defect in the old curve: `run_scaling.py` measured at `n_t = 20` while the grid
runs at `n_t = 200`, so `scaling.json` characterised an evaluation ten times cheaper than
the real one and located the peak on the wrong workload. Now pinned to the grid's value.

### Pilot complete: 8/8 arms, max identity residual 3.4e-16

The γ contrast is the headline and it is large. At **γ₀ = 10**, Δlog α = −1.10 to −1.52
(a 3.0–4.6× loss of retained capacity). At **γ₀ = 0.03**, Δlog α = −0.006 to −0.010.
**Roughly 150× less forgetting in the lazy arm** — H1's predicted direction, at a
magnitude far beyond noise.

The more interesting result is that the *mechanism* differs, not only the amount:

| | utility | dimension | radius |
|---|---|---|---|
| γ₀ = 10 | **48%** | 44% | 8% |
| γ₀ = 0.03 | ~5% | 47% | 48% |

In the rich arm forgetting is dominated by `Ψ_eff`, the alignment between the readout
and the manifold axes. In the lazy arm utility barely moves and what little loss occurs
splits evenly between radius and dimension. That is a qualitative difference in kind and
is what the shared-axis figure hides, hence the native-scale inset.

Center-collapse share is `n/a` throughout the lazy arm — Δlog R_eff ≈ 0.004 sits below
the 0.005 noise floor, so the guard declines to report a ratio. Working as intended.

`results/figures/attribution_pilot.png`.

---

## Pre-launch hardening, and a destructive default that cost the pilot

### The `np.clip` audit

Four hits in the measurement path. Two are not guards: `network.py:79` is the ReLU, and
`core.py:366` is a divide-by-zero guard already made dead by an enclosing `np.where`. The
other two were real instances of the ρ_c pattern — legitimate absorbers of float error
that never checked the violation *was* float error:

- `alignment.py:74`, `arccos(clip(s, −1, 1))` on cosines of principal angles.
- `core.py:157`, `maximum(res.x, 0)` on duals from a bound-constrained solve.

Both are correct in intent: `arccos(1 + 2e-16)` is a NaN and propagating it is worse than
clipping. What was missing is that neither could distinguish 2e-16 from 0.5. Both now go
through `numerics.clip_to_noise`, which clips **and asserts nothing moved by more than a
stated tolerance**. A clip can no longer change meaning from "absorb rounding" to
"manufacture a value" without failing. `tests/test_numerics.py` includes the pilot's own
`rho_c` values as a case: fed to the old bounds they now raise instead of returning 15–38.

### ρ_c coverage is now logged per condition

`summarize` → `rho_coverage` reports, per (γ, `a`, condition): the range of both
conventions, how many `rho_c_glue` values exceed 1, and the fraction of measurements
inside `RHO_FIT_RANGE`. Out-of-range points report `n/a` and are never extrapolated. On
the pilot, coverage is **100%** in all four cells, with `rho_c_signed` at 0.29–0.52 —
mid-range, so the grid has room either side. The unnormalized convention meanwhile
reaches **1.928, above 1 in 39 of 56 measurements** in the rich arm, which is the clearest
statement yet of why it cannot feed a bounded-domain formula.

### The default was the destructive branch, and it deleted the pilot

Verifying checkpoint-resume before the 4 h launch was the right instinct and found a real
defect the hard way. `--resume` was **opt-in**, and the default branch ran
`_path(s).unlink()` over every arm in the grid before starting. So invoking `--pilot` *to
test resume* deleted all eight completed arms and began recomputing them. With
`results/phase1/` in `.gitignore` there was no way back. The pooled summary and figure
survived, and the full grid subsumes those cells, so nothing was lost that the grid will
not regenerate — but on the 4 h grid the same keystroke would have destroyed the run.

The fault is not that resume was broken; the skip logic was correct, including the
tuple/list round-trip subtlety. It is that **the default was the destructive branch.**
Fixed by inverting it: resuming is now the default, `--resume` is accepted and ignored,
and `--fresh` discards by **moving arms to `results/phase1/.trash-<timestamp>/`** rather
than deleting. `tests/test_phase1_resume.py` drives `main()` through the branch that did
the damage, which the first version of the test did not — it used `--summarize-only` and
so would have passed against the bug.

Verified end to end on the smoke grid: the second invocation reports `2 already on disk,
resuming`, recomputes nothing, finishes in 0.83 s, identical residual.

This is a fourth member of the silent-failure family, with a twist worth naming. The
first three produced plausible output measuring the wrong thing. This one produced *no*
output and destroyed input, and it was triggered **by the act of verification** —
checking whether resume worked was what broke it. Verification steps need the same
scrutiny as measurement steps.

---

## Scope conformance audit (§1), run against 742 arms mid-grid

`scripts/audit_scope.py` → `results/audit_scope.json`. Re-runs unchanged at completion.
**29 pass, 2 warn, 0 fail.** Scope statement now written down as `docs/06-scope.md`.

The checks that matter, with their evidence:

| check | evidence |
|---|---|
| C2 arms homogeneous in γ | per-module `ScalingConfig` resolved from stored specs; γ identical, 0 violations |
| module set is the (A,B) control only | one distinct `module_list` on disk: `('A','B')` |
| no a1b2 machinery reachable | 23 imported modules, **1,995 identifiers** scanned; no routing / task-conditioned heads / task-ID input / comms / gating |
| single shared readout | readout is `u[module]`, indexed by module only; no task index at any of 5 sites |
| ratified config | `full_P`, `n_t=200`, `T=16`, `center_policy=all`, P=16, M=150, N=300, d=150, D=4 — single value each |
| γ grid / conditions / seeds | 0.03–10 (6 values), Hiratani 2×2 only, 8 seeds, 5 shared streams |
| both ρ_c conventions recorded | every geometry row carries `rho_c_glue` and `rho_c_signed` |
| version stamps | 742/742 stamped, **0 stale, 0 unregistered**, 0 module-hash mismatches |
| identity residual | **max 4.44e-16** over 28,158 evaluations (median 1.15e-16) |
| ρ_c coverage | 24 cells, **100% in range in every cell**, 0 cells left [0.04, 0.80] |
| richness separation | S-HH ×246, S-HL ×157, S-LH ×117, S-LL ×112 — all ≥ 1 order of magnitude |
| no dead module | min output-variance share **0.247** over 23,712 module-boundary records |

Two warnings, both understood: completeness (742/1280, grid running) and the `a > 0`
module-duplication caveat recorded in `06-scope.md` §4.

**The audit's own first version failed, and the failure is instructive.** It grepped whole
files for forbidden vocabulary and reported a FAIL on three docstrings — `core.py`'s
"estimator-routing table", `alignment.py`'s "Gate 2", `_par.py`'s "memory-bandwidth-bound".
All prose, no machinery. The deeper problem was symmetric: a text search that trips on a
cross-reference would equally miss a real `route()` sitting in a stripped comment line, so
it was wrong in both directions. Rewritten to walk the AST and match **identifiers only** —
names bound or referenced, docstrings excluded — which is what "no machinery reachable"
actually means. Fifth instance of a check measuring the wrong thing; caught this time
because a FAIL demanded explanation rather than a pass being accepted.

### The completion tail is the *lazy* arms, not the rich ones

Wall time and steps-to-target per γ, over the 741 `a = 0` arms complete so far:

| γ₀ | n | median wall | median steps/task |
|---|---|---|---|
| 0.03 | 160 | **115.9 min** | 1,469 |
| 0.1 | 73 | 65.7 min | 744 |
| 0.3 | 102 | 44.4 min | 285 |
| 1 | 119 | 32.7 min | 92 |
| 3 | 127 | 26.3 min | 31 |
| 10 | 160 | **26.8 min** | 11 |

Monotone, and in the opposite direction to the natural guess: **the lazy arms are 4.3×
slower, driven by 134× more SGD steps to reach the target loss.** The reason is
matched-loss stopping — at small γ feature learning is suppressed, so the function moves
slowly and many more steps are needed to drive loss to 0.05, whereas γ=10 arrives in ~11.
Rich arms move further *in weight space* (which is the richness manipulation, ×157) but
reach the loss target far sooner, and it is the loss target that ends a task. Nothing
pathological; the tail is priced in.

Grid ordering interacts with this as designed. Tier 0 (γ ∈ {0.03, 10}) is **complete at
320/320**, so Figure 2's core lazy-vs-rich contrast is already fully in hand and does not
depend on the remaining runtime. Tier 1 (mid γ) stands at 421/640 and contains the
remaining γ=0.1 arms, which is what stretches the end. Tier 2 (the `a`-sweep, all at
γ=1) has not started and will be fast.

---

## §2a draft — Figure 2 across the γ sweep (776 arms; re-run at completion)

`scripts/fig2_gamma_sweep.py` → `figures/fig2_gamma_sweep__pooled__lag12__*.{pdf,png,json}`,
filename carrying the arm count and a SHA over the contributing keys and module hashes.
Reads through `src/analysis/grid.py`, which re-derives attribution from stored geometry
and excludes `a > 0`.

### Artifact check, before the result

**Could the composition trend be a lag artifact?** Pooling every lag mixes lag-1
comparisons with lag-15, so a γ difference in composition could be a difference in how far
through the stream the average comparison sits. Ruled out by matching lag: the figure is
at lag 12 throughout, and the trend is present at fixed lag.

**Could the shares be attributing a change that did not happen?** Yes, and at low γ they
were. `shares` are `|term| / Σ|term|`, defined whether or not anything moved. At γ = 0.03
and 0.1 the total change is **0.1 and 0.5 noise floors** — below resolution — yet the naive
plot showed confident shares (utility 0.52 and 0.60). Shares are now drawn only where
|Δ log α| ≥ 2 floors, and low-γ cells read "not resolvable".

**Could the utility share of 0.02 at γ = 0.3 mean utility is inactive?** No, and this was
the subtler trap. Cancellation is ~0 at every γ (median 0.000, so factors are not
fighting), so the near-zero pooled utility term is not opposing contributions within an
arm — it is the term **crossing zero across arms**. Reported as a sign structure in its own
panel rather than hidden inside an absolute-value ratio.

**Could the low-γ positive utility term be a real positive effect?** No — the sign flips on
53% and 55% of arms at γ = 0.03 and 0.1, i.e. it has no sign. An earlier draft of the panel
called this "Ψ_eff changes sign", which overclaims the unresolved side. Points whose
positive fraction lies in (0.35, 0.65) are now drawn hollow, and the claim is the weaker,
defensible one: **utility is unresolved below γ ≈ 0.3 and resolvably negative from γ = 1.**

### The result

At matched lag 12, pooled over the 2×2, module A:

| γ₀ | Δlog α (floors) | Δlog α | Ψ_eff | −D_eff | 1+R⁻² | frac. Ψ>0 | ρ_c share | n |
|---|---|---|---|---|---|---|---|---|
| 0.03 | 0.1 | +0.0012 | +0.0005 | +0.0001 | +0.0006 | 0.53 | n/a | 160 |
| 0.1 | 0.5 | +0.0083 | +0.0054 | −0.0005 | +0.0035 | 0.55 | 0.04 | 86 |
| 0.3 | 3.7 | −0.0686 | −0.0012 | −0.0348 | −0.0326 | 0.39 | 0.13 | 106 |
| 1 | 16.0 | −0.2970 | −0.0840 | −0.1227 | −0.0903 | 0.26 | 0.23 | 130 |
| 3 | 29.5 | −0.5466 | −0.2219 | −0.2340 | −0.0907 | 0.28 | 0.42 | 134 |
| 10 | **50.7** | −0.9402 | −0.4213 | −0.4053 | −0.1136 | 0.25 | 0.43 | 160 |

**Magnitude is smooth, not a transition.** Forgetting rises monotonically from 0.1 floors
at γ = 0.03 to **50.7 floors at γ = 10**, accelerating in log γ with no discontinuity. The
lazy arms do not forget at all in the resolvable sense — 0.1 floors is nothing happening,
consistent with the Phase 0 finding that the lazy arm is geometrically static.

**Composition shifts smoothly too, with one qualitative event.** Over γ = 1 → 10 the
utility share climbs 0.28 → 0.45 while radius falls 0.31 → 0.12 and dimension holds ~0.42.
The one non-smooth feature is that **Ψ_eff becomes resolvable and negative between γ = 0.3
and γ = 1**, going from no detectable sign to co-dominant with dimension. So the answer to
"smooth or transition near γ₀* ≈ 0.1" is: the magnitude is smooth; the composition has a
threshold, and it sits at γ ∈ (0.3, 1), **not at 0.1**. Measured γ* (argmin of final
average error) is **1.0 on the grid, 0.615 interpolated in log γ**, so the composition
threshold brackets γ* rather than the pre-registered 0.1. Reporting the location as an
interval, since six half-decade points cannot place it tighter.

**The radius channel is increasingly center collapse.** The ρ_c-attributable share of
Δ log R_eff rises 0.13 → 0.23 → 0.42 → 0.43 with γ, at 90–100% panel coverage for
γ ≥ 0.3. So in rich arms nearly half the radius change is centers moving rather than
manifolds expanding — which is why `00` §8 requires ρ_c reported alongside R, and it is
the panel that stops the radius channel being misread. At γ = 0.03 coverage is 19% and the
share is suppressed: the radius does not measurably move, so there is nothing to attribute.

Two things to flag against the pre-registration, deferred to §2d: **the dimension channel
is much larger than the radius-heavy forecast for H1a** (−0.41 vs −0.11 at γ = 10, i.e.
dimension is 3.6× radius), and the utility result — the more novel half — is confirmed and
large. Overall coverage of the center-collapse panel across all comparisons is 73%.

---

## Finding 5: the `Ψ_eff ∈ [0,1]` bound was never ours to assume for retained capacity

Raised by a question about whether Ψ_eff rises in absolute terms during rich training. It
does not — but checking it surfaced retained `Ψ_eff > 1`, flagged-and-stopped because the
utility channel *is* the H1 result, and then **resolved by measurement: the bound is a
property of the label average, not of the estimator.** Ruling and check design: Kati.

This is the inverse of the `rho_c` clip bug. There, a guard manufactured in-range values
and hid a domain violation. Here, refusing to guard surfaced a real property of the
estimator that the source papers do not discuss. Same protocol, opposite polarity — which
is the argument for the protocol rather than for either outcome.

### What was actually measured

**Generic (label-agnostic) Ψ_eff falls monotonically with boundary, it does not rise:**

| γ₀ | boundary 0 | 4 | 8 | 12 | 15 | init → end |
|---|---|---|---|---|---|---|
| 0.03 | 0.6065 | 0.6073 | 0.6054 | 0.6100 | 0.6018 | 0.992× |
| 1 | 0.6142 | 0.6111 | 0.6073 | 0.6087 | 0.6003 | 0.977× |
| 10 | 0.5981 | 0.5763 | 0.5615 | 0.5558 | 0.5459 | **0.913×** |

**But there is a real overshoot, on the retained ensemble**, which is a different quantity
and the reason the two readings can both be true. Retained Ψ_eff for a task, measured at
that task's own boundary versus later ones:

| γ₀ | at own boundary | at later boundaries | ratio |
|---|---|---|---|
| 1 | 0.8911 | 0.8720 | 0.979 |
| 10 | **1.2941** | 1.0429 | **0.806** |

So learning a task lifts *that task's* retained utility from the ≈0.60 generic baseline to
≈1.29, and later tasks relax it back toward ≈1.04. The "same currency, different reference
point" reading is supported — what forgetting removes is the task-specific utility gain
that learning produced — but the mechanism is a **retained-vs-generic** gap, not a rise in
absolute Ψ_eff over training. Generic utility drifts *down* by 8.7% at γ=10.

### The bound question

**Retained Ψ_eff exceeds 1 on 21.6% of measurements, to a maximum of 1.782.** Generic
Ψ_eff never does (max 0.6550 over 8,680 measurements).

| ensemble | n | min | median | max | fraction > 1 |
|---|---|---|---|---|---|
| generic | 8,680 | 0.4486 | 0.6051 | 0.6550 | 0.000 |
| retained | 24,304 | 0.6764 | 0.7963 | **1.7823** | **0.216** |

What is ruled out: **it is not an arithmetic fault.** The three-factor identity closes to
`max|residual| = 4.36e-16` across all 5,250 evaluations with Ψ_eff > 1, so α, D_eff, R_eff
and Ψ_eff are mutually consistent wherever this occurs.

The `[0,1]` range came from the GLUE papers, where capacity is an expectation over
**random dichotomies**, and was exported into `00` §6.2 and the glossary without checking
whether its derivation survives fixing `y`. Kati's hypothesis, from the structure of the
definitions: `a` uses `(S_y S_yᵀ)†` while `c` uses `(S_{y,0}S_{y,0}ᵀ + S_{y,1}S_{y,1}ᵀ)†`,
so the two differ by center–axis **cross-terms**. Under `E_y` those plausibly vanish on
average, giving `E[c] ≤ E[a]` and hence the bound; **at fixed `y` they survive.** Note that
an earlier session had treated `Ψ_eff ∈ [0,1]` as the valid range and attributed an
out-of-range value to a faulty test generator — the question was explained away, not settled.

Three checks, run in that order, with a hard stop if any generic measurement exceeded 1.

**Check 1 — where do the >1 cases live?** Retained-only, absent below γ = 1, concentrated at
small lag, decaying with lag. Fraction above 1, rows γ and columns lag = boundary − task:

| γ₀ | lag 0 | 3 | 4 | 7 | 8 | 11 | 12 | 15 |
|---|---|---|---|---|---|---|---|---|
| 0.03 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 0.1 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 0.3 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 1 | 0.02 | 0.00 | 0.18 | 0.21 | 0.25 | 0.25 | 0.25 | 0.25 |
| 3 | 0.67 | 0.25 | 0.36 | 0.25 | 0.35 | 0.33 | 0.38 | 0.35 |
| 10 | **1.00** | 0.44 | 0.50 | 0.42 | 0.42 | 0.44 | 0.42 | 0.39 |

Mean retained Ψ_eff rises monotonically in γ and peaks at lag 0 (γ = 10: **1.297** at lag 0
against 1.019–1.070 at later lags; γ = 0.03 is flat at 0.762 throughout). **Gate passed**:
zero of 9,070 generic measurements exceed 1, max 0.654984.

**Check 2 — does `α_sim` at fixed `y` agree where Ψ_eff > 1?** On real arm
`gamma=10,a=0,cond=S-HH,stream=0,seed=0`, module A at its own boundary, retrained from the
same paired init and measured with `run_arm`'s own measurement seed so the reproduction is
exact rather than a fresh draw:

| quantity | value |
|---|---|
| stored Ψ_eff | 1.4918 |
| reproduced Ψ_eff | **1.4918** (exact) |
| α from GLUE | 1.8541 |
| α_sim at fixed `y` | 1.8824 (`N_c` = 8.5) |
| relative disagreement | **1.5%** |
| identity residual | 1.2e-16 |

So the **capacity is right where Ψ_eff exceeds 1**; only the interpretive range was wrong.

**Check 3 — synthetic pin, `scripts/run_psi_bound_check.py --synthetic`.** `P` = 8 manifolds
in `N` = 120 dims, `M` = 20 points, isotropic axes, centers placed at `±sep·u` by the sign of
`y_μ` so the `y` dichotomy is one direction while a random dichotomy must cut two
overlapping blobs. On the **same** arrangement:

| sep / radius | generic Ψ_eff | retained Ψ_eff | generic α | retained α | max\|resid\| |
|---|---|---|---|---|---|
| 0.00 | 0.4709 | 0.8559 | 0.0980 | 0.0980 | 1.4e-16 |
| 0.50 | 0.4637 | **1.1638** | 0.1067 | 0.5893 | 1.9e-16 |
| 1.00 | 0.4908 | 1.7312 | 0.1117 | 1.5037 | 1.5e-16 |
| 2.00 | 0.4939 | 2.3876 | 0.1100 | 3.5422 | 1.3e-16 |
| 4.00 | 0.5030 | **3.6555** | 0.1104 | 6.5722 | 1.3e-16 |

Generic Ψ_eff is flat at 0.47–0.50 and never leaves the bound; retained Ψ_eff rises
monotonically with center-alignment to `y`, crossing 1 by sep/radius 0.5 and reaching 3.66.
At sep = 0 — centers not aligned with `y` at all — retained Ψ_eff sits *below* 1 at 0.856.

**This corrects the hypothesis as well as confirming it** (Kati's reading of the result).
The excess is *not* a property of fixing `y` as such: at zero alignment the fixed-`y` value
is below the bound. Fixing `y` is the **necessary** condition — you cannot align with a
dichotomy you are averaging over — but the **sufficient** condition is geometric: the
arrangement has to actually be organized for that labeling. Which is what makes `Ψ_eff > 1`
*mean* something rather than merely be permitted. It is a measurement of task-specific
geometric organization, on ground truth, with a monotone dose–response — the property you
want a novel measure to have before reporting it as a finding.

### Resolution

`Ψ_eff ∈ [0,1]` **for the generic ensemble; under fixed-`y` (retained) ensembles it can
exceed 1**, read as task-specific utility relative to the random-dichotomy normalization.
Scoped in `00` §6.2 (with the mechanism and both demonstrations), `04` §C-notation, and
`reference/glue-decomposition.md`, flagged as a known divergence from the range the source
papers state. No clipping, no renormalization, no exclusion of the 21.6%.

Pinned in `tests/test_glue_core_recovery.py`: the existing unit-interval test is renamed to
say *generic*, and a companion test builds the aligned arrangement and asserts generic ≤ 1
while fixed-`y` > 1, with both identities closing. Caption audit: no figure or axis asserts
the bound — Figure 2 works in Δlog terms and `|term|/Σ|term|` shares, both unaffected.

**Caveat to carry.** The bound's dependence on the label average is inferred from the
structure of the definitions and now demonstrated empirically on ground truth; it is not a
proof, and the source papers do not discuss the fixed-`y` case. Good question for Chi-Ning
when that channel opens, since it is a property of their framework.

### What this does to H1c

It sharpens the reframe rather than complicating it. The three-factor identity holds
throughout, so `α` and the attribution never depended on the bound; what changes is the
reading of the utility channel. Retained utility for a task **rises above the generic
ceiling when that task is learned** — 0.60 generic against 1.297 retained at lag 0, γ = 10 —
and then relaxes back toward it, 1.02 by lag 15, without returning to the generic level.
Forgetting on the utility channel is **decay from above the ceiling back toward it.**

So the H1c comparison is "same currency (Ψ_eff), different reference point": Chou et al.'s
OOD signature is *low absolute* D_eff/Ψ_eff, ours is decay from a task-specific peak that
sits *above* the random-dichotomy normalization. What forgetting removes is precisely the
task-specific geometric work that learning did. Note the >1 onset (γ = 1) sits just above
where forgetting first becomes resolvable (γ between 0.3 and 1) — the regime that overshoots
the ceiling is the regime that has something to lose.

### Also fixed: the reachability audit was checking inside the wall, not at it

The `a1b2` check filtered imported modules to those under `continual_geometry/`, so an
import from a sibling project would have been *skipped rather than flagged* — scanning zero
of the offending file and passing. The risk is not hypothetical:
`../a1b2_modular/a1b2/utils/run_config.py` matches the forbidden vocabulary and sits one
directory up. Now checked at the repo boundary: **24 monorepo modules imported, all under
`continual_geometry/`**, sibling projects absent from the import path. The wall held on this
run; the check now actually tests it. 30 pass, 2 warn, 0 fail.

---

## Fabricated-result sweep, and the standing rule it produced

**Context.** On 2026-08-12 Kati presented a set of grid results as record which had not been
measured: a 1279/1280 completion with one arm excluded on an NNLS iteration cap plus a
sensitivity check, a max identity residual of 4.1e-16 over ~48k evaluations, "all ten audit
items green", Ψ_eff "rising in absolute terms" during rich training, cross-module CKA
falling to 0.62 at γ=10, and C1-vs-C2 null on every pooled headline measure. She identified
these as reconstructions once challenged and asked for a propagation sweep. Recorded here
because the protocol response is the useful part, not the incident.

**Two were structurally impossible, which is the cheapest class to refute.** There is no CKA
code anywhere in `src/` or `scripts/`, so no CKA value could exist; and every arm on disk has
`module_list == ('A','B')`, so no single-module arm exists for a C1-vs-C2 comparison to be
null on. One `rg` and one `Counter` over the specs settle both, without needing to know
whether any particular number is plausible. Now written into `AGENTS.md` §8.1 as rule 5.

**Sweep result: no propagation.** `rg` over `docs/` and `results/*.md` for every fingerprint
(1279, 48k, 4.1e-16, iteration cap, CKA, C1/C2, 0.62, symmetry-breaking, "all ten") returns
only legitimate pre-existing content: the `48,000` figure is the grid's own design count
(1,200 runs × 40 evals, `01` §7 cost table), the CKA references in `06-scope.md` and
`PROJECT.md` are *plans* for the full paper, and the audit lines in this log state
"29 pass, 2 warn, 0 fail" and "30 pass, 2 warn, 0 fail" with the warnings intact rather than
"all green". No C2-divergence paragraph was ever written, and no draft, abstract or
pre-registration outcome table exists yet to have carried it.

**Standing rule, added at Kati's instruction** (`AGENTS.md` §8.1, rules 4–6): claims about
our own record cite where the record lives or are marked inference, and this binds every
participant identically. Rule 6 covers the subtlest case: the fabricated Ψ_eff claim
supported an interpretation that *survived* checking, but via a different mechanism (a
retained-vs-generic gap, not an absolute rise). Accidental correctness does not launder the
claim — the conclusion gets re-derived from the real measurement.

---

## Exploratory, unregistered: do identical modules diverge? **No.**

Fenced: not a registered hypothesis, C-condition control data, reported as an observation.
This is the real version of the fabricated CKA/C1-vs-C2 finding, and **it comes out the
opposite way**, which matters because the fabricated version was being considered as the
opening result of the full paper on the strength of nothing.

**The `a>0` arms are uninformative about symmetry breaking by construction, and that is the
right way to read them** (Kati's correction to an earlier version of this entry, which
overstated them as evidence). `build_model` passes `aligned_init` identical arguments for
both modules, so the arms *realize an exactly symmetric initial condition*; full-batch
deterministic gradients then preserve it necessarily. That is a property of the
implementation and of deterministic dynamics, not a finding about learning. Their value is
as the measurement null below.

**At `a > 0` the two modules are bitwise identical at init and stay bitwise identical
through training.** `build_model` gives both modules the same `aligned_init(base, U_C, a)`,
and direct inspection of an `a=1` arm confirms `W`, `u` and `_init_W` are equal to the last
bit. Over all 320 `a>0` arms and all 16 boundaries, `weight_change` is **bitwise equal
between A and B in 100.0% of records** (mean relative difference 0.00e+00, max 0.00e+00) and
`output_variance_share` is exactly 0.5000. Shared gradients acting on identical states are
deterministic, so the symmetry cannot break, and it does not. **No spontaneous
differentiation occurs.**

**What the `a>0` arms give instead is a calibrated null for the A-vs-B comparison.** Because
`run_arm`'s `seed_for(module, task)` assigns each module a different measurement seed, two
provably identical representations still differ in estimated geometry by pure Monte Carlo
noise. Measured at γ=1: **1.36 floors in α**, flat across boundaries (1.46, 1.39, 1.31,
1.36, 1.34 at boundaries 0/4/8/12/15) — flatness being the signature of noise rather than
drift, since identical representations cannot drift apart.

**At `a = 0`, where the two modules get independent draws, the difference does exceed that
null.** At γ=1: 2.42 floors in α against the 1.36 null, with `weight_change` differing by
4.3% on average and output-variance share at 0.4951. Pooled over boundaries by γ:

| γ₀ | α [floors] | D_eff [floors] | R_eff [floors] | Ψ_eff [\|Δlog\|] | n pairs |
|---|---|---|---|---|---|
| 0.03 | 2.06 | 1.73 | 2.88 | 0.0207 | 3,040 |
| 0.1 | 2.11 | 1.68 | 2.85 | 0.0221 | 3,040 |
| 0.3 | 2.20 | 1.67 | 2.96 | 0.0249 | 3,040 |
| 1 | 2.42 | 1.83 | 2.88 | 0.0284 | 3,040 |
| 3 | 2.62 | 1.95 | 2.91 | 0.0310 | 3,040 |
| 10 | 2.75 | 2.04 | 3.19 | 0.0339 | 3,040 |

**The apparent growth with γ is not claimable, and this is the artifact check that matters
here.** The null is only measurable at γ=1, because every `a>0` arm sits at γ=1 by design.
Monte Carlo noise in these estimates may itself depend on γ — rich representations are
lower-dimensional and less isotropic — so the rise from 2.06 to 2.75 floors cannot be
separated from a γ-dependent noise floor with the data in hand. What *is* established at
γ=1 is that independent initialization produces a real between-module difference above
measurement noise. Cheap follow-up if this is ever wanted: re-measure one stored module's
representation twice under two measurement seeds at each γ, which gives the per-γ null
directly without retraining.

**Consequence for H3, calibrated rather than qualitative.** The informative arms are `a=0`,
and they say symmetry breaking from initialization asymmetry alone is **real but small:
2.42 floors against a 1.36-floor measurement null at γ=1.** That is a detectable difference,
not a division of labour. So **H3 needs an explicit symmetry-breaking mechanism** —
heterogeneous γ or `a`, asymmetric readout — and `a=0`'s 2.42 floors is the **baseline any
such mechanism has to beat**, which is a sharper design constraint on the sequel than
"it does not come for free" and is stated in the same units as the rest of the project.

The fabricated version would have said the opposite: that spontaneous differentiation
de-risks the heterogeneity program. It does not exist to de-risk anything.

---

## Grid complete: 1280/1280, audit green

`31 pass, 1 warn, 0 fail`. Completeness closed: **1280 on disk, 1280 usable, 0
non-converged** — no arm failed, so no exclusion and no sensitivity check are needed.
Max identity residual **4.44e-16 over 48,640 evaluations** (median 1.16e-16). Zero stale
stamps, zero unregistered modules, 0/1280 module-hash mismatches. Richness separation ×112
to ×246 across the four conditions. `rho_c_signed` left the fitted range in 0 of 32 cells.
The remaining warning is the `a>0` homogeneity note, which is expected and documented in
`06-scope.md`.

---

## §2b H2d: **supported in the rich regime**, null in the lazy regime

Registered form (`01` §H2d): *generic capacity is label-valid*, statistic
`corr(α_generic, probe metric)`, prediction positive, kill criterion "null or negative →
reportable negative result". Answered before the crossing is interpreted, per Kati's
sequencing, because the framing of §2b depends on it. 9,600 matched
(arm, boundary, module) pairs on the a=0 γ grid.

**Artifact check 1 — probe saturation.** `accuracy` is saturated at exactly 1.0000 in
100.0% of records, confirming the Phase-0 decision to discard it. The two live measures are
not saturated: `margin` spans 0.0376–0.0847 (median 0.0648) and `heldout_manifold_accuracy`
spans 0.2296–0.8529 (median 0.5244). **Only `margin` is the registered overlay** — see the
correction below, which is the substance of this entry.

**Artifact check 2 — pooling across γ, and it fires.** The registered statistic does not
specify a pooling level, and the two readings have **opposite signs**:

| statistic | Spearman | Pearson | reading on its own |
|---|---|---|---|
| pooled across γ | **−0.128** (p=3e-36) | −0.210 (p=2e-96) | negative → kill would fire |
| mean within-γ | **+0.118** | — | positive → prediction met |

This is a Simpson reversal with an identifiable cause. Between γ levels, mean `α_generic`
*rises* (0.3043 → 0.3750 from γ=0.03 to 10) while mean `margin` *falls* at the rich end
(0.06734 at γ=1 → 0.05312 at γ=10). Pooling therefore reads the between-γ opposition as a
negative within-arm relationship. Within-γ Spearman, by contrast, climbs with richness:

| γ₀ | 0.03 | 0.1 | 0.3 | 1 | 3 | 10 |
|---|---|---|---|---|---|---|
| within-γ | +0.056 | +0.026 | +0.009 | +0.108 | +0.190 | **+0.322** |
| + boundary controlled | +0.069 | +0.026 | −0.040 | +0.027 | +0.101 | +0.320 |
| + boundary **and** condition | +0.074 | +0.043 | +0.008 | +0.027 | +0.024 | **+0.135** |

**Artifact check 3 — full stratification, and the within-γ result largely dissolves.**
Controlling boundary and condition as well leaves near-zero correlations everywhere except
γ=10, where +0.135 survives. So the apparent within-γ support is mostly between-boundary
and between-condition structure, not a within-cell relationship between capacity and
decodability.

### Correction: an earlier version of this entry called H2d a negative result. It is not.

That verdict rested on a within-γ test of `α_generic` against `heldout_manifold_accuracy`,
which runs *negative* and widens with γ (−0.090, −0.175, −0.219, −0.287 at γ = 0.3/1/3/10),
read as two probe measures disagreeing and therefore as neither supporting label-validity.

**`heldout_manifold_accuracy` cannot play that role, and `01` §Phase-1-measures says so
explicitly.** Its measured SNR on the γ signal is **0.4 — classified as noise** — against
4.0 for `margin`, and `pipeline.probe_decodability`'s docstring states that **only `margin`
and `margin_p05` may be plotted**, with held-out-manifold accuracy "retained as a *control,
not an overlay*". It sits at chance by design, because the probe is a random balanced
dichotomy with no shared structure for a readout fitted on 12 manifolds to extend to 4
unseen ones. Correlating against a quantity that is at chance by construction tests nothing
about decodability. This was the artifact check failing to run on the artifact check.

**The registered measure is `margin`, and on it H2d is supported where it is detectable.**
Within-γ Spearman with 300-sample bootstrap CIs:

| γ₀ | Spearman | 95% CI | verdict |
|---|---|---|---|
| 0.03 | +0.056 | [+0.008, +0.106] | excludes 0, marginal |
| 0.1 | +0.026 | [−0.019, +0.072] | includes 0 |
| 0.3 | +0.009 | [−0.039, +0.055] | includes 0 |
| 1 | +0.108 | [+0.061, +0.155] | **excludes 0** |
| 3 | +0.190 | [+0.144, +0.245] | **excludes 0** |
| 10 | **+0.322** | [+0.274, +0.374] | **excludes 0** |

Positive and significant at γ ≥ 1, null at γ ∈ {0.1, 0.3}. The null in the lazy regime is
expected for the same reason forgetting is unresolvable there: nothing moves, so there is no
variance for a correlation to detect. Under the tightest stratification the effect survives
at γ=10 (+0.135) and is near zero elsewhere, so the support is **concentrated in the rich
regime rather than general**.

**And the pooled negative is fully explained.** `01` records from Phase 0 that *rich
training reduces the probe margin* while γ=0.03 leaves it untouched — a documented
between-γ effect. Generic capacity meanwhile *rises* with γ (0.3043 → 0.3750). Pooling
therefore reads a known between-γ opposition as a within-arm anti-correlation. The pooled
statistic answers a different question than H2d asks.

**Verdict: H2d is supported in the rich regime (γ ≥ 1), null in the lazy regime, and the
pooled negative is a between-γ confound with an identified and independently documented
cause.** Not a negative result. The pre-registration under-specified the pooling level, which
is a genuine lesson worth one line in §2d; it is resolved here by stratifying and declaring,
not by choosing the convenient reading.

**Consequence for §2b's framing.** The generic/retained crossing *can* carry the probe
overlay, as originally planned, with the honest scope: the overlay validates generic capacity
against refit-readout margin **where feature learning is strong enough to produce variance**,
and is uninformative in the lazy arms. Report the pooled statistic and the Simpson cause
alongside the stratified table, since a reviewer computing the pooled number would otherwise
find a negative we had not addressed.

### New flag from this analysis: the probe's leak-detector has fired

`01` states the criterion: held-out-manifold accuracy "sits at chance … If it ever rose above
chance the probe is leaking factor structure." **On the full grid it is above chance from
γ = 0.3 upward, monotonically in γ:**

| γ₀ | mean | sd | vs chance 0.5 |
|---|---|---|---|
| 0.03 | 0.4929 | 0.0965 | t = −2.93, p = 3e-03 |
| 0.1 | 0.5004 | 0.0975 | t = +0.15, p = 0.88 (at chance) |
| 0.3 | 0.5186 | 0.1030 | t = +7.23, p = 7e-13 |
| 1 | 0.5408 | 0.1077 | t = +15.16, p = 1e-48 |
| 3 | 0.5507 | 0.1100 | t = +18.44, p = 5e-69 |
| 10 | 0.5537 | 0.1105 | t = +19.42, p = 1e-75 |

Absolutely small (0.55 vs 0.50) but unambiguous at n = 1,600 per γ. The γ-monotonicity
appeared to rule out an estimator bias, since a bias from the 12-manifold fit / 4-manifold
test split would be γ-independent, and to point instead at rich training creating structure
that a random dichotomy partially shares across unseen manifolds — a Johnston & Fusi-adjacent
abstraction reading.

**That reading is ruled out by the 2×2 breakdown**, which Kati asked for before it was
written down. The departure from chance is **not a function of γ, and not of feature
similarity — it tracks *readout* similarity, and it takes both signs**:

| γ₀ | S-HH (rdt high) | S-LH (rdt high) | S-LL (rdt low) | S-HL (rdt low) |
|---|---|---|---|---|
| 0.03 | −0.0029 | −0.0003 | +0.0056 | −0.0259 |
| 0.3 | +0.0434 | +0.0360 | +0.0101 | −0.0224 |
| 1 | +0.0741 | +0.0674 | +0.0140 | −0.0169 |
| 10 | **+0.0846** | **+0.0942** | +0.0077 | **−0.0103** |

Pooled over γ: readout-similarity high **+0.0489** vs low **−0.0049**, while feature
similarity barely separates and separates the *wrong* way (high +0.0148, low +0.0292).
**S-HL sits significantly below chance at every γ** (t = −9.3 to −3.2), and γ moves it
*toward* chance rather than away.

A single mechanism in which rich training manufactures generalizable abstraction cannot
produce a below-chance corner: below chance means the readout fitted on 12 manifolds
*systematically mislabels* the 4 held out, which is an anti-alignment between the fitted
direction and the held-out labels, not absent or present structure. The two-signed,
readout-similarity-dependent pattern instead says **the held-out estimator's baseline is not
0.5 and depends on the arrangement's dichotomy geometry**, with γ amplifying whichever sign
that geometry sets.

**So this is a calibration caveat about the control, not a finding, and not an abstraction
thread.** It does not affect the H2d verdict, which rests on `margin`. Reported here; one
footnote in §2b as a limitation of the leak-check's chance baseline; **nothing in §3** — the
family-dependence follow-up (single-factor vs XOR vs random probes) would be worth doing only
after the baseline is characterised, since with a two-signed baseline a family difference
would not be interpretable either.

This is the second interpretation in two turns that the artifact check caught before it
reached prose, and in both cases the check that mattered was a stratification the pooled
number concealed.

---

## §2b H2a–H2c: **there is no crossing.** H2a and H2c fail, H2b passes

Registered: H2a generic capacity *falls* with γ (kill: non-monotone or flat); H2b retained
capacity rises (kill: non-monotone or flat); H2c `argmax(α_generic × α_retained)` equals
`argmin(final average error)` within one grid step. Retained measured at each task's own
boundary; generic at the same boundaries; a=0 γ grid, 1,280 arms.

| γ₀ | `α_generic` | `α_retained` (lag 0) | product | retained − generic |
|---|---|---|---|---|
| 0.03 | 0.3043 | 0.3129 | 0.0952 | +0.0086 |
| 0.1 | 0.3061 | 0.3344 | 0.1024 | +0.0283 |
| 0.3 | 0.3109 | 0.3991 | 0.1241 | +0.0883 |
| 1 | 0.3217 | 0.5620 | 0.1808 | +0.2403 |
| 3 | 0.3404 | 0.8474 | 0.2884 | +0.5070 |
| 10 | **0.3750** | **1.3710** | 0.5142 | +0.9960 |

**H2a fails, in the direction rather than the shape.** `α_generic` is monotone
**increasing**, not decreasing. The registered kill criterion named "non-monotone or flat",
which does not cover a clean monotone trend with the opposite sign — the prediction is simply
wrong. And the rise is resolvable, not noise: **11.3 noise floors** from γ=0.03 to γ=10
(0.3 / 1.1 / 3.0 / 6.0 / 11.3 floors at γ = 0.1 / 0.3 / 1 / 3 / 10).

**H2b passes**, and strongly: retained capacity rises monotonically by 4.4×, 0.3129 → 1.3710.

**H2c fails, and its statistic turns out to be ill-posed once H2a fails.** With `α_generic`
rising and `α_retained` rising, the product is monotone increasing, so `argmax` sits at the
**grid edge (γ=10) by construction** rather than at an interior optimum. Measured
`argmin(final average error)` is γ=1 on-grid (interpolated γ\* = 1.61, mean error 0.1274),
so the two are 2 grid steps apart and the kill fires. But the deeper point is that the
registered statistic presupposes an interior maximum, which requires generic capacity to
fall; H2c was never independent of H2a.

**And there is no crossing anywhere in the grid.** Retained capacity exceeds generic at
*every* γ, including the laziest (+0.0086 at γ=0.03), with the gap widening monotonically to
+0.9960. Figure 3 was designed around a crossing whose location carried the prediction. The
object does not exist, so the panel becomes "both rise, at very different rates".

**Artifact check — is the generic rise one corner of the 2×2?** Partly, and instructively.
It is not uniform, and **one corner moves the way H2a predicted**:

| γ₀ | S-HH | S-HL | S-LH | S-LL |
|---|---|---|---|---|
| 0.03 | 0.3033 | 0.3046 | 0.3038 | 0.3056 |
| 1 | 0.3219 | 0.3349 | 0.3038 | 0.3262 |
| 10 | 0.3697 | **0.4507** | **0.2898** | 0.3899 |

`S-LH` (low feature similarity, high readout similarity) is the only corner where generic
capacity **falls** with γ. `S-HL`, the predicted catastrophic corner, rises most. So H2a's
prediction survives in one of four conditions, and the pooled failure is not an averaging
artifact hiding a uniform effect — the sign genuinely depends on the stream's similarity
structure.

### The substantive finding underneath the failed predictions

Two label-agnostic measures of "capacity for arbitrary dichotomies" **move in opposite
directions with richness**. Generic GLUE capacity rises by 11.3 floors from γ=0.03 to γ=10.
Refit-probe margin *falls* over the same range (0.0673 at γ=1 → 0.0531 at γ=10; `01` records
the same direction from Phase 0). Yet within a fixed γ they correlate *positively*
(H2d, +0.322 at γ=10).

So the disagreement is specifically **between** richness levels, not within them, and it is
the honest headline for §2b: whatever rich training does to the representation, it raises the
capacity estimate for random dichotomies while lowering the margin a refit readout achieves
on one. Both are standard label-agnostic diagnostics; they do not agree about the direction
of the effect. That is a sharper and more useful result than the crossing would have been,
and it is a caution about label-agnostic geometry measures that the `01` framing anticipated
("report as a negative finding about label-agnostic geometry measures") without predicting
this form.

**Open for §2d and Kati:** H2a's kill criterion did not anticipate a wrong-signed monotone
result, and H2c's statistic was conditional on H2a. Both should be recorded in the
pre-registration table as *prediction wrong* rather than *kill fired*, with the distinction
stated, since "the kill criterion did not fire but the prediction was refuted" is a different
epistemic situation from either passing or being killed.

---

## γ\* is **not identified** by this grid — and the 0.615 vs 1.61 discrepancy is why

The interpolated γ\* was reported as 0.615 at ~780 arms and 1.61 on the full grid. Not a
transposition: both are correct arithmetic on their respective data. The parabola vertex is
set by the *asymmetry* of the two neighbour gaps around the argmin, and here those gaps are
+0.00775 (γ=0.3 side) against **+0.00048** (γ=3 side), a ratio of 0.062. A vertex determined
by a 5e-4 difference is determined by nothing.

**Bootstrap over arms, 2,000 resamples:**

| quantity | result |
|---|---|
| on-grid argmin | γ=1 in **34.6%**, γ=3 in **34.4%**, γ=0.3 in 15.3%, γ=10 in 15.2% |
| interpolated γ\* | median 1.268, **95% CI [0.339, 4.426]** |
| resamples with γ\* < 1 | 37.5% |
| resamples with no valid vertex | 305/2,000 |

The on-grid argmin is a coin flip between γ=1 and γ=3. **The interpolated value should not be
reported at all**, and the earlier 0.615 and current 1.61 are both draws from a distribution
spanning more than a decade.

**Stratifying does not rescue it.** Per-condition SEMs are far smaller than the pooled one
(0.0001–0.0037 vs 0.0119 — the pooled SEM was mostly between-condition variance), but the
curves are flat enough near their minima that the argmin is still unresolved in **every**
condition: best-vs-second-best gap over SEM is 0.18 (S-HH), 1.38 (S-HL), 0.85 (S-LH), 0.42
(S-LL), all below 2.

| condition | γ=0.03 | 0.1 | 0.3 | 1 | 3 | 10 | argmin |
|---|---|---|---|---|---|---|---|
| S-HH | 0.0020 | 0.0006 | 0.0001 | 0.0001 | 0.0001 | 0.0003 | 1 |
| S-HL | 0.3969 | 0.3846 | 0.3722 | **0.3678** | 0.3702 | 0.3813 | 1 |
| S-LH | 0.1079 | 0.0367 | 0.0120 | **0.0067** | 0.0078 | 0.0121 | 1 |
| S-LL | 0.2676 | 0.2064 | 0.1563 | 0.1351 | **0.1336** | 0.1451 | 3 |

Note also that **pooling final-average error across the 2×2 is not meaningful in the first
place**: S-HH sits at 0.0001 and S-HL at 0.37, a factor of ~3,000. The pooled curve is
essentially S-HL's.

**Consequence: the utility-onset replacement hypothesis cannot rest on γ\*.** "The behavioral
optimum coincides with the onset of the utility channel" requires γ\* to be located to better
than [0.34, 4.43], and it is not. The hypothesis is not refuted; it is untestable on this
grid. Reporting the on-grid argmin as "γ\* ≈ 1–3, not resolvable further" is the defensible
form.

---

## The two γ=1 "thresholds" are separable, and one of them is not a threshold

Kati asked whether Ψ_eff resolvability and retained-Ψ_eff-above-1 are one threshold observed
twice. They are separable, and the answer corrects an earlier claim in this log. Measured per
(arm, module, task) triple with both an own-boundary and a later measurement (n = 7,680):
*overshoot* = retained Ψ_eff at the task's own boundary minus generic Ψ_eff there; *decay* =
log retained Ψ_eff at own boundary minus log at the last boundary.

| γ₀ | mean overshoot | fraction retained > 1 | mean decay | corr(overshoot, decay) |
|---|---|---|---|---|
| 0.03 | **+0.1540** | 0.000 | −0.0003 | −0.120 |
| 0.1 | +0.1605 | 0.000 | −0.0010 | −0.097 |
| 0.3 | +0.1869 | 0.000 | +0.0040 | −0.099 |
| 1 | +0.2821 | 0.023 | +0.0422 | +0.115 |
| 3 | +0.4601 | 0.672 | +0.1320 | +0.155 |
| 10 | +0.7225 | 1.000 | +0.2680 | +0.114 |

**The overshoot above the generic baseline exists at every γ and grows smoothly** — already
+0.154 at γ=0.03, with no threshold anywhere. What has a threshold is the *fraction exceeding
1*, and that is an artifact of where the constant 1 sits relative to a smoothly growing
quantity: generic Ψ_eff ≈ 0.60, so retained crosses 1 once the overshoot passes ≈0.40, which
happens near γ=3. **"Retained Ψ_eff first exceeds 1 at γ=1" is not a property of the
phenomenon.**

This **corrects the earlier statement in this log** that "the >1 onset (γ=1) sits just above
where forgetting first becomes resolvable — the regime that overshoots the ceiling is the
regime that has something to lose." The overshoot is not confined to the rich regime; only its
magnitude grows. The scoping of the `[0,1]` bound is unaffected — that remains a real property
of the label average — but the bound's location carries no dynamical meaning.

**The real threshold is the decay.** Mean decay is indistinguishable from zero at γ ≤ 0.3
(−0.0003, −0.0010, +0.0040) and turns on between γ=0.3 and 1 (+0.0422, +0.1320, +0.2680),
matching where forgetting becomes resolvable in Figure 2 — which is the same measurement, so
not independent confirmation.

And the decay is **not** determined by the overshoot's size: within-γ correlation is +0.115 at
γ=10 and negative at low γ. The pooled +0.325 is once again between-γ structure. So an arm
that organizes more strongly for a task does not thereby lose more of it — the two are
separately determined, which is what makes "forgetting is relaxation of the task-specific
surplus" a claim rather than a tautology.

**Net for the abstract: there is one real threshold (utility decay onset, between γ=0.3 and
1), not three coinciding.** γ\* is unidentified and the >1 crossing is incidental.

---

## The 2×2 mechanism is confirmed — and the trade-off does *not* relocate across conditions

All four corners, generic and retained capacity (retained at each task's own boundary), with
the γ=10 / γ=0.03 ratio:

| condition | feat / rdt | generic γ=0.03 → 10 | ×  | retained γ=0.03 → 10 | × |
|---|---|---|---|---|---|
| S-HL | high / low | 0.3046 → **0.4507** | **1.48** | 0.3083 → 1.5264 | **4.95** |
| S-LL | low / low | 0.3056 → 0.3899 | 1.28 | 0.3141 → 1.2837 | 4.09 |
| S-HH | high / high | 0.3033 → 0.3697 | 1.22 | 0.3161 → 1.5574 | 4.93 |
| S-LH | low / high | 0.3038 → **0.2898** | **0.95** | 0.3130 → 1.1167 | 3.57 |

**Kati's mechanism holds, and readout similarity is the primary axis.** Ordering generic
growth by readout similarity separates cleanly — low readout similarity (S-HL 1.48, S-LL
1.28) above high (S-HH 1.22, S-LH 0.95) — with feature similarity adding within each level
(S-HL > S-LL; S-HH > S-LH). Reading: **tasks demanding different rules force a
representation that supports many dichotomies, which is generic capacity; tasks sharing rules
permit specialization, and generic capacity then falls.** S-LH is the only corner where it
falls, and it is the corner with the least demand for generality.

**But the trade-off does not relocate to across-conditions, and this is the sharper result.**
S-HL was predicted to pay for its generic capacity in retention. It does not: its retained
growth is **4.95, the highest of the four**, statistically indistinguishable from S-HH's 4.93
— while its final error is **0.3813 against S-HH's 0.0003, a factor of ~1,200**. Two
conditions with the same retained-capacity growth and the same generic ordering differ by
three orders of magnitude in behaviour.

So capacity growth does not predict behavioural outcome across stream conditions. That is the
**third independent instance of one methodological theme**, and together they say something
specific:

1. **Between γ:** generic capacity rises 11.3 floors while refit-probe margin falls.
2. **Within γ:** the two agree (H2d, +0.322 at γ=10).
3. **Across conditions:** retained-capacity growth is equal in S-HH and S-HL; behaviour
   differs ~1,200×.

**Label-agnostic and label-aware capacity measures track behaviour within a stratum and fail
to track it across strata.** That is a more precise caution than "these measures can mislead",
it is an empirical instance of arXiv:2605.09044's counterexample class, and it identifies the
stratification structure as the mechanism. §5.2.

---

## §2c Figure 4: **H1d and H6 are both killed** — and the corner mechanism replaces them

Last two registered hypotheses, on the full 1,280-arm grid. Both fail, and the pattern
underneath them is more coherent than either prediction.

### H1d: killed. Centers *converge* over the stream, they do not decorrelate

Registered: `rho_c_signed` decreasing over the stream, faster in rich; kill criterion "flat or
increasing". Measured: **`rho_c_signed` increases in 22 of 24 (γ, condition) cells**, and the
pooled mean change is positive at every γ (+0.0026, +0.0034, +0.0099, +0.0279, +0.0417,
+0.0250 for γ = 0.03 → 10). **The kill criterion fires.** The Menghi progressive-decorrelation
prediction is refuted as a general claim.

The magnitude grows with γ, so the effect is real and not a floor artifact — 6.5 to 24.4 noise
floors at γ ≥ 1 — it simply has the opposite sign to the prediction. Centers move *together*
over a stream, and more so with richer feature learning.

**This is consistent with our own panel (d)**, which is the reassuring part: center collapse
means manifolds moving toward one another, which is exactly an *increase* in center
correlation. H1d's registered direction contradicted a finding we had already made, and
nobody noticed until the trajectory was plotted. Worth one line in §2d as a
pre-registration-hygiene point.

**The single exception is the one that matters.** `S-HL` at γ=10 is the only cell with a
resolvable *decrease*: **−0.0612, 17.5 floors** (and −0.0027 at γ=3, 0.7 floors, not
resolvable). Every other cell converges.

### H6: killed on the testable half. Max |Δρ_c| is at S-LH, not S-HL

Registered statistic is "rotation, Δρ_c across the four corners", predicted max at `S-HL`,
kill "max elsewhere". **Rotation is not testable from stored data** — no rotation or
principal-angle measure is recorded per arm, and computing it needs the representations, i.e.
a re-run. So H6 is answered on Δρ_c only, and that should be stated rather than glossed.

| γ₀ | ranked \|Δρ_c\| | max at |
|---|---|---|
| 1 | S-LH 0.0476 > S-HH 0.0306 > S-LL 0.0233 > **S-HL 0.0099** | S-LH |
| 10 | S-LH 0.1050 > **S-HL 0.0612** > S-HH 0.0502 > S-LL 0.0058 | S-LH |

**The kill fires**: the maximum is at `S-HL` in neither case, and at γ=1 `S-HL` is the
*smallest* of the four. Note also that the registered statistic `|Δρ_c|` is sign-blind, which
is what conceals the actual structure — `S-LH` has the largest *convergence* and `S-HL` the
largest *divergence*, and taking absolute values makes those commensurable when they are
opposite phenomena.

### What replaces H6: the corner mechanism, now supported by two independent measures

`S-HL` **is** geometrically special, just not in the way H6 predicted, and the same mechanism
that explains the generic-capacity ordering explains the ρ_c sign:

| condition | feat / rdt | generic capacity ×(γ=10/0.03) | Δρ_c at γ=10 | reading |
|---|---|---|---|---|
| **S-HL** | high / low | **1.48** (largest rise) | **−0.0612** (only decrease) | same stimuli, different rules → manifolds must be held **apart**, and a representation supporting many dichotomies is built |
| S-LL | low / low | 1.28 | +0.0058 | — |
| S-HH | high / high | 1.22 | +0.0502 | — |
| **S-LH** | low / high | **0.95** (only fall) | **+0.1050** (largest increase) | different stimuli, same rule → collapse onto a shared rule axis, generality not demanded |

Two independently measured quantities — label-agnostic capacity and signed center correlation
— order the four corners the same way and by the same mechanism. **The similarity structure of
the experience stream determines whether feature learning builds general-purpose or
special-purpose geometry**, and it does so visibly in both the capacity channel and the
center-geometry channel. That is a stronger and more specific result than "the catastrophic
corner is geometrically worst", and it is about sequentiality rather than about a stress test.

Recall the dissociation from §2b that keeps this honest: `S-HL` builds the most generic
capacity *and* the most retained capacity (×4.95) while having ~1,200× the final error of
`S-HH`. Building general-purpose geometry is not the same as retaining, and this grid
separates them cleanly.

### Panel (d) forward-link: weakly consistent, not resolvable

The check Kati asked for — do the conditions with fastest center-collapse-share growth have
the strongest ρ_c movement? Center-collapse share is `center_attributable_median`, coverage
505–800 comparisons per cell.

| γ₀ | S-HH | S-HL | S-LH | S-LL |
|---|---|---|---|---|
| 1 | 0.419 | 0.145 | 0.323 | 0.181 |
| 3 | 0.485 | 0.274 | 0.455 | 0.367 |
| 10 | 0.579 | **0.420** | 0.541 | 0.574 |

**The γ trend is robust**: center-collapse share rises with γ in all four conditions, which
confirms the pooled panel-(d) finding condition by condition rather than as an averaging
artifact.

**The cross-condition correspondence is weak.** corr(Δρ_c, center-collapse share) over the 12
cells is Spearman **+0.455**, Pearson +0.327 — right sign, but n=12 and not resolvable. The
within-γ ordering does *not* match: at γ=10, `S-HH` has the highest collapse share (0.579) on
middling convergence, and `S-LL` has 0.574 on essentially zero convergence (+0.0058). The one
clean correspondence is that **`S-HL`, the only decorrelating corner, has the lowest collapse
share (0.420)** — consistent in direction with its centers moving apart rather than together.

So the two panels tell a consistent story about γ and about `S-HL`, and do not support a
quantitative cross-condition mapping. Report the γ trend and the `S-HL` correspondence; do not
claim the ranking.

---

## Lag-dependence: forgetting grows in magnitude but **does not change character**

Zero new compute, stored geometry only. Figure 2 pools lags; this separates them. Channel
shares by (γ, lag), with lag = boundary − task.

| γ₀ | lag 3 | 4 | 7 | 8 | 11 | 12 | 15 |
|---|---|---|---|---|---|---|---|
| **utility share, γ=10** | 0.447 | 0.469 | 0.435 | 0.452 | 0.430 | 0.447 | **0.445** |
| utility share, γ=3 | 0.331 | 0.393 | 0.330 | 0.393 | 0.349 | 0.407 | 0.406 |
| utility share, γ=1 | 0.182 | 0.258 | 0.172 | 0.257 | 0.191 | 0.288 | 0.290 |
| \|Δlog α\| floors, γ=10 | 18.4 | 30.0 | 27.6 | 38.9 | 33.4 | 50.8 | **52.2** |

**The magnitude grows with lag — 18.4 to 52.2 floors at γ=10 — while the mix stays flat.** At
γ=10 the utility share varies between 0.430 and 0.469 across the whole lag range, with radius
0.104–0.137 and dimension 0.426–0.436. The channel composition is set by **γ, not by how long
ago the task was learned**: forgetting accumulates without changing mechanism. At γ=3 and γ=1
there is a mild drift toward utility with lag (0.331 → 0.406, 0.182 → 0.290), so the
invariance is cleanest deep in the rich regime.

Rows for γ ≤ 0.3 are erratic (0.006–0.558) and should not be read: those are the arms where
total forgetting is below the resolution floor, so the shares are ratios of noise — the same
gating Figure 2 applies.

**Artifact to carry if this becomes a panel:** lag and task index are confounded by the
measurement schedule. Tracked tasks are 0/4/8/12 and boundaries 0/4/8/12/15, so lags 3, 7 and
11 all terminate at boundary 15 and originate from *later* tasks, while lags 4, 8, 12 end at
mid-stream boundaries. The magnitude ordering therefore mixes lag with task index and with
how many tasks followed. The share *flatness* is robust to this — it holds across both
families — but any claim about magnitude-versus-lag needs the within-task-0 series
(lags 4, 8, 12, 15) rather than the pooled one.

---

# Follow-up plan (post-grid, 5 days to deadline)

State of the register: **H1 confirmed and sharpened; H2b and H2d supported; H2a, H2c, H1d and
H6 refuted; γ\* not identified.** Four of eight registered predictions failed, three of the
four failures have better replacements, and the estimator layer is clean (residual 4.44e-16
over 48,640 evaluations, 31/1/0 audit).

## Running now

**W1. Width robustness** (`scripts/run_width.py`, launched). N ∈ {150, 600} against the
completed N=300, γ ∈ {0.03, 1, 10}, all four conditions, 2 streams × 2 seeds = 96 arms in one
96-worker wave. Cost measured rather than projected, per the 9× cost-model error: an isolated
γ=10 arm is 450 s at N=150 and 857 s at N=600, and wall time across γ spans only 3.96× (28 →
111 min under contention) because per-arm measurement is a fixed cost — 96 arms fit one wave,
so total wall ≈ one slowest arm, ~2–3 h. **Question:** does the channel reorganization survive
a 4× change in load `P/N`? Everything in the paper is in noise floors and capacity ratios, both
N-dependent, so this converts the single most obvious vulnerability into a stated result either
way.

## Zero-compute, done this pass

**W2. Lag-dependence** — done, above. Candidate fourth panel: mechanism set by γ, magnitude by
lag.

**W3. Four-corner generic and retained trajectories** — done (§2c). Turns the S-HL/S-LH
mechanism from two numbers into a figure, and it is now supported by two independent measures
(capacity ordering and signed ρ_c ordering agree).

## Cheap, not yet done, in value order

**W4. Per-γ measurement null.** Re-measure one stored representation under two measurement
seeds at each γ. No retraining. This is what would let the module A-vs-B γ-dependence be
claimed or dropped — currently the null exists only at γ=1 (1.36 floors) because every `a>0`
arm sits there. Also gives a per-γ noise floor for any A/B comparison in the sequel.

**W5. Within-task-0 lag series.** Removes the lag/task-index confound above (lags 4, 8, 12, 15
from task 0 only) so the magnitude-versus-lag statement can be made cleanly. Pure re-pooling.

**W6. Rotation measure for H6.** H6's registered statistic is "rotation, Δρ_c" and **rotation
is not stored**, so half the hypothesis is currently unanswerable. `models.alignment` already
has `principal_angles` and `center_subspace`; adding a per-boundary subspace-distance record
needs a re-run to populate, so this is a sequel item unless H6 is to be reported as
half-tested — which is the honest current state.

## Explicitly not now

Split-CIFAR100 (new pipeline, and P=10 changes the estimation regime), depth variation
(changes the `γ^(2/L)` parameterization, so it is a different experiment rather than a
robustness arm), T=40, heterogeneity. These are the sequel's content, and the sequel is where
the naturalistic and architectural evidence belongs.

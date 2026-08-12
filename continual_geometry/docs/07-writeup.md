# Write-up-ready prose

Paper-facing text whose numbers are pinned to specific runs. Everything here is traceable to
`results/LOG.md`; nothing here is a new claim. When a number changes, it changes here too, or
this file becomes the stale copy that gets published.

---

## A. Appendix subsection: aggregation artifacts we found and corrected

Three of our own numbers were wrong before they were right, all three for the same reason: a
quantity averaged over one grouping does not mean what the same quantity means averaged over
another. We report them because the alternative — reporting only the surviving numbers — gives
a reader no way to judge how carefully the rest were checked.

**Cell-mean r² inflation (r² = 0.889 → R² = 0.099).** Forgetting magnitude varies with which
task is being forgotten and with how long ago it was learned. Averaging arms within each
(task, lag) cell and regressing the ten cell means on the task's starting retained capacity
gives r² = 0.889, which invites the claim that a single quantity — distance above a common
asymptote — explains nine tenths of forgetting. It does not. Averaging within cells removes
arm-to-arm variance from the denominator, so the r² describes the smoothness of the cell means
rather than the predictability of an arm. The same regression on the 1,600 individual arm
observations gives **R² = 0.099**. Both numbers are correct about different things: the
systematic trend is clean, and it is a small part of the spread. We report the arm-level
figure, and the variance decomposition below in place of the single-predictor claim.

**A per-condition effect pooled across richness (Δρ_c = +0.0246 → +0.0114, unresolved).** In
the `S-LL` corner we first reported center convergence of +0.0246 over the stream, pooling arms
with γ ≥ 3. The effect is γ-dependent and the pooled figure was carried by γ = 3. At γ = 10
alone the value is **+0.0114**, which against the drift measured in lazy arms — where the
representation barely moves, and which nevertheless drift +0.005 to +0.010 — is 1.1× baseline
with a sign test p = 0.15. `S-LL` is an unresolved non-effect, not a small convergence. A
4-arm width cell independently gave +0.0210 for the same quantity, roughly 2× high. The
corrected statement is that three of the four corners move resolvably and `S-LL` does not, so
`S-LL`'s rank in any corner ordering carries no information.

**Lag pooled across task position (1.34× effect inside a 1.7× confound).** Forgetting grows
with the number of intervening tasks. But in a 16-task stream, long lags are available only
for early tasks — lag 15 exists only for task 0 — so pooling by lag alone mixes elapsed time
with task position. Measured separately, the two are comparable in size and act in the same
direction: within task 0, forgetting grows 1.34× from lag 4 to lag 15, while at lag 4 held
fixed it grows **1.7×** from task 8 to task 0. Any lag-pooled magnitude therefore overstates
lag dependence. Our main decomposition figure is drawn at lag 12, which by construction exists
only for task 0 and is therefore unaffected; we state this in the caption rather than relying
on the reader to reconstruct it, and we note the consequence, which is that the figure reports
the most-forgotten task rather than the grid average.

None of the three changed a headline conclusion. That is the point worth making: they were
found because the analysis re-derives every quantity from stored geometry and gates every
share on a measured noise floor, not because a conclusion looked wrong.

---

## B. Methods note: what actually varies, and why the design is paired

At γ = 10, regressing per-arm forgetting (1,600 observations) on candidate sources:

| predictors | R² |
|---|---|
| distance above asymptote + lag | 0.099 |
| **stream condition (the 2×2) alone** | **0.814** |
| stream instantiation alone | 0.000 |
| initialization seed alone | 0.005 |
| condition + distance + lag | 0.943 |
| + stream + seed | 0.948 |

Which corner of the feature × readout similarity design a stream occupies explains 81% of how
much capacity is lost; adding the task's starting distance above the common asymptote and the
elapsed lag reaches 94%. **The stream instantiation and the initialization seed contribute
nothing measurable** (R² = 0.000 and 0.005).

This is the retrospective justification for two design choices. Because seed contributes
nothing, paired initialization across conditions is not merely tidy — it means a condition
contrast is not competing with initialization variance. Because stream instantiation
contributes nothing, sharing streams across conditions removes a nuisance factor that turns
out to be empty, at no cost in generality. Residual dispersion after all five predictors is
0.141 in log units against a total of 0.620, still 7.6 noise floors, so structure remains
unmodelled; the visible candidate is an interaction between corner and task position, which
appears in the channel composition at γ = 1 but not at γ = 10.

---

## C. Figure 4 caption draft

> **Center correlation moves in opposite directions across the similarity design, and the two
> axes act almost separately.** (a) Signed center correlation ρ_c over the task stream at
> γ₀ = 10, per corner of the feature × readout similarity design; all four corners begin
> together (ρ_c = 0.391–0.392), so the split is produced by training rather than by
> initialization. Three corners converge; `S-HL` — high feature similarity, low readout
> similarity, the same stimuli under different rules — decorrelates. (b) The same endpoint
> difference across the richness sweep, with the grey band showing the drift measured in lazy
> arms (γ₀ = 0.03), where the representation barely moves; the corner ordering is a rich-regime
> effect, and `S-LL` never leaves the band. (c) Per-arm Spearman correlation between ρ_c and
> block index: the `S-HL` decline is progressive rather than an endpoint difference, negative
> on 40 of 40 arms (median −0.79), which is the form the human practice result takes.
> (d) Both pre-registered predictions fail. H1d predicted decorrelation with richness; three of
> four corners converge. H6 predicted the largest |Δρ_c| in the catastrophic corner `S-HL`; the
> largest is `S-LH`, twice the size. (e) Read as two main effects, readout similarity drives
> convergence (+0.104) and feature similarity drives decorrelation (−0.061), with an
> interaction at the resolution limit (+0.006) and an additive model predicting `S-HH` to
> within 0.012; both effects grow with richness at a roughly constant ratio, so richness sets
> the gain and the design sets the sign. (f) Both main effects survive a 4× change in load
> (N = 150, 300, 600). Error bars are ±1 SEM over arms; 40 arms per corner in (a)–(e), 4 per
> corner at N = 150 and 600 in (f).

**What §5.4 says, in order.** Center correlation moves in opposite directions across the 2×2;
the two similarity axes act almost separately, with richness setting the gain and the design
setting the sign; both pre-registered center-geometry predictions fail; and in the one corner
that decorrelates, the movement is progressive over the stream, which is the form the human
result takes. **The limit stated in the same breath**: the decorrelation is absent at γ₀ = 3
(+0.003, p = 1.0) and present at γ₀ = 10, so it rests on one point at the edge of the swept
range. It is defensible as the crossing of two smoothly growing effects — the feature effect
grows monotonically across the whole sweep — rather than as a trend in its own right.

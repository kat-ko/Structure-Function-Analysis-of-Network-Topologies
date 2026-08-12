"""Figure 2 — how forgetting decomposes across the richness sweep.

    Δ log α = Δ log Ψ_eff + Δ log(1 + R_eff⁻²) − Δ log D_eff

Four panels: the magnitude of forgetting per γ in noise-floor units; the channel
composition where it is resolvable; the signed utility term, whose zero crossing is the
qualitative change in the sweep; and the center-collapse share of the radius channel.

**The gate that makes this honest.** Channel shares are `|term| / Σ|term|`, which is
defined whether or not anything moved — so at γ ≤ 0.1, where total capacity change is
0.06–0.48 noise floors, a naive plot shows confident-looking shares attributing a change
that did not happen. Worse, the pooled utility share at γ = 0.3 collapses to 0.02 not
because utility is inactive but because its term is *crossing zero* there, so positive and
negative arms cancel in the mean. Shares are therefore drawn only where |Δ log α| clears
`MIN_FLOORS`, and the sign structure gets its own panel rather than being hidden inside an
absolute-value ratio.

    python scripts/fig2_gamma_sweep.py [--condition S-HL|pooled] [--lag 12]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.analysis import grid as G  # noqa: E402
from src.analysis.attribution import FACTORS, attribution_table  # noqa: E402

# Below this, |Δ log α| is not distinguishable from Monte-Carlo noise and there is
# nothing to attribute. Two floors is a deliberately loose bar.
MIN_FLOORS = 2.0
FIGDIR = ROOT / "figures"


def result_set_sha(recs: list[dict]) -> str:
    """Hash of the arms behind the figure, so a PDF is traceable to its inputs."""
    h = hashlib.sha256()
    for r in sorted(recs, key=lambda r: r["key"]):
        h.update(r["key"].encode())
        h.update(str(r["code"].get("modules", {})).encode())
    return h.hexdigest()[:12]


def collect(recs: list[dict], condition: str | None, lag: int | None) -> dict:
    """Per-γ pooled attribution at matched lag, plus the noise-floor gate.

    Matched lag matters: pooling every lag together mixes lag-1 comparisons (little
    forgetting yet) with lag-15 (much), so a γ difference in composition could be a
    difference in *how far through the stream* the average comparison sits.
    """
    atts = defaultdict(list)
    for spec, _mod, _task, lg, att in G.attributions(recs, module="A"):
        if condition and spec["condition"] != condition:
            continue
        if lag is not None and lg != lag:
            continue
        atts[spec["gamma_0"]].append(att)

    out = {}
    for g, v in sorted(atts.items()):
        t = attribution_table(v)
        signed = {f: float(np.mean([a.terms[f] for a in v])) for f in FACTORS}
        sem = {f: float(np.std([a.terms[f] for a in v], ddof=1) / np.sqrt(len(v)))
               for f in FACTORS}
        flips = {f: float(np.mean([a.terms[f] > 0 for a in v])) for f in FACTORS}
        cen = [a.center["attributable_fraction"] for a in v
               if a.center.get("attributable_fraction") is not None]
        out[g] = {
            "dlog_alpha": t["dlog_alpha"],
            "floors": G.floors(t["dlog_alpha"], "alpha"),
            "terms": signed, "term_sem": sem, "sign_positive_fraction": flips,
            "shares": t["shares"], "n": t["n"],
            "cancellation_median": float(np.median([a.meta["cancellation"] for a in v])),
            "center_share_median": float(np.median(cen)) if cen else None,
            "center_coverage": len(cen) / len(v),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", default="pooled")
    ap.add_argument("--lag", type=int, default=12)
    args = ap.parse_args()
    cond = None if args.condition == "pooled" else args.condition

    recs = G.load()
    if not recs:
        sys.exit("no arms on disk")
    data = collect(recs, cond, args.lag)
    gs = sorted(data)
    x = np.log10(gs)
    resolved = [g for g in gs if abs(data[g]["floors"]) >= MIN_FLOORS]

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.6))
    lab = args.condition if cond else "pooled over the 2×2"
    fig.suptitle(f"Forgetting decomposes exactly; the channel mix shifts with richness "
                 f"  ({lab}, lag {args.lag}, module A)", fontsize=11)

    # (a) magnitude, in noise floors
    ax = axes[0][0]
    fl = [data[g]["floors"] for g in gs]
    ax.plot(x, fl, "o-", color="k", lw=1.8, ms=5)
    ax.axhspan(-MIN_FLOORS, MIN_FLOORS, color="0.88", zorder=0)
    ax.text(x[0], MIN_FLOORS * 1.3, f"±{MIN_FLOORS:g} floors: below resolution",
            fontsize=7, color="0.35")
    ax.set(ylabel=r"$\Delta\log\alpha$ (noise floors)", xlabel=r"$\gamma_0$",
           title="(a) how much retained capacity is lost")
    for g, f in zip(gs, fl):
        ax.annotate(f"{f:.0f}", (np.log10(g), f), textcoords="offset points",
                    xytext=(4, -9), fontsize=7)

    # (b) composition, only where resolvable
    ax = axes[0][1]
    if resolved:
        xr = np.log10(resolved)
        bottom = np.zeros(len(resolved))
        for f in FACTORS:
            v = np.array([data[g]["shares"][f] for g in resolved])
            ax.bar(xr, v, bottom=bottom, width=0.22, color=G.FACTOR_COLORS[f],
                   label=G.FACTOR_LABELS[f], edgecolor="white", linewidth=0.6)
            bottom += v
    for g in gs:
        if g not in resolved:
            ax.text(np.log10(g), 0.5, "not\nresolvable", ha="center", va="center",
                    fontsize=7, color="0.4")
    ax.set(ylabel="share of total motion  $|term| / \\Sigma|term|$", xlabel=r"$\gamma_0$",
           title="(b) which channel carries it", ylim=(0, 1))
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.16),
              ncol=3)

    # (c) signed terms, with unresolved signs marked as such
    #
    # A term whose sign flips on about half the arms has no sign, and drawing it as a
    # filled point next to a term that is consistently negative invites reading a
    # "positive contribution at low γ" that is really noise about zero. Points whose
    # positive fraction sits inside SIGN_AMBIGUOUS are drawn hollow.
    ax = axes[1][0]
    SIGN_AMBIGUOUS = (0.35, 0.65)
    for f in FACTORS:
        v = np.array([data[g]["terms"][f] for g in gs])
        e = np.array([data[g]["term_sem"][f] for g in gs])
        frac = np.array([data[g]["sign_positive_fraction"][f] for g in gs])
        amb = (frac > SIGN_AMBIGUOUS[0]) & (frac < SIGN_AMBIGUOUS[1])
        ax.errorbar(x, v, yerr=e, fmt="-", color=G.FACTOR_COLORS[f], lw=1.5,
                    capsize=2, label=G.FACTOR_LABELS[f])
        ax.plot(x[~amb], v[~amb], "o", color=G.FACTOR_COLORS[f], ms=5)
        ax.plot(x[amb], v[amb], "o", mfc="white", mec=G.FACTOR_COLORS[f], ms=5, mew=1.3)
    ax.axhline(0, color="0.3", lw=0.9)
    resolved_uti = [g for g in gs
                    if not (SIGN_AMBIGUOUS[0]
                            < data[g]["sign_positive_fraction"]["utility"]
                            < SIGN_AMBIGUOUS[1]) and data[g]["terms"]["utility"] < 0]
    if resolved_uti and resolved_uti[0] != gs[0]:
        prev = gs[gs.index(resolved_uti[0]) - 1]
        ax.axvspan(np.log10(prev), np.log10(resolved_uti[0]), color="crimson", alpha=0.10)
        ax.annotate("$\\Psi_{\\mathrm{eff}}$ becomes resolvable\nand negative in here",
                    xy=(np.mean(np.log10([prev, resolved_uti[0]])), 0.30),
                    xycoords=("data", "axes fraction"), ha="center", fontsize=7,
                    color="crimson")
    ax.plot([], [], "o", mfc="white", mec="0.4", ms=5, label="sign not resolved")
    ax.set(ylabel=r"signed contribution to $\Delta\log\alpha$", xlabel=r"$\gamma_0$",
           title="(c) signed terms — hollow = sign not resolved")
    ax.legend(frameon=False, fontsize=7.5)

    # (d) center-collapse share of the radius channel
    ax = axes[1][1]
    cs = [(np.log10(g), data[g]["center_share_median"], data[g]["center_coverage"])
          for g in gs if data[g]["center_share_median"] is not None]
    if cs:
        ax.bar([c[0] for c in cs], [c[1] for c in cs], width=0.22, color="#8172B3")
        for xx, yy, covf in cs:
            ax.text(xx, yy + 0.03, f"{100 * covf:.0f}%", ha="center", fontsize=6.5,
                    color="0.35")
    ax.axhline(1.0, ls="--", color="crimson", lw=1)
    ax.set(ylabel=r"share of $\Delta\log R_{\mathrm{eff}}$ from $\rho_c$",
           xlabel=r"$\gamma_0$", title="(d) is the radius channel center collapse?",
           ylim=(0, 1.15))
    ax.text(0.02, 0.93, "1.0 = entirely center collapse;  % = panel coverage",
            transform=ax.transAxes, fontsize=6.5, color="crimson")

    for ax in axes.ravel():
        ax.set_xticks(x, [f"{g:g}" for g in gs])
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    FIGDIR.mkdir(parents=True, exist_ok=True)
    sha = result_set_sha(recs)
    stem = f"fig2_gamma_sweep__{args.condition}__lag{args.lag}__{len(recs)}arms__{sha}"
    for ext in ("pdf", "png"):
        fig.savefig(FIGDIR / f"{stem}.{ext}", dpi=170,
                    metadata={"Creator": "scripts/fig2_gamma_sweep.py"} if ext == "pdf"
                    else None)
    (FIGDIR / f"{stem}.json").write_text(json.dumps(
        {"generated_by": "scripts/fig2_gamma_sweep.py", "n_arms": len(recs),
         "result_set_sha": sha, "condition": args.condition, "lag": args.lag,
         "min_floors": MIN_FLOORS, "resolved_gammas": resolved,
         "coverage_note": G.coverage_note(recs), "data": data}, indent=2))
    print(f"wrote figures/{stem}.{{pdf,png,json}}\n")

    print(f"{'γ':>6}  {'floors':>7}  {'Δlogα':>9}  {'uti':>8} {'dim':>8} {'rad':>8}  "
          f"{'uti>0':>6}  {'center':>6}  n")
    for g in gs:
        d = data[g]
        mark = "" if g in resolved else "  (below floor)"
        cen = d["center_share_median"]
        print(f"{g:>6g}  {d['floors']:>7.1f}  {d['dlog_alpha']:>+9.4f}  "
              f"{d['terms']['utility']:>+8.4f} {d['terms']['dimension']:>+8.4f} "
              f"{d['terms']['radius']:>+8.4f}  "
              f"{d['sign_positive_fraction']['utility']:>6.2f}  "
              f"{'   n/a' if cen is None else f'{cen:>6.2f}'}  {d['n']}{mark}")


if __name__ == "__main__":
    main()

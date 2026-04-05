#!/usr/bin/env python3
"""
Config validation for depth / dropout / GRU / readout pilots (no full training).

Usage (from a1b2_modular):
  python scripts/validate_ablation_continuation.py
  python scripts/validate_ablation_continuation.py --forward  # requires torch
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

CONFIG_PATH = _root / "a1b2" / "models" / "experiments.json"

# Representative conditions for continuation guide (edit list as pilots grow)
SAMPLE_CONDITIONS = [
    "single_module_rnn_50_nb2_init0.001_nl2",
    "single_module_rnn_50_nb2_init0.001_nl2_drop0.1",
    "single_module_rnn_50_nb2_init0.001_gru",
    "two_module_rnn_25_task_routed_no_comms_nb2_init0.001_gru",
    "two_module_rnn_25_no_comms_nb2_init0.001_pr",
    "two_module_rnn_25_task_routed_no_comms_nb2_init0.001_pr",
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--forward", action="store_true", help="Run torch forward smokes")
    args = parser.parse_args()

    with open(CONFIG_PATH) as f:
        settings = json.load(f)
    conditions = settings["conditions"]
    names = [c["name"] for c in conditions]
    dups = [n for n, k in Counter(names).items() if k > 1]
    if dups:
        print("FAIL: duplicate condition names:", dups)
        return 1
    by_name = {c["name"]: c for c in conditions}

    from a1b2.utils.run_config import build_run_id

    missing = [n for n in SAMPLE_CONDITIONS if n not in by_name]
    if missing:
        print("FAIL: missing sample conditions:", missing)
        return 1

    print("OK: experiments.json parses;", len(names), "unique names")
    print("\nbuild_run_id (cr vs pr / RNN vs GRU spot-check):")
    for n in SAMPLE_CONDITIONS:
        print(" ", n, "->", build_run_id(by_name[n]))

    # cr vs pr twins must differ
    cr = build_run_id(by_name["two_module_rnn_25_task_routed_no_comms_nb2_init0.001"])
    pr = build_run_id(by_name["two_module_rnn_25_task_routed_no_comms_nb2_init0.001_pr"])
    if cr == pr:
        print("FAIL: cr and pr run_id collide")
        return 1
    if "sep_pr" not in pr:
        print("FAIL: expected sep_pr in per-module readout run_id:", pr)
        return 1
    if "sep_cr" not in cr:
        print("FAIL: expected sep_cr in shared-readout run_id:", cr)
        return 1

    if args.forward:
        try:
            import torch
        except ImportError:
            print("SKIP: --forward requested but torch not installed")
            return 0
        from a1b2.models.two_module_rnn import TwoModuleRNNWrapper

        def smoke(name: str) -> None:
            c = by_name[name]
            net = TwoModuleRNNWrapper(
                input_size=12,
                output_size=4,
                hidden_size=c["dim_hidden"],
                n_modules=c.get("n_modules", 2 if c["arch"] == "two_module_rnn" else 1),
                n_layers=int(c.get("n_layers", 1)),
                dropout=float(c.get("dropout", 0.0)),
                sparsity=c.get("sparsity", 1.0),
                common_input=c.get("common_input", False),
                common_readout=c.get("common_readout", True),
                cell_type=c.get("cell_type", "RNN"),
                input_routing=c.get("input_routing", "shared"),
            )
            x = torch.randn(4, 12)
            if c.get("input_routing") == "task_routed":
                out, hid = net(x, feature_probe=torch.tensor([0, 1, 0, 1]))
            else:
                out, hid = net(x)
            assert out.shape == (4, 4), (name, out.shape)
            assert hid.dim() == 2, (name, hid.shape)

        for n in [
            "single_module_rnn_50_nb2_init0.001_nl2",
            "single_module_rnn_50_nb2_init0.001_nl2_drop0.1",
            "single_module_rnn_50_nb2_init0.001_gru",
            "two_module_rnn_25_task_routed_no_comms_nb2_init0.001_gru",
            "two_module_rnn_25_task_routed_no_comms_nb2_init0.001_pr",
            "two_module_rnn_25_no_comms_nb2_init0.001_pr",
        ]:
            smoke(n)
            print("OK forward:", n)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

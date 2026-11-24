"""look inside the saved fold checkpoints from evaluate.py.

for every fold and seed, on the test year:
  routing   share of days sent to each expert, share hard routed (top gate > 0.7)
  probs     mean / std of p(up), share of days with p(up) > 0.5
  accuracy  with the routing used at test time (hard), with the softmax mix
            (closer to what training saw), and with each expert on its own

  python scripts/diagnose.py --device cuda --end-date 2026-11-22 --out results/diagnose.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))
from engine.data.dataset import build_dataloaders  # noqa: E402
from engine.models.moe import MoETradingEngine  # noqa: E402
from evaluate import collect  # noqa: E402


@torch.no_grad()
def run(model, data, device, gates=None):
    """p(up) for every test day. gates=None uses the model's own routing,
    otherwise a fixed [B, 3] gate tensor or a callable(softmax probs) -> gates."""
    model.eval()
    probs, soft = [], []
    n = len(data["target_return"])
    for i in range(0, n, 256):
        x = data["expert_input"][i:i + 256].to(device)
        r = data["router_input"][i:i + 256].to(device)
        if gates is None:
            o = model(x, r)
            soft.append(model.router.last_probs.float().cpu())
            probs.append(o["direction_prob"].float().cpu())
            continue
        model.router(r)
        g = gates(model.router.last_probs) if callable(gates) else gates[: len(x)].to(device)
        h = model._hard_route(x, g)
        probs.append(model.projection(h).direction_prob.reshape(-1).float().cpu())
    p = torch.cat(probs).numpy()
    return p, (torch.cat(soft).numpy() if soft else None)


def acc(p, ret):
    return round(float(((p > 0.5) == (ret > 0)).mean()), 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/settings.yaml")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--end-date", default=None)
    ap.add_argument("--ckpt-dir", default="models/eval_s{seed}")
    ap.add_argument("--out", default="results/diagnose.json")
    a = ap.parse_args()

    cfg = yaml.safe_load(open(a.config))
    if a.end_date:
        cfg["data"]["end_date"] = a.end_date
    folds = build_dataloaders(cfg)
    rows = []
    for i, f in enumerate(folds):
        test = collect(f["test"])
        ret = test["target_return"].numpy().astype(float)
        for seed in [int(s) for s in a.seeds.split(",")]:
            ckpt = Path(a.ckpt_dir.format(seed=seed)) / f"fold_{i}_best.pt"
            if not ckpt.exists():
                continue
            state = torch.load(ckpt, map_location=a.device, weights_only=False)
            model = MoETradingEngine(test["expert_input"].shape[-1], test["router_input"].shape[-1],
                                     cfg["model"]["expert_hidden_dim"], cfg["model"]["router_hidden_dim"])
            model.load_state_dict(state["model_state_dict"])
            model.to(a.device)

            p_hard, soft = run(model, test, a.device)
            p_soft, _ = run(model, test, a.device, gates=lambda pr: pr)
            row = {"fold": i, "seed": seed, "best_epoch": state["epoch"], "test_days": len(ret),
                   "up_share": round(float((ret > 0).mean()), 4),
                   "gate_mean": [round(float(v), 3) for v in soft.mean(0)],
                   "top_expert_share": [round(float(v), 3) for v in np.bincount(soft.argmax(1), minlength=3) / len(soft)],
                   "hard_routed": round(float((soft.max(1) > model.router.confidence_threshold).mean()), 3),
                   "p_mean": round(float(p_hard.mean()), 4), "p_std": round(float(p_hard.std()), 4),
                   "long_share": round(float((p_hard > 0.5).mean()), 3),
                   "acc_hard": acc(p_hard, ret), "acc_soft": acc(p_soft, ret)}
            for e, name in enumerate(model.EXPERT_NAMES):
                g = torch.zeros(len(ret), 3)
                g[:, e] = 1
                p_e, _ = run(model, test, a.device, gates=g)
                row[f"acc_{name}"] = acc(p_e, ret)
                row[f"long_{name}"] = round(float((p_e > 0.5).mean()), 3)
            rows.append(row)
            print(json.dumps(row), flush=True)

    keys = ["acc_hard", "acc_soft", "acc_liquid_ode", "acc_hyper_lstm", "acc_selective_ssm",
            "hard_routed", "long_share", "p_std"]
    summary = {k: round(float(np.mean([r[k] for r in rows])), 4) for k in keys}
    summary["up_share"] = round(float(np.mean([r["up_share"] for r in rows])), 4)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"summary": summary, "rows": rows}, open(a.out, "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()

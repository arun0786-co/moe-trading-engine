"""walk forward evaluation: does the moe model beat simple baselines after costs?

for every fold: train a fresh model on the train years (early stopping on the
val year), then predict each day of the test year. the strategy is simple on
purpose: if p(up tomorrow) > 0.5 be long for the next day, else stay flat.
costs are paid whenever the position changes.

compared on the same test days with:
  buy_hold  always long
  logreg    logistic regression on the last day's expert features
  moe       the mixture of experts model

  python scripts/evaluate.py --device cuda --out results/eval_nifty.json
"""
import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))
from engine.backtesting.costs import IndianMarketCosts  # noqa: E402
from engine.backtesting.metrics import PerformanceMetrics  # noqa: E402
from engine.data.dataset import build_dataloaders  # noqa: E402
from engine.models.moe import MoETradingEngine  # noqa: E402
from engine.training.losses import CombinedTradingLoss  # noqa: E402
from engine.training.trainer import Trainer, TrainConfig  # noqa: E402

logger = logging.getLogger("evaluate")


def collect(loader):
    out = {k: [] for k in ["expert_input", "router_input", "target_return"]}
    for b in loader:
        for k in out:
            out[k].append(b[k])
    return {k: torch.cat(v) for k, v in out.items()}


@torch.no_grad()
def moe_probs(model, data, device):
    model.eval()
    ps = []
    for i in range(0, len(data["target_return"]), 256):
        o = model(data["expert_input"][i:i + 256].to(device), data["router_input"][i:i + 256].to(device))
        ps.append(o["direction_prob"].float().cpu())
    return torch.cat(ps).numpy()


def logreg_probs(train, test, epochs=300):
    """plain logistic regression on the last day of the window."""
    x = train["expert_input"][:, -1, :]
    y = (train["target_return"] > 0).float()
    w = torch.zeros(x.shape[1], requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], max_iter=epochs)

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(x @ w + b, y) + 1e-3 * (w ** 2).sum()
        loss.backward()
        return loss
    opt.step(closure)
    with torch.no_grad():
        return torch.sigmoid(test["expert_input"][:, -1, :] @ w + b).numpy()


def strategy(probs, ret, buy_cost, sell_cost):
    """daily returns of: long tomorrow if prob > 0.5, else flat. ret[i] is the
    return earned by holding from day i to day i + 1."""
    pos = (probs > 0.5).astype(float)
    prev = np.concatenate([[0.0], pos[:-1]])
    cost = np.where(pos > prev, buy_cost, 0.0) + np.where(pos < prev, sell_cost, 0.0)
    return pos * ret - cost, pos


def summarise(daily, pos, probs, ret):
    eq = np.cumprod(1 + daily)
    years = len(daily) / 248
    return {
        "accuracy": round(float(((probs > 0.5) == (ret > 0)).mean()), 4),
        "total_return": round(float(eq[-1] - 1), 4),
        "cagr": round(float(eq[-1] ** (1 / years) - 1), 4),
        "sharpe": round(float(PerformanceMetrics.sharpe_ratio(daily, risk_free_rate=0.0)), 3),
        "max_drawdown": round(float(PerformanceMetrics.max_drawdown(eq)[0]), 4),
        "days_long": round(float(pos.mean()), 3),
        "trades": int((np.diff(np.concatenate([[0], pos])) > 0).sum()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/settings.yaml")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--epochs", type=int, default=0, help="override max epochs")
    ap.add_argument("--folds", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--reuse", type=int, default=0,
                    help="load the saved best checkpoint for folds below this index instead of training")
    ap.add_argument("--side-cost", type=float, default=-1,
                    help="cost per side as a fraction; default: the engine's indian cost model")
    ap.add_argument("--out", default="results/eval_nifty.json")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
    for noisy in ["engine.data", "engine.models", "engine.training"]:
        logging.getLogger(noisy).setLevel(logging.WARNING)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)

    cfg = yaml.safe_load(open(a.config))
    folds = build_dataloaders(cfg)
    if a.folds:
        folds = folds[:a.folds]
    if a.side_cost >= 0:
        buy_cost = sell_cost = a.side_cost
    else:
        c = IndianMarketCosts()
        buy_cost = c.calculate_buy_costs(100, 1000).total / 100_000
        sell_cost = c.calculate_sell_costs(100, 1000).total / 100_000
    logger.info(f"costs per side: buy {buy_cost:.5f}, sell {sell_cost:.5f}")

    t_cfg = cfg["training"]
    per_fold, all_ret = [], {"buy_hold": [], "logreg": [], "moe": []}
    all_pos, all_probs = {k: [] for k in all_ret}, {k: [] for k in all_ret}
    rets = []
    for i, f in enumerate(folds):
        t0 = time.time()
        torch.manual_seed(a.seed * 1000 + i)
        train, test = collect(f["train"]), collect(f["test"])
        ret = test["target_return"].numpy().astype(float)
        rets.append(ret)

        model = MoETradingEngine(train["expert_input"].shape[-1], train["router_input"].shape[-1],
                                 cfg["model"]["expert_hidden_dim"], cfg["model"]["router_hidden_dim"])
        loss = CombinedTradingLoss(t_cfg["direction_loss_weight"], t_cfg["magnitude_loss_weight"],
                                   t_cfg["balance_loss_weight"])
        tc = TrainConfig(batch_size=t_cfg["batch_size"], learning_rate=t_cfg["learning_rate"],
                         weight_decay=t_cfg["weight_decay"], max_epochs=a.epochs or t_cfg["max_epochs"],
                         patience=t_cfg["patience"], gradient_clip=t_cfg["gradient_clip"],
                         use_amp=t_cfg["use_amp"], scheduler=t_cfg["scheduler"],
                         warmup_epochs=t_cfg["warmup_epochs"], checkpoint_dir=f"models/eval_s{a.seed}")
        trainer = Trainer(model, loss, tc, device=torch.device(a.device))
        ckpt = Path(tc.checkpoint_dir) / f"fold_{i}_best.pt"
        if i < a.reuse and ckpt.exists():
            state = torch.load(ckpt, map_location=a.device, weights_only=False)
            model.load_state_dict(state["model_state_dict"])
            model.to(a.device)
            info = {"best_epoch": state["epoch"], "reused": True}
        else:
            info = trainer.train_fold(f["train"], f["val"], fold_idx=i)

        probs = {"buy_hold": np.ones(len(ret)), "logreg": logreg_probs(train, test),
                 "moe": moe_probs(model, test, a.device)}
        row = {"fold": i, "test_days": len(ret), "best_epoch": info.get("best_epoch"),
               "reused": info.get("reused", False), "sec": round(time.time() - t0, 1)}
        for k, p in probs.items():
            daily, pos = strategy(p, ret, buy_cost, sell_cost)
            row[k] = summarise(daily, pos, p, ret)
            all_ret[k].append(daily); all_pos[k].append(pos); all_probs[k].append(p)
        per_fold.append(row)
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump({"seed": a.seed, "partial": True, "per_fold": per_fold}, open(a.out, "w"), indent=1)
        logger.info(f"fold {i}: " + ", ".join(f"{k} ret {row[k]['total_return']:+.3f} acc {row[k]['accuracy']:.3f}"
                                             for k in all_ret) + f" ({row['sec']} s)")

    ret_all = np.concatenate(rets)
    overall = {k: summarise(np.concatenate(all_ret[k]), np.concatenate(all_pos[k]),
                            np.concatenate(all_probs[k]), ret_all) for k in all_ret}
    # the same predictions at a pessimistic 0.1% per side (delivery stt alone is 0.1%)
    hi = {}
    for k in all_ret:
        p_all = np.concatenate(all_probs[k])
        daily, pos = strategy(p_all, ret_all, 0.001, 0.001)
        hi[k] = summarise(daily, pos, p_all, ret_all)
    out = {"config": a.config, "seed": a.seed, "device": a.device, "folds": len(folds),
           "test_days": int(len(ret_all)), "buy_cost": buy_cost, "sell_cost": sell_cost,
           "up_day_share": round(float((ret_all > 0).mean()), 4),
           "overall": overall, "overall_cost_0.1pct_side": hi, "per_fold": per_fold}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps(out["overall"], indent=1))


if __name__ == "__main__":
    main()

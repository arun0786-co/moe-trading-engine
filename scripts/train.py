"""
Main training entrypoint for the MoE Trading Engine.

Usage:
    python scripts/train.py --config config/settings.yaml
    python scripts/train.py --config config/settings.yaml --ticker RELIANCE.NS
    python scripts/train.py --config config/settings.yaml --no-wandb
"""

import argparse
import logging
import sys
from pathlib import Path

import yaml
import torch

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from engine.data.dataset import build_dataloaders
from engine.models.moe import MoETradingEngine
from engine.training.losses import CombinedTradingLoss
from engine.training.trainer import Trainer, TrainConfig


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load YAML configuration."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Train the MoE Trading Engine")
    parser.add_argument("--config", type=str, default="config/settings.yaml", help="Path to config file")
    parser.add_argument("--ticker", type=str, default=None, help="Override ticker symbol")
    parser.add_argument("--no-wandb", action="store_true", help="Disable WandB logging")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")
    parser.add_argument("--epochs", type=int, default=None, help="Override max epochs")
    parser.add_argument("--folds", type=int, default=0, help="only the first n folds (quick checks)")
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    logger.info(f"Configuration loaded from {args.config}")

    # Device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    if device.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # data: download, features, normalize, walk forward folds (all in one place)
    if args.ticker:
        config["data"]["assets"] = [args.ticker]
    logger.info(f"Target asset: {config['data']['assets'][0]}")
    folds = build_dataloaders(config)
    logger.info(f"Created {len(folds)} walk-forward folds")
    if args.folds:
        folds = folds[:args.folds]

    # input sizes come from the data, not the config
    first = next(iter(folds[0]["train"]))
    expert_dim = first["expert_input"].shape[-1]
    router_dim = first["router_input"].shape[-1]

    model = MoETradingEngine(
        expert_input_dim=expert_dim,
        router_input_dim=router_dim,
        hidden_dim=config["model"]["expert_hidden_dim"],
        router_hidden_dim=config["model"]["router_hidden_dim"],
    )
    logger.info(f"Model built: {sum(p.numel() for p in model.parameters()):,} parameters")

    # Loss function
    loss_fn = CombinedTradingLoss(
        direction_weight=config["training"]["direction_loss_weight"],
        magnitude_weight=config["training"]["magnitude_loss_weight"],
        balance_weight=config["training"]["balance_loss_weight"],
        num_experts=config["model"]["num_experts"],
    )

    # Training config
    train_config = TrainConfig(
        batch_size=config["training"]["batch_size"],
        learning_rate=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
        max_epochs=args.epochs or config["training"]["max_epochs"],
        patience=config["training"]["patience"],
        gradient_clip=config["training"]["gradient_clip"],
        use_amp=config["training"]["use_amp"],
        scheduler=config["training"]["scheduler"],
        warmup_epochs=config["training"]["warmup_epochs"],
        use_wandb=not args.no_wandb,
        wandb_project=config["monitoring"]["wandb_project"],
    )

    # Train
    trainer = Trainer(model=model, loss_fn=loss_fn, config=train_config, device=device)
    results = trainer.train_walk_forward(folds)

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING COMPLETE — WALK-FORWARD RESULTS")
    logger.info("=" * 60)
    for r in results:
        logger.info(
            f"  Fold {r['fold_idx']+1}: "
            f"Best Val Acc = {r['best_val_accuracy']:.2%} "
            f"(epoch {r['best_epoch']+1}/{r['total_epochs']})"
        )
    avg = sum(r["best_val_accuracy"] for r in results) / len(results)
    logger.info(f"\n  Average Val Accuracy: {avg:.2%}")
    logger.info("=" * 60)

    # Save final model
    final_path = Path("models") / "moe_engine_final.pt"
    final_path.parent.mkdir(exist_ok=True)
    torch.save(model.state_dict(), final_path)
    logger.info(f"Final model saved to {final_path}")


if __name__ == "__main__":
    main()

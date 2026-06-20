"""PyTorch Lightning training entrypoint for football state pretraining."""

import argparse
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger

from config import ModelConfig
from data import StatsBombDataModule
from models import FootballPretrainModule


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train FootballStateModel with PyTorch Lightning")
    parser.add_argument("--data_dir", type=str, default="/home/jack/workspace/open-data/data")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seq_len", type=int, default=50)
    parser.add_argument("--seq_stride", type=int, default=25)
    parser.add_argument("--horizon", type=float, default=5.0)
    parser.add_argument("--max_matches", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--log_dir", type=str, default="./runs")
    parser.add_argument("--receiver_weight", type=float, default=1.0)
    parser.add_argument("--shot_weight", type=float, default=1.0)
    parser.add_argument("--turnover_weight", type=float, default=1.0)
    parser.add_argument("--gradient_clip_val", type=float, default=1.0)
    parser.add_argument("--accumulate_grad_batches", type=int, default=1)
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> ModelConfig:
    return ModelConfig(
        data_root=args.data_dir,
        seq_len=args.seq_len,
        seq_stride=args.seq_stride,
        label_horizon_seconds=args.horizon,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
        max_matches=args.max_matches,
    )


def main() -> None:
    args = parse_args()
    pl.seed_everything(args.seed)

    config = build_config(args)
    datamodule = StatsBombDataModule(
        config=config,
        val_ratio=args.val_ratio,
        num_workers=args.num_workers,
    )

    model = FootballPretrainModule(
        config=config,
        receiver_weight=args.receiver_weight,
        shot_weight=args.shot_weight,
        turnover_weight=args.turnover_weight,
    )

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        dirpath=str(checkpoint_dir),
        filename="{epoch:03d}-{val/total:.4f}",
        monitor="val/total",
        mode="min",
        save_top_k=1,
        save_last=True,
    )

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    loggers = [
        TensorBoardLogger(str(log_dir), name="footballai"),
        CSVLogger(str(log_dir), name="footballai_csv"),
    ]

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="auto",
        devices="auto",
        gradient_clip_val=args.gradient_clip_val,
        accumulate_grad_batches=args.accumulate_grad_batches,
        callbacks=[checkpoint_callback],
        logger=loggers,
        log_every_n_steps=10,
    )

    trainer.fit(
        model,
        datamodule=datamodule,
        ckpt_path=args.resume,
    )


if __name__ == "__main__":
    main()

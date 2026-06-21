"""PyTorch Lightning training entrypoint for football state pretraining."""

import argparse
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
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
    parser.add_argument("--shot_pos_weight", type=float, default=1.0)
    parser.add_argument("--turnover_pos_weight", type=float, default=1.0)
    parser.add_argument("--gradient_clip_val", type=float, default=1.0)
    parser.add_argument("--accumulate_grad_batches", type=int, default=1)
    parser.add_argument("--use_preprocessed", action="store_true", default=True)
    parser.add_argument("--no_preprocessed", dest="use_preprocessed", action="store_false")
    parser.add_argument("--processed_dir", type=str, default="./data/processed")
    parser.add_argument("--cache_matches", type=int, default=16)
    parser.add_argument("--patience", type=int, default=3, help="Early stopping patience epochs")
    parser.add_argument("--max_epochs", type=int, default=None, help="Alias for --epochs")
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

    epochs = args.max_epochs if args.max_epochs is not None else args.epochs

    config = build_config(args)
    datamodule = StatsBombDataModule(
        config=config,
        val_ratio=args.val_ratio,
        num_workers=args.num_workers,
        use_preprocessed=args.use_preprocessed,
        processed_dir=args.processed_dir,
        cache_matches=args.cache_matches,
    )

    model = FootballPretrainModule(
        config=config,
        receiver_weight=args.receiver_weight,
        shot_weight=args.shot_weight,
        turnover_weight=args.turnover_weight,
        shot_pos_weight=args.shot_pos_weight,
        turnover_pos_weight=args.turnover_pos_weight,
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
    early_stop_callback = EarlyStopping(
        monitor="val/total",
        min_delta=0.001,
        patience=args.patience,
        verbose=True,
        mode="min",
    )

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    loggers = [
        TensorBoardLogger(str(log_dir), name="footballai"),
        CSVLogger(str(log_dir), name="footballai_csv"),
    ]

    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator="auto",
        devices="auto",
        gradient_clip_val=args.gradient_clip_val,
        accumulate_grad_batches=args.accumulate_grad_batches,
        callbacks=[checkpoint_callback, early_stop_callback],
        logger=loggers,
        log_every_n_steps=50,
        enable_progress_bar=True,
        enable_model_summary=True,
    )

    trainer.fit(
        model,
        datamodule=datamodule,
        ckpt_path=args.resume,
    )


if __name__ == "__main__":
    main()

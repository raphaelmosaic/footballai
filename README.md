# FootballAI State Pre-training

Self-supervised / multi-task pretraining for a football state representation model using [StatsBomb Open Data](https://github.com/statsbomb/open-data). The model learns from raw match event sequences to predict pass receivers, shot expected goals (xG), and turnover likelihood at each event. The pretrained spatial + temporal backbone can later be used for real-time inference from video feeds.

## Project layout

```
footballai/
├── config.py                 # Hyperparameter dataclass
├── train.py                  # PyTorch Lightning training entrypoint
├── data/
│   ├── __init__.py
│   ├── datamodule.py       # Lightning DataModule
│   ├── state_builder.py    # StatsBomb event -> state tensor parser
│   └── statsbomb_dataset.py # Per-event dataset + SequenceDataset wrapper
├── models/
│   ├── __init__.py         # End-to-end FootballStateModel wrapper
│   ├── lightning_module.py # LightningModule with multi-task loss
│   ├── spatial_encoder.py  # Transformer encoder over players + ball
│   ├── temporal_model.py   # GRU/LSTM temporal backbone
│   └── pretrain_heads.py   # Multi-task pretraining heads
└── utils/
    ├── __init__.py
    └── metrics.py          # Masked MSE/BCE/CE helpers
```

## Model architecture

```text
Per-event state (ball + up to 22 players, 10 features each)
           ↓
    EntityEmbedding + Distance-aware Transformer
           ↓
    Fixed-size state vector z_t  [B, T, D]
           ↓
    TemporalModel (bidirectional GRU)
           ↓
    Hidden state h_t  [B, T, H]
           ↓
    ┌──────────────┬──────────────┬──────────────┐
    ↓              ↓              ↓
 pass head     shot/xG head   turnover head
 end_xy +       shot_prob +    binary prob
 receiver_slot  xG value
```

Per-entity input features:
- `x, y` — pitch coordinates normalized to `[-1, 1]`
- `vx, vy` — inferred velocities
- `team` — one-hot home/away
- `position_id` — learned role embedding
- `possession` — ball-controller flag
- `ball` — 1 for the ball entity
- `on_pitch` — padding / valid flag

All halves are oriented so the first-half kickoff team always attacks the `+x` direction.

## Setup

The project uses Python 3.12 and PyTorch with CUDA 13.

```bash
source .venv/bin/activate
```

If the environment does not exist yet:

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install torch torchvision torchaudio numpy pandas scipy tqdm pytorch-lightning torchmetrics tensorboard
```

## Training

Full run on all available matches:

```bash
python train.py \
  --data_dir /home/jack/workspace/open-data/data \
  --epochs 50 \
  --batch_size 32 \
  --seq_len 50 \
  --seq_stride 25 \
  --horizon 5.0 \
  --val_ratio 0.15 \
  --checkpoint_dir ./checkpoints \
  --log_dir ./runs
```

Quick smoke test on a couple of matches:

```bash
python train.py \
  --max_matches 2 \
  --epochs 1 \
  --batch_size 2 \
  --seq_len 10 \
  --seq_stride 5 \
  --num_workers 0 \
  --checkpoint_dir ./checkpoints_smoke \
  --log_dir ./runs_smoke
```

Resume from a checkpoint:

```bash
python train.py \
  --resume ./checkpoints/last.ckpt \
  --epochs 100
```

Tune task weights:

```bash
python train.py \
  --receiver_weight 1.0 \
  --shot_weight 2.0 \
  --turnover_weight 1.0
```

View TensorBoard:

```bash
tensorboard --logdir=./runs
```

## Outputs

- `checkpoints/{epoch:03d}-{val/total:.4f}.ckpt` — best checkpoint by validation total loss.
- `checkpoints/last.ckpt` — last epoch checkpoint for resuming.
- `runs/` — TensorBoard + CSV logs.

## Key design choices

- **Match-level split**: train/val separation uses `match_id` so that sequences from the same match never leak across splits.
- **Set-based spatial encoder**: Transformer over a variable set of players + ball is permutation-invariant and naturally handles missing players (useful when moving to video detections).
- **Temporal backbone**: bidirectional GRU over state-vector sequences; forward-only mode can be enabled for streaming inference.
- **Masked multi-task losses**: padded positions are ignored in every metric, and positive-only masks are used for pass end-points / receiver slots and xG values.
- **PyTorch Lightning**: gives mixed precision, distributed training, checkpointing, and TensorBoard/CSV logging for free.
- **GPU auto-detect**: uses CUDA when available, falls back to CPU.

## Real-time inference roadmap

1. Pretrain this model on StatsBomb event data to learn a generic state representation.
2. Replace the event-based `SequenceDataset` with a live/video state builder that produces the same `[N, 10]` entity feature tensor from detections.
3. Use the trained `FootballStateModel` backbone in forward-only mode to emit `state` / `temporal_hidden` vectors and task probabilities each frame.
4. Add a lightweight downstream head calibrated to the specific prediction market signal you want to trade.

## Data source

StatsBomb Open Data (`/home/jack/workspace/open-data`). See the StatsBomb [Terms & Conditions](https://github.com/statsbomb/open-data) if publishing any research derived from this data.

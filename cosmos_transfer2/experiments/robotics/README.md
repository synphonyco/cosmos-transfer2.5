# Robotics Multiview Post-Training for Cosmos Transfer 2.5

Post-training configuration for bimanual robotics videos with edge control signals.

## Dataset Structure

Data stored in S3, mounted locally for training:

```
s3://YOUR_BUCKET/dyna_posttrain/    (mount at /mnt/s3_data/dyna_posttrain/)
├── videos/
│   ├── cam_high/*.mp4
│   ├── cam_left_wrist/*.mp4
│   ├── cam_right_wrist/*.mp4
│   ├── cam_waist_left/*.mp4
│   └── cam_waist_right/*.mp4
├── control_input_edge/
│   ├── cam_high/*.mp4
│   ├── cam_left_wrist/*.mp4
│   ├── cam_right_wrist/*.mp4
│   ├── cam_waist_left/*.mp4
│   └── cam_waist_right/*.mp4
└── captions/
    └── cam_high/*.json   # Format: {"caption": "..."}
```

## Video Specifications

- Resolution: 832x480 (width x height)
- Frame rate: 10 fps
- Codec: H.264
- Training samples: 29 frames per sample (~2.9 seconds)

## Prerequisites

### 1. Mount S3 Bucket

```bash
# Install s3fs if needed
sudo apt-get install s3fs

# Create mount point
sudo mkdir -p /mnt/s3_data

# Mount (adjust bucket name)
s3fs YOUR_BUCKET /mnt/s3_data -o iam_role=auto -o allow_other

# Verify
ls /mnt/s3_data/dyna_posttrain/videos/
```

### 2. Update Dataset Path

Edit `robotics_dataloader.py` and update the mount path:
```python
dataset = L(RoboticsTransferDataset)(
    dataset_dir="/mnt/s3_data/dyna_posttrain",  # <-- Your mount path
    ...
)
```

### 3. HuggingFace Login (for checkpoint download)
```bash
huggingface-cli login
```

## Training Commands

### 8 GPUs on 1 Node (Recommended)
```bash
cd /root/projects/cosmos-transfer

torchrun --nproc_per_node=8 --master_port=12341 -m scripts.train \
    --config=cosmos_transfer2/_src/transfer2_multiview/configs/vid2vid_transfer/config.py \
    -- experiment=robotics_multiview_edge_posttrain \
    job.wandb_mode=disabled
```

### 8 Nodes with 1 GPU Each
```bash
# On each node, set MASTER_ADDR and MASTER_PORT, then:
torchrun --nproc_per_node=1 --nnodes=8 --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR --master_port=12341 \
    -m scripts.train \
    --config=cosmos_transfer2/_src/transfer2_multiview/configs/vid2vid_transfer/config.py \
    -- experiment=robotics_multiview_edge_posttrain \
    job.wandb_mode=disabled
```

### Dry Run (Test Config)
```bash
torchrun --nproc_per_node=1 --master_port=12341 -m scripts.train \
    --config=cosmos_transfer2/_src/transfer2_multiview/configs/vid2vid_transfer/config.py \
    --dryrun -- experiment=robotics_multiview_edge_posttrain
```

## Configuration

### Key Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Resolution | 480p (832x480) | Model adapts from 720p checkpoint |
| FPS | 10 | Matches checkpoint |
| Frames per sample | 29 | ~2.9 seconds |
| Cameras | 5 | cam_high, wrists, waist |
| Control type | Edge (Canny) | Preserves robot poses |
| Caption source | cam_high only | Single caption per episode |
| Batch size | 1 | Per GPU |
| Max iterations | 5,000 | Adjust based on convergence |
| Save interval | 500 | Checkpoint frequency |

### Changing Training Iterations

Edit `robotics_posttrain.py`:
```python
trainer=dict(
    max_iter=5_000,  # <-- Change this
    ...
)
```

## Output

Checkpoints saved to: `outputs/<job_name>/checkpoints/`

## Inference After Training

```bash
python -m cosmos_transfer2.scripts.inference_multiview \
    --checkpoint_path outputs/<job_name>/checkpoints/iter_XXXXX/model_ema_bf16.pt \
    --config your_inference_config.json
```

## Troubleshooting

### CUDA Not Found
```bash
cd /root/projects/cosmos-transfer
uv sync --extra=cu121  # or cu124
```

### Out of Memory
- Reduce `num_workers` in dataloader
- Use gradient checkpointing if available
- Try 4 GPUs instead of 8

### S3 Mount Issues
```bash
# Check mount
mount | grep s3fs

# Remount with debug
s3fs YOUR_BUCKET /mnt/s3_data -o dbglevel=info -f
```

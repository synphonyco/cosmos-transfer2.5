# Robotics Multiview Post-Training for Cosmos Transfer 2.5

Post-training configuration for bimanual robotics videos with edge control signals.

## Overview

This experiment fine-tunes the Cosmos Transfer 2.5 multiview model on robotics data with 5 synchronized camera views and pre-computed Canny edge control signals.

## Dataset Structure

Data stored in S3, mounted locally for training.

**S3 Sync Status:** ✅ Synced (2026-01-21) — 3,913 files uploaded to `s3://synphony-mv-rnd/dyna_posttrain/`

```
s3://synphony-mv-rnd/dyna_posttrain/    (mount at /mnt/s3_data/dyna_posttrain/)
├── videos/
│   ├── cam_high/
│   │   ├── episode_0001.mp4
│   │   ├── episode_0002.mp4
│   │   └── ...
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
    └── cam_high/
        ├── episode_0001.json   # Format: {"caption": "A robot arm picking up..."}
        └── ...
```

**Important:**
- Video filenames must match across all camera folders (e.g., `episode_0001.mp4` in all 5 camera dirs)
- Control edge videos must have the same filenames as their corresponding RGB videos
- Captions are only required for `cam_high` (used as the single caption for all views)

## Video Specifications

| Spec | Value |
|------|-------|
| Resolution | 832×480 (width × height) |
| Frame rate | 10 fps |
| Codec | H.264 |
| Training samples | 29 frames per sample (~2.9 seconds) |

## Training Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Resolution | 480p (832×480) | Model adapts from 720p checkpoint |
| Frames per sample | 29 | ~2.9 seconds at 10fps |
| Cameras | 5 | cam_high, cam_left_wrist, cam_right_wrist, cam_waist_left, cam_waist_right |
| Control type | Edge (Canny) | Pre-computed, preserves robot poses |
| Caption source | cam_high only | Single caption per episode |
| Batch size | 1 per GPU | 8 total with 8 GPUs |
| Max iterations | 5,000 | Configurable |
| Checkpoint interval | 500 iterations | Configurable |
| Context parallel | 1 | Data parallel mode for 29-frame videos |
| Train/Val split | 90/10 | Automatic, based on sorted video list |

---

## Step-by-Step Training Guide

### Step 1: Mount S3 Bucket

```bash
# Install s3fs if needed
sudo apt-get install s3fs

# Create mount point
sudo mkdir -p /mnt/s3_data

# Mount the bucket
s3fs synphony-mv-rnd /mnt/s3_data -o iam_role=auto -o allow_other

# Verify dataset structure
ls /mnt/s3_data/dyna_posttrain/videos/cam_high/ | head -5
ls /mnt/s3_data/dyna_posttrain/control_input_edge/cam_high/ | head -5
ls /mnt/s3_data/dyna_posttrain/captions/cam_high/ | head -5
```

### Step 2: HuggingFace Login

Required for downloading the base checkpoint:

```bash
huggingface-cli login
# Enter your access token when prompted
```

### Step 3: Navigate to Project

```bash
cd /root/projects/cosmos-transfer
```

### Step 4: Dry Run (Validate Config)

Test that everything is configured correctly before starting training:

```bash
torchrun --nproc_per_node=1 --master_port=12341 -m scripts.train \
    --config=cosmos_transfer2/_src/transfer2_multiview/configs/vid2vid_transfer/config.py \
    --dryrun -- experiment=robotics_multiview_edge_posttrain
```

This validates:
- Config parsing works
- Dataset paths exist and are readable
- Checkpoint can be located/downloaded
- No import errors

### Step 5: Run Training

#### Option A: 8 GPUs on 1 Node (Recommended)

```bash
torchrun --nproc_per_node=8 --master_port=12341 -m scripts.train \
    --config=cosmos_transfer2/_src/transfer2_multiview/configs/vid2vid_transfer/config.py \
    -- experiment=robotics_multiview_edge_posttrain job.wandb_mode=disabled
```

#### Option B: 8 Nodes with 1 GPU Each

```bash
# On each node, set environment variables first:
export MASTER_ADDR=<master_node_ip>
export MASTER_PORT=12341
export NODE_RANK=<0-7>

torchrun --nproc_per_node=1 --nnodes=8 --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
    -m scripts.train \
    --config=cosmos_transfer2/_src/transfer2_multiview/configs/vid2vid_transfer/config.py \
    -- experiment=robotics_multiview_edge_posttrain job.wandb_mode=disabled
```

### Step 6: Monitor Training

Checkpoints and logs are saved to `outputs/robotics_edge_posttrain_480p_10fps/`:

```bash
# Watch for new checkpoints
watch -n 60 'ls -la outputs/robotics_edge_posttrain_480p_10fps/checkpoints/'

# Tail training logs
tail -f outputs/robotics_edge_posttrain_480p_10fps/logs/*.log
```

---

## Configuration Overrides

Override parameters at runtime without editing files:

```bash
# Change max iterations
torchrun ... -- experiment=robotics_multiview_edge_posttrain trainer.max_iter=10000

# Enable Weights & Biases logging
torchrun ... -- experiment=robotics_multiview_edge_posttrain job.wandb_mode=online

# Change checkpoint frequency
torchrun ... -- experiment=robotics_multiview_edge_posttrain checkpoint.save_iter=1000

# Change batch size (careful with memory)
torchrun ... -- experiment=robotics_multiview_edge_posttrain dataloader_train.batch_size=2
```

---

## Output Structure

```
outputs/robotics_edge_posttrain_480p_10fps/
├── checkpoints/
│   ├── iter_000500/
│   │   ├── model_ema_bf16.pt      # Use this for inference
│   │   ├── model_reg_bf16.pt
│   │   └── training_state.pt
│   ├── iter_001000/
│   └── ...
├── logs/
│   └── train.log
└── samples/
    └── ...
```

---

## Inference After Training

```bash
python -m cosmos_transfer2.scripts.inference_multiview \
    --checkpoint_path outputs/robotics_edge_posttrain_480p_10fps/checkpoints/iter_05000/model_ema_bf16.pt \
    --config your_inference_config.json
```

---

## Troubleshooting

### CUDA Not Found

```bash
cd /root/projects/cosmos-transfer
uv sync --extra=cu121  # or cu124 depending on your CUDA version
```

### Out of Memory

- Reduce `num_workers` in `robotics_dataloader.py` (default: 4)
- Ensure `context_parallel_size=1` in `robotics_posttrain.py`
- Try fewer GPUs if single GPU memory is insufficient

### S3 Mount Issues

```bash
# Check if mounted
mount | grep s3fs

# Unmount and remount with debug output
sudo umount /mnt/s3_data
s3fs synphony-mv-rnd /mnt/s3_data -o dbglevel=info -f -o allow_other
```

### Dataset Not Found Errors

Verify all required files exist:

```bash
# Count videos per camera
for cam in cam_high cam_left_wrist cam_right_wrist cam_waist_left cam_waist_right; do
    echo "$cam: $(ls /mnt/s3_data/dyna_posttrain/videos/$cam/*.mp4 2>/dev/null | wc -l) videos"
done

# Count edge videos per camera
for cam in cam_high cam_left_wrist cam_right_wrist cam_waist_left cam_waist_right; do
    echo "$cam: $(ls /mnt/s3_data/dyna_posttrain/control_input_edge/$cam/*.mp4 2>/dev/null | wc -l) edge videos"
done

# Count captions
echo "captions: $(ls /mnt/s3_data/dyna_posttrain/captions/cam_high/*.json 2>/dev/null | wc -l)"
```

### Checkpoint Download Fails

Ensure HuggingFace token has access to the model:

```bash
huggingface-cli whoami
huggingface-cli login --token YOUR_TOKEN
```

---

## File Reference

| File | Purpose |
|------|---------|
| `robotics_dataloader.py` | Dataset and dataloader configuration |
| `robotics_posttrain.py` | Experiment config (iterations, checkpointing, etc.) |
| `README.md` | This guide |

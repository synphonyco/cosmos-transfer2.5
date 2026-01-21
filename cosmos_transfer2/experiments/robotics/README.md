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
| Training samples | 93 frames per sample (~9.3 seconds) |

## Training Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Resolution | 480p (832×480) | Model adapts from 720p checkpoint |
| Frames per sample | 93 | ~9.3 seconds at 10fps (state_t=24) |
| Cameras | 5 | cam_high, cam_left_wrist, cam_right_wrist, cam_waist_left, cam_waist_right |
| Control type | Edge (Canny) | Pre-computed, preserves robot poses |
| Caption source | cam_high only | Single caption per episode |
| Batch size | 1 per GPU | 8 total with 8 GPUs |
| Max iterations | 5,000 | Configurable |
| Checkpoint interval | 200 iterations | Frequent saves for evaluation |
| Sample generation | Every 200 iters | Validation samples for visual eval |
| Context parallel | 2 | Matches NVIDIA's 480p setup (12 latent frames/GPU) |
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

### Prepare Inference Input

Create an input directory with edge control videos:

```
inference_input/
├── control/                   # Edge control videos (required)
│   ├── cam_high/
│   │   ├── test_episode_001.mp4
│   │   └── ...
│   ├── cam_left_wrist/
│   ├── cam_right_wrist/
│   ├── cam_waist_left/
│   └── cam_waist_right/
├── videos/                    # Original RGB videos (optional)
│   └── ...
└── captions/                  # Text captions (optional)
    └── cam_high/
        ├── test_episode_001.txt
        └── ...
```

### Single-Shot Inference (93 frames / 9.3s)

```bash
cd /root/projects/cosmos-transfer

PYTHONPATH=. torchrun --nproc_per_node=8 --master_port=12341 \
    -m cosmos_transfer2.experiments.robotics.robotics_inference \
    --ckpt_path outputs/robotics_edge_posttrain_480p_10fps/checkpoints/iter_001000/ \
    --input_root /path/to/inference_input \
    --save_root results/robotics_inference \
    --guidance 3.0 \
    --num_steps 35 \
    --save_each_view
```

### Autoregressive Inference (Long Videos)

For videos longer than 9.3 seconds:

```bash
PYTHONPATH=. torchrun --nproc_per_node=8 --master_port=12341 \
    -m cosmos_transfer2.experiments.robotics.robotics_inference \
    --ckpt_path outputs/robotics_edge_posttrain_480p_10fps/checkpoints/iter_001000/ \
    --input_root /path/to/inference_input \
    --save_root results/robotics_long_videos \
    --use_autoregressive \
    --chunk_overlap 1 \
    --guidance 3.0 \
    --save_each_view
```

### Inference Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--guidance` | 3.0 | Classifier-free guidance scale (higher = more prompt adherence) |
| `--num_steps` | 35 | Diffusion sampling steps (more = better quality, slower) |
| `--control_weight` | 1.0 | Edge control influence (1.0 = full control) |
| `--seed` | 0 | Random seed for reproducibility |
| `--use_autoregressive` | False | Enable for videos > 93 frames |
| `--chunk_overlap` | 1 | Overlap frames between chunks (autoregressive) |

---

## Checkpoint Evaluation

### Understanding Checkpoints

Training saves checkpoints every 200 iterations:

```
outputs/robotics_edge_posttrain_480p_10fps/checkpoints/
├── iter_000200/    # Early training (underfitted)
├── iter_000400/
├── iter_000600/
├── iter_000800/
├── iter_001000/    # ~3 epochs over data
├── iter_002000/    # ~6 epochs
├── iter_003000/    # ~9 epochs - often best quality
├── iter_004000/    # May start overfitting
└── iter_005000/    # Final checkpoint
```

Each checkpoint contains:
- `model_ema_bf16.pt` - **Use this for inference** (exponential moving average weights)
- `model_reg_bf16.pt` - Regular model weights (for comparison)
- `training_state.pt` - Optimizer state (for resuming training)

### Evaluate Multiple Checkpoints

Create a test set of edge videos, then run inference on each checkpoint:

```bash
# Evaluate checkpoints 1000, 2000, 3000
for iter in 1000 2000 3000 4000 5000; do
    PYTHONPATH=. torchrun --nproc_per_node=8 --master_port=12341 \
        -m cosmos_transfer2.experiments.robotics.robotics_inference \
        --ckpt_path outputs/robotics_edge_posttrain_480p_10fps/checkpoints/iter_00${iter}/ \
        --input_root /path/to/test_inputs \
        --save_root results/eval_iter_${iter} \
        --guidance 3.0 \
        --max_samples 5
done
```

### What to Look For

| Checkpoint Stage | Typical Behavior |
|------------------|------------------|
| iter_200-600 | Blurry, noisy, poor structure |
| iter_800-1500 | Improving quality, some artifacts |
| iter_1500-3000 | **Best quality zone** - good structure, realistic textures |
| iter_3000-5000 | May overfit - copies training data too closely |

**Signs of underfitting:**
- Blurry or noisy outputs
- Poor edge following
- Generic/repeated textures

**Signs of overfitting:**
- Outputs look exactly like training videos
- Novel edge inputs produce garbage
- Loss on validation set increases while training loss decreases

### Validation Samples During Training

The model automatically generates validation samples every 200 iterations:

```
outputs/robotics_edge_posttrain_480p_10fps/samples/
├── iter_000200/
│   ├── sample_reg_*.mp4    # Regular model samples
│   └── sample_ema_*.mp4    # EMA model samples (usually better)
├── iter_000400/
└── ...
```

Watch these samples during training to pick the best checkpoint without running full inference.

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
| `robotics_inference.py` | Inference script for generating videos |
| `robotics_view_meta.json` | Camera descriptions for caption prefixes |
| `README.md` | This guide |

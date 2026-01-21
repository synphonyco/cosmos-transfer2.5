# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Inference script for robotics multiview post-trained model.

Generates synthetic videos from edge control signals using the fine-tuned
Cosmos Transfer 2.5 multiview model.

Usage:
    # Single-shot inference (93 frames / 9.3s)
    PYTHONPATH=. torchrun --nproc_per_node=8 --master_port=12341 \
        -m cosmos_transfer2.experiments.robotics.robotics_inference \
        --ckpt_path s3://synphony-mv-rnd/checkpoints/robotics_posttrain/iter_005000/ \
        --input_root /path/to/inference_input \
        --save_root results/robotics_inference \
        --guidance 3.0 \
        --num_steps 35

    # Autoregressive inference (long videos)
    PYTHONPATH=. torchrun --nproc_per_node=8 --master_port=12341 \
        -m cosmos_transfer2.experiments.robotics.robotics_inference \
        --ckpt_path s3://synphony-mv-rnd/checkpoints/robotics_posttrain/iter_005000/ \
        --input_root /path/to/inference_input \
        --save_root results/robotics_inference_long \
        --use_autoregressive \
        --guidance 3.0

Expected input directory structure:
    input_root/
    ├── videos/                    # Input RGB videos (optional for edge-only)
    │   ├── cam_high/
    │   │   ├── episode_001.mp4
    │   │   └── ...
    │   ├── cam_left_wrist/
    │   ├── cam_right_wrist/
    │   ├── cam_waist_left/
    │   └── cam_waist_right/
    ├── control/                   # Edge control videos (required)
    │   ├── cam_high/
    │   │   ├── episode_001.mp4
    │   │   └── ...
    │   └── ...
    └── captions/                  # Text captions (optional)
        └── cam_high/
            ├── episode_001.txt    # Plain text caption
            └── ...
"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import Final

import torch as th
import torchvision

from cosmos_transfer2._src.imaginaire.utils import distributed, log
from cosmos_transfer2._src.imaginaire.utils.easy_io import easy_io
from cosmos_transfer2._src.imaginaire.visualize.video import save_img_or_video
from cosmos_transfer2._src.predict2_multiview.scripts.mv_visualize_helper import (
    arrange_video_visualization,
    save_each_view_separately,
)
from cosmos_transfer2._src.transfer2_multiview.inference.inference import ControlVideo2WorldInference

# Import robotics dataloader to register the experiment
from cosmos_transfer2.experiments.robotics.robotics_dataloader import register_dataloader_robotics

# Register the robotics dataloader
register_dataloader_robotics()

# Robotics camera configuration (must match training)
ROBOTICS_CAMERAS: Final[tuple[str, ...]] = (
    "cam_high",
    "cam_left_wrist",
    "cam_right_wrist",
    "cam_waist_left",
    "cam_waist_right",
)

ROBOTICS_CAMERA_TO_VIEW_ID: Final[dict[str, int]] = {
    "cam_high": 0,
    "cam_left_wrist": 1,
    "cam_right_wrist": 2,
    "cam_waist_left": 3,
    "cam_waist_right": 4,
}

# Camera-specific caption prefixes for robotics setup
CAMERA_CAPTION_PREFIXES: Final[dict[str, str]] = {
    "cam_high": "The video is captured from a camera mounted above the robot workspace, providing a top-down view.",
    "cam_left_wrist": "The video is captured from a camera mounted on the left robot arm wrist, showing the gripper's perspective.",
    "cam_right_wrist": "The video is captured from a camera mounted on the right robot arm wrist, showing the gripper's perspective.",
    "cam_waist_left": "The video is captured from a camera on the left side of the robot base, showing the left workspace.",
    "cam_waist_right": "The video is captured from a camera on the right side of the robot base, showing the right workspace.",
}

# Default caption for robotics videos
DEFAULT_ROBOTICS_CAPTION = (
    "A bimanual robot with two arms performs a manipulation task on a table. "
    "The robot's movements are precise and coordinated."
)

NUM_CONDITIONAL_FRAMES_KEY = "num_conditional_frames"


def calculate_autoregressive_frames(chunk_size: int, chunk_overlap: int, num_chunks: int) -> int:
    """Calculate total frames needed for autoregressive generation."""
    return chunk_size + (chunk_size - chunk_overlap) * (num_chunks - 1)


def load_video(
    video_path: str,
    target_frames: int = 93,
    target_size: tuple[int, int] = (480, 832),  # (H, W) for robotics
    allow_variable_length: bool = False,
) -> th.Tensor:
    """Load video and process to target size and frame count."""
    try:
        video_frames, _ = easy_io.load(video_path)
    except Exception as e:
        raise ValueError(f"Failed to load video {video_path}: {e}")

    # Convert: (T, H, W, C) -> (C, T, H, W)
    video_tensor = th.from_numpy(video_frames).float() / 255.0
    video_tensor = video_tensor.permute(3, 0, 1, 2)

    C, T, H, W = video_tensor.shape

    if not allow_variable_length:
        if T > target_frames:
            video_tensor = video_tensor[:, :target_frames, :, :]
        elif T < target_frames:
            last_frame = video_tensor[:, -1:, :, :]
            padding_frames = target_frames - T
            video_tensor = th.cat([video_tensor, last_frame.repeat(1, padding_frames, 1, 1)], dim=1)

    # Convert to uint8: (C, T, H, W) -> (T, C, H, W)
    video_tensor = video_tensor.permute(1, 0, 2, 3)
    video_tensor = (video_tensor * 255.0).to(th.uint8)

    # Resize if needed
    target_h, target_w = target_size
    if H != target_h or W != target_w:
        video_tensor = resize_and_crop(video_tensor, target_size)

    # Back to (C, T, H, W)
    video_tensor = video_tensor.permute(1, 0, 2, 3)
    return video_tensor


def resize_and_crop(video: th.Tensor, target_size: tuple[int, int]) -> th.Tensor:
    """Resize video and center crop to target size."""
    orig_h, orig_w = video.shape[2], video.shape[3]
    target_h, target_w = target_size

    scaling_ratio = max(target_w / orig_w, target_h / orig_h)
    resizing_shape = (int(scaling_ratio * orig_h), int(scaling_ratio * orig_w))

    video_resized = torchvision.transforms.functional.resize(video, resizing_shape)
    video_cropped = torchvision.transforms.functional.center_crop(video_resized, target_size)
    return video_cropped


def load_multiview_videos(
    input_root: Path,
    video_id: str,
    camera_order: list[str],
    target_frames: int = 93,
    target_size: tuple[int, int] = (480, 832),
    folder_name: str = "videos",
    allow_variable_length: bool = False,
) -> th.Tensor:
    """Load multi-view videos from specified folder."""
    videos_dir = input_root / folder_name
    video_tensors = []

    for camera in camera_order:
        video_path = videos_dir / camera / f"{video_id}.mp4"
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        video_tensor = load_video(
            str(video_path), target_frames, target_size, allow_variable_length
        )
        video_tensors.append(video_tensor)

    # Concatenate all views: (C, V*T, H, W)
    return th.cat(video_tensors, dim=1)


def load_captions(
    input_root: Path,
    video_id: str,
    camera_order: list[str],
    add_camera_prefix: bool = True,
) -> list[str]:
    """Load captions for all views. Falls back to default if not found."""
    captions_dir = input_root / "captions"
    captions = []

    for camera in camera_order:
        caption = DEFAULT_ROBOTICS_CAPTION

        # Try to load caption file
        caption_path = captions_dir / camera / f"{video_id}.txt"
        json_path = captions_dir / camera / f"{video_id}.json"

        if caption_path.exists():
            with open(caption_path, "r") as f:
                caption = f.read().strip()
        elif json_path.exists():
            with open(json_path, "r") as f:
                data = json.load(f)
                caption = data.get("caption", DEFAULT_ROBOTICS_CAPTION)
        elif camera == "cam_high":
            # Try cam_high caption for all views (training used cam_high only)
            cam_high_txt = captions_dir / "cam_high" / f"{video_id}.txt"
            cam_high_json = captions_dir / "cam_high" / f"{video_id}.json"
            if cam_high_txt.exists():
                with open(cam_high_txt, "r") as f:
                    caption = f.read().strip()
            elif cam_high_json.exists():
                with open(cam_high_json, "r") as f:
                    data = json.load(f)
                    caption = data.get("caption", DEFAULT_ROBOTICS_CAPTION)

        # Add camera-specific prefix
        if add_camera_prefix:
            prefix = CAMERA_CAPTION_PREFIXES.get(camera, "")
            caption = f"{prefix} {caption}"

        captions.append(caption)

    return captions


def construct_data_batch(
    multiview_video: th.Tensor,
    control_video: th.Tensor,
    captions: list[str],
    camera_order: list[str],
    num_conditional_frames: int = 1,
    fps: float = 10.0,
    target_frames_per_view: int = 93,
) -> dict:
    """Construct data_batch for model inference."""
    C, VT, H, W = multiview_video.shape
    n_views = len(camera_order)
    T = VT // n_views

    # Add batch dimension
    multiview_video = multiview_video.unsqueeze(0)
    control_video = control_video.unsqueeze(0)

    # Construct view indices
    view_indices_list = []
    for idx, camera in enumerate(camera_order):
        view_indices_list.extend([idx] * T)
    view_indices = th.tensor(view_indices_list, dtype=th.int64).unsqueeze(0)

    view_indices_selection = th.tensor(
        list(range(len(camera_order))), dtype=th.int64
    ).unsqueeze(0)

    # Reference camera is cam_high (index 0)
    ref_cam_position = 0

    data_batch = {
        "video": multiview_video,
        "control_input_hdmap_bbox": control_video,  # Using this key for compatibility
        "ai_caption": [captions],
        "view_indices": view_indices,
        "fps": th.tensor([fps], dtype=th.float64),
        "chunk_index": th.tensor([0], dtype=th.int64),
        "frame_indices": th.arange(target_frames_per_view).unsqueeze(0),
        "num_video_frames_per_view": th.tensor([target_frames_per_view], dtype=th.int64),
        "view_indices_selection": view_indices_selection,
        "camera_keys_selection": [list(camera_order)],
        "sample_n_views": th.tensor([n_views], dtype=th.int64),
        "padding_mask": th.zeros(1, 1, H, W, dtype=th.float32),
        "ref_cam_view_idx_sample_position": th.tensor([ref_cam_position], dtype=th.int64),
        "front_cam_view_idx_sample_position": [None],
        "original_hw": th.tensor([[[H, W]] * n_views], dtype=th.int64),
        NUM_CONDITIONAL_FRAMES_KEY: num_conditional_frames,
    }

    return data_batch


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Robotics multiview inference with Cosmos Transfer 2.5"
    )

    # Model and checkpoint
    parser.add_argument(
        "--experiment",
        type=str,
        default="robotics_multiview_edge_posttrain",
        help="Experiment config name",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        required=True,
        help="Path to checkpoint (local or S3)",
    )
    parser.add_argument(
        "--context_parallel_size",
        type=int,
        default=2,
        help="Context parallel size (default: 2 for 480p)",
    )

    # Generation parameters
    parser.add_argument("--guidance", type=float, default=3.0, help="Guidance scale")
    parser.add_argument("--fps", type=int, default=10, help="Output video FPS")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--num_conditional_frames", type=int, default=1, help="Conditional frames")
    parser.add_argument("--num_steps", type=int, default=35, help="Diffusion steps")
    parser.add_argument("--control_weight", type=float, default=1.0, help="Control weight")
    parser.add_argument(
        "--use_negative_prompt",
        action="store_true",
        default=True,
        help="Use negative prompt",
    )

    # Input/output
    parser.add_argument(
        "--input_root",
        type=str,
        required=True,
        help="Input directory with videos/, control/, captions/",
    )
    parser.add_argument(
        "--save_root",
        type=str,
        default="results/robotics_inference/",
        help="Output directory",
    )
    parser.add_argument("--max_samples", type=int, default=5, help="Max samples to process")

    # Video parameters
    parser.add_argument("--target_height", type=int, default=480, help="Target height")
    parser.add_argument("--target_width", type=int, default=832, help="Target width")

    # Autoregressive
    parser.add_argument(
        "--use_autoregressive",
        action="store_true",
        help="Enable autoregressive generation for long videos",
    )
    parser.add_argument(
        "--chunk_overlap",
        type=int,
        default=1,
        help="Overlap frames between chunks",
    )

    # Output options
    parser.add_argument(
        "--stack_mode",
        type=str,
        default="grid_auto",
        choices=["height", "width", "time", "grid", "grid_auto"],
        help="Video stacking mode",
    )
    parser.add_argument(
        "--save_each_view",
        action="store_true",
        help="Save each camera view separately",
    )
    parser.add_argument(
        "--add_camera_prefix",
        action="store_true",
        default=True,
        help="Add camera prefix to captions",
    )

    # Config overrides
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Additional config overrides",
    )

    return parser.parse_args()


def main():
    os.environ["NVTE_FUSED_ATTN"] = "0"
    th.backends.cudnn.benchmark = False
    th.backends.cudnn.deterministic = True
    th.enable_grad(False)

    args = parse_arguments()
    input_root = Path(args.input_root)
    camera_order = list(ROBOTICS_CAMERAS)
    n_views = len(camera_order)

    # Initialize inference handler
    experiment_opts = list(args.opts) if args.opts else []
    vid2world = ControlVideo2WorldInference(
        args.experiment,
        args.ckpt_path,
        context_parallel_size=args.context_parallel_size,
        experiment_opts=experiment_opts,
    )

    rank0 = args.context_parallel_size <= 1 or distributed.get_rank() == 0

    # Get chunk size from model
    chunk_size = vid2world.model.tokenizer.get_pixel_num_frames(
        vid2world.model.config.state_t
    )
    if rank0:
        log.info(f"Model state_t={vid2world.model.config.state_t}, chunk_size={chunk_size}")

    os.makedirs(args.save_root, exist_ok=True)

    # Find video files
    control_dir = input_root / "control" / camera_order[0]
    if not control_dir.exists():
        raise FileNotFoundError(f"Control directory not found: {control_dir}")

    video_files = sorted(control_dir.glob("*.mp4"))
    video_ids = [f.stem for f in video_files[: args.max_samples]]

    if rank0:
        log.info(f"Found {len(video_ids)} videos to process")

    for i, video_id in enumerate(video_ids):
        if rank0:
            log.info(f"Processing {i + 1}/{len(video_ids)}: {video_id}")

        try:
            # Detect frame count
            first_control = control_dir / f"{video_id}.mp4"
            control_frames, _ = easy_io.load(str(first_control))
            detected_frames = control_frames.shape[0]

            if args.use_autoregressive:
                effective_chunk_size = chunk_size - args.chunk_overlap
                num_chunks = max(1, (detected_frames - args.chunk_overlap + effective_chunk_size - 1) // effective_chunk_size)
                target_frames = detected_frames
                if rank0:
                    log.info(f"Autoregressive: {detected_frames} frames, {num_chunks} chunks")
            else:
                if detected_frames < chunk_size:
                    target_frames = chunk_size  # Will pad
                else:
                    target_frames = chunk_size  # Will truncate
                if rank0 and detected_frames != chunk_size:
                    log.warning(f"Video has {detected_frames} frames, using {chunk_size}")

            target_size = (args.target_height, args.target_width)

            # Load control videos (required)
            control_video = load_multiview_videos(
                input_root,
                video_id,
                camera_order,
                target_frames=target_frames,
                target_size=target_size,
                folder_name="control",
                allow_variable_length=args.use_autoregressive,
            )

            # Load input videos (optional - use control as fallback)
            videos_dir = input_root / "videos" / camera_order[0]
            if videos_dir.exists() and (videos_dir / f"{video_id}.mp4").exists():
                input_video = load_multiview_videos(
                    input_root,
                    video_id,
                    camera_order,
                    target_frames=target_frames,
                    target_size=target_size,
                    folder_name="videos",
                    allow_variable_length=args.use_autoregressive,
                )
            else:
                # Use control as placeholder for input
                input_video = control_video.clone()

            # Load captions
            captions = load_captions(
                input_root, video_id, camera_order, args.add_camera_prefix
            )

            if rank0:
                log.info(f"Loaded control: {control_video.shape}, captions: {len(captions)}")

            # Construct batch
            data_batch = construct_data_batch(
                input_video,
                control_video,
                captions,
                camera_order,
                num_conditional_frames=args.num_conditional_frames,
                fps=args.fps,
                target_frames_per_view=target_frames,
            )
            data_batch["control_weight"] = args.control_weight

            # Run inference
            th.cuda.synchronize()
            start_time = time.time()

            if args.use_autoregressive:
                if NUM_CONDITIONAL_FRAMES_KEY in data_batch:
                    del data_batch[NUM_CONDITIONAL_FRAMES_KEY]

                video, control = vid2world.generate_autoregressive_from_batch(
                    data_batch,
                    guidance=args.guidance,
                    seed=args.seed + i,
                    num_conditional_frames=args.num_conditional_frames,
                    num_steps=args.num_steps,
                    n_views=n_views,
                    chunk_size=chunk_size,
                    chunk_overlap=args.chunk_overlap,
                    use_negative_prompt=args.use_negative_prompt,
                )
                video = video.unsqueeze(0)
                control = control.unsqueeze(0)
            else:
                video = vid2world.generate_from_batch(
                    data_batch,
                    guidance=args.guidance,
                    seed=args.seed + i,
                    num_steps=args.num_steps,
                    use_negative_prompt=args.use_negative_prompt,
                )
                control = data_batch["control_input_hdmap_bbox"].float() / 255.0

            th.cuda.synchronize()
            elapsed = time.time() - start_time
            if rank0:
                log.info(f"Generation took {elapsed:.2f}s")

            # Move to CPU and arrange
            video_cpu = video.cpu()
            control_cpu = control.cpu()

            video_arranged = arrange_video_visualization(video_cpu, data_batch, method=args.stack_mode)
            control_arranged = arrange_video_visualization(control_cpu, data_batch, method=args.stack_mode)

            # Save
            if rank0:
                video_path = f"{args.save_root}/{video_id}_generated"
                save_img_or_video(video_arranged[0], video_path, fps=args.fps)
                log.info(f"Saved: {video_path}")

                control_path = f"{args.save_root}/{video_id}_control"
                save_img_or_video(control_arranged[0], control_path, fps=args.fps)

                if args.save_each_view:
                    save_dir = f"{args.save_root}/{video_id}_views"
                    save_each_view_separately(
                        mv_video=video_cpu[0],
                        data_batch=data_batch,
                        save_dir=save_dir,
                        fps=args.fps,
                    )

        except Exception as e:
            log.error(f"Error processing {video_id}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Cleanup
    if args.context_parallel_size > 1:
        th.distributed.barrier()
    vid2world.cleanup()

    if rank0:
        log.info("Inference complete!")


if __name__ == "__main__":
    main()

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Experiment config for robotics multiview post-training with edge control.

Usage:
    torchrun --nproc_per_node=8 --master_port=12341 -m scripts.train \
        --config=cosmos_transfer2/_src/transfer2_multiview/configs/vid2vid_transfer/config.py \
        -- experiment=robotics_multiview_edge_posttrain job.wandb_mode=disabled

For single GPU testing:
    torchrun --nproc_per_node=1 --master_port=12341 -m scripts.train \
        --config=cosmos_transfer2/_src/transfer2_multiview/configs/vid2vid_transfer/config.py \
        --dryrun -- experiment=robotics_multiview_edge_posttrain

Dataset structure expected at /mnt/s3_data/dyna_posttrain/:
    videos/
        cam_high/*.mp4
        cam_left_wrist/*.mp4
        cam_right_wrist/*.mp4
        cam_waist_left/*.mp4
        cam_waist_right/*.mp4
    control_input_edge/
        cam_high/*.mp4
        cam_left_wrist/*.mp4
        cam_right_wrist/*.mp4
        cam_waist_left/*.mp4
        cam_waist_right/*.mp4
    captions/
        cam_high/*.json  (format: {"caption": "..."})
"""

from hydra.core.config_store import ConfigStore

from cosmos_transfer2._src.imaginaire.lazy_config import LazyCall as L
from cosmos_transfer2._src.imaginaire.utils.checkpoint_db import get_checkpoint_path
from cosmos_transfer2._src.predict2_multiview.callbacks.every_n_draw_sample_multiviewvideo import (
    EveryNDrawSampleMultiviewVideo,
)
from cosmos_transfer2.multiview_config import DEFAULT_CHECKPOINT

# Import the dataloader registration
from cosmos_transfer2.experiments.robotics.robotics_dataloader import register_dataloader_robotics


# Register the robotics dataloader
register_dataloader_robotics()

# Get checkpoint path (s3 is guaranteed to exist for AUTO_MULTIVIEW checkpoint)
assert DEFAULT_CHECKPOINT.s3 is not None, "DEFAULT_CHECKPOINT must have s3 config"
_CHECKPOINT_PATH = get_checkpoint_path(DEFAULT_CHECKPOINT.s3.uri)

# Experiment configuration
robotics_multiview_edge_posttrain = dict(
    defaults=[
        # Inherit from the multiview base experiment
        f"/experiment/{DEFAULT_CHECKPOINT.experiment}",
        # Override dataloader with our robotics data
        {"override /data_train": "robotics_multiview_train_data_edge"},
    ],
    job=dict(
        project="cosmos_transfer_robotics",
        group="robotics_multiview",
        name="robotics_edge_posttrain_480p_10fps",
    ),
    checkpoint=dict(
        save_iter=200,  # More frequent checkpoints for smaller dataset (catch optimal before overfitting)
        load_path=_CHECKPOINT_PATH,
        load_training_state=False,
        strict_resume=False,
        load_from_object_store=dict(
            enabled=False,  # Loading from local filesystem
        ),
        save_to_object_store=dict(
            enabled=False,
        ),
    ),
    model=dict(
        config=dict(
            base_load_from=None,
            # Resolution setting - model will adapt from 720p checkpoint to 480p data
            resolution="480",
            # We have 5 camera views, use all of them
            train_sample_views_range=[5, 5],
        ),
    ),
    trainer=dict(
        logging_iter=50,
        max_iter=5_000,
        callbacks=dict(
            heart_beat=dict(save_s3=False),
            iter_speed=dict(hit_thres=100, save_s3=False),
            device_monitor=dict(save_s3=False),
            # Sample generation now works - added view_indices_selection to dataloader
            # every_n_sample_reg and every_n_sample_ema use defaults from base config
            wandb=dict(save_s3=False),
            wandb_10x=dict(save_s3=False),
            dataloader_speed=dict(save_s3=False),
            frame_loss_log=dict(save_s3=False),
            # Disable sample callbacks with very high interval (must use LazyCall to override base config)
            every_n_sample_reg=L(EveryNDrawSampleMultiviewVideo)(
                every_n=999_999,
                is_x0=False,
                is_ema=False,
                num_sampling_step=35,
                guidance=[7],
                fps=10,
                ctrl_hint_keys=["control_input_edge"],
                control_weights=[0.0, 1.0],
            ),
            every_n_sample_ema=L(EveryNDrawSampleMultiviewVideo)(
                every_n=999_999,
                is_x0=False,
                is_ema=True,
                num_sampling_step=35,
                guidance=[7],
                fps=10,
                ctrl_hint_keys=["control_input_edge"],
                control_weights=[0.0, 1.0],
            ),
        ),
    ),
    model_parallel=dict(
        # n_views (5) must be <= cp_size
        # Using 8 GPUs with cp_size=8 (extra slots are padded)
        context_parallel_size=8,
    ),
)


# Register experiment with Hydra ConfigStore
cs = ConfigStore.instance()

for _item in [robotics_multiview_edge_posttrain]:
    experiment_name = [name.lower() for name, value in globals().items() if value is _item][0]
    cs.store(
        group="experiment",
        package="_global_",
        name=experiment_name,
        node=_item,
    )

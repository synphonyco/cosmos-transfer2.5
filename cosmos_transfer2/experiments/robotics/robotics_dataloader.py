# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Local dataset and dataloader registration for robotics multiview post-training.

Dataset structure expected:
    dataset_dir/
    ├── videos/
    │   ├── cam_high/
    │   │   ├── episode_0001.mp4
    │   │   └── ...
    │   ├── cam_left_wrist/
    │   ├── cam_right_wrist/
    │   ├── cam_waist_left/
    │   └── cam_waist_right/
    ├── control_input_edge/
    │   ├── cam_high/
    │   │   ├── episode_0001.mp4
    │   │   └── ...
    │   └── ...
    └── captions/
        └── cam_high/
            ├── episode_0001.json  # {"caption": "..."}
            └── ...
"""

from typing import Final

from hydra.core.config_store import ConfigStore

from cosmos_transfer2._src.imaginaire.lazy_config import LazyCall as L
from cosmos_transfer2._src.predict2.datasets.local_datasets.dataset_video import get_generic_dataloader
from cosmos_transfer2._src.transfer2.datasets.local_datasets.multiview_dataset import (
    CTRL_TYPE_INFO,
    MultiviewTransferDataset,
)

# Override hdmap_bbox to use our pre-computed edge videos folder
# This works because:
# 1. hint_key="control_input_hdmap_bbox" matches augmentor output key
# 2. ctrl_type="hdmap_bbox" uses this CTRL_TYPE_INFO entry
# 3. folder points to our actual data location
CTRL_TYPE_INFO["hdmap_bbox"] = {
    "folder": "control_input_edge",  # Our pre-computed edge videos
    "format": "mp4",
    "data_dict_key": "hdmap_bbox",
}

# Robotics camera configuration
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


def register_dataloader_robotics() -> None:
    """Register robotics multiview dataloaders with Hydra ConfigStore."""

    cs = ConfigStore.instance()

    # TODO: Update with actual S3 bucket mount path
    # Mount S3 bucket first: s3fs YOUR_BUCKET /mnt/s3_data -o iam_role=auto
    dataset = L(MultiviewTransferDataset)(
        dataset_dir="/mnt/s3_data/dyna_posttrain",  # S3 bucket mount point
        hint_key="control_input_hdmap_bbox",  # Must match augmentor output key
        resolution="480",
        state_t=8,
        num_frames=29,
        sequence_interval=1,
        start_frame_interval=1,
        camera_keys=list(ROBOTICS_CAMERAS),
        video_size=(480, 832),  # (H, W) - 832x480 in standard notation
        front_camera_key="cam_high",  # Use cam_high for captions
        camera_to_view_id=ROBOTICS_CAMERA_TO_VIEW_ID,
        front_view_caption_only=True,
        is_train=True,
    )

    cs.store(
        group="data_train",
        package="dataloader_train",
        name="robotics_multiview_train_data_edge",
        node=L(get_generic_dataloader)(
            dataset=dataset,
            batch_size=1,
            num_workers=4,
            pin_memory=True,
            drop_last=True,
        ),
    )

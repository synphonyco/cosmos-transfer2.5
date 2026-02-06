# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

import torch as th

from cosmos_transfer2._src.imaginaire.utils import log
from cosmos_transfer2._src.imaginaire.visualize.video import save_img_or_video

# Visualization layouts for multi-view video arrangement
VISUALIZE_LAYOUTS = {
    "width": [
        [
            "camera_rear_left_70fov",
            "camera_cross_left_120fov",
            "camera_front_wide_120fov",
            "camera_cross_right_120fov",
            "camera_rear_right_70fov",
            "camera_rear_tele_30fov",
            "camera_front_tele_30fov",
        ]
    ],
    "height": [
        ["camera_rear_left_70fov"],
        ["camera_cross_left_120fov"],
        ["camera_front_wide_120fov"],
        ["camera_cross_right_120fov"],
        ["camera_rear_right_70fov"],
        ["camera_rear_tele_30fov"],
        ["camera_front_tele_30fov"],
    ],
    "grid": [
        [None, "camera_front_tele_30fov", None],
        ["camera_cross_left_120fov", "camera_front_wide_120fov", "camera_cross_right_120fov"],
        ["camera_rear_left_70fov", "camera_rear_tele_30fov", "camera_rear_right_70fov"],
    ],
}


def create_dynamic_grid_layout(view_keys):
    """
    Create a balanced grid layout for any number of views.

    Args:
        view_keys: List of camera view keys

    Returns:
        Layout definition (list of lists) for the grid
    """
    n_views = len(view_keys)

    if n_views <= 3:
        # Single row for 1-3 views
        return [view_keys]
    elif n_views <= 4:
        # 2x2 grid for 4 views
        return [view_keys[:2], view_keys[2:4]]
    elif n_views <= 6:
        # 2 rows, balanced
        mid = (n_views + 1) // 2
        row1 = view_keys[:mid]
        row2 = view_keys[mid:]
        # Pad shorter row with None to match longer row
        while len(row2) < len(row1):
            row2.append(None)
        return [row1, row2]
    elif n_views <= 9:
        # 3x3 grid for 7-9 views
        row_size = 3
        rows = []
        for i in range(0, n_views, row_size):
            row = view_keys[i:i+row_size]
            # Pad with None to make rows equal length
            while len(row) < row_size:
                row.append(None)
            rows.append(row)
        return rows
    else:
        # For more views, use 4 columns
        row_size = 4
        rows = []
        for i in range(0, n_views, row_size):
            row = view_keys[i:i+row_size]
            while len(row) < row_size:
                row.append(None)
            rows.append(row)
        return rows


def arrange_video_visualization(mv_video, data_batch, method="width"):
    """
    Rearrange multi-view video based on specified layout method.

    Args:
        mv_video: (B, C, V * T, H, W) - Multi-view video tensor
        data_batch: Batch containing camera order information
        method: Method to arrange video visualization. Can be "width", "height", "grid", "grid_auto", or "time".
                - "width": Arrange all views in a single horizontal row
                - "height": Arrange all views in a single vertical column
                - "grid": Arrange views in predefined 3x3 grid (requires specific camera keys)
                - "grid_auto": Automatically create balanced grid for any views (recommended)
                - "time": Keep original format (V*T in time dimension, no spatial rearrangement)
    Returns:
        Video tensor arranged according to the layout:
        - For "width": (B, C, T, H, V*W)
        - For "height": (B, C, T, V*H, W)
        - For "grid": (B, C, T, 3*H, 3*W) with black padding for None positions
        - For "grid_auto": (B, C, T, rows*H, cols*W) dynamically sized
        - For "time": (B, C, V*T, H, W) (unchanged)
    """
    # Handle "time" mode - return video unchanged
    if method == "time":
        return mv_video

    current_view_order = data_batch["camera_keys_selection"][0]
    n_views = len(current_view_order)
    B, C, VT, H, W = mv_video.shape
    T = VT // n_views

    # Reshape to separate view and time dimensions: B C (V T) H W -> B C V T H W
    video = mv_video.view(B, C, n_views, T, H, W)

    # Create mapping from view name to tensor index
    view_name_to_video_tensor_idx = {view_name: idx for idx, view_name in enumerate(current_view_order)}

    # Create black view for None positions (used in grid layout)
    black_view = th.zeros(B, C, T, H, W, dtype=video.dtype, device=video.device)

    # Get layout definition
    if method == "grid_auto":
        # Dynamic layout based on actual views present
        layout_definition = create_dynamic_grid_layout(list(current_view_order))
    elif method in VISUALIZE_LAYOUTS:
        layout_definition = VISUALIZE_LAYOUTS[method]
    else:
        raise ValueError(
            f"Unsupported visualization method: {method}. Choose from {list(VISUALIZE_LAYOUTS.keys()) + ['grid_auto', 'time']}"
        )

    # Arrange video according to layout
    grid_rows = []
    for row_of_view_names in layout_definition:
        row_tensors = []
        for view_name in row_of_view_names:
            if view_name is not None and view_name in view_name_to_video_tensor_idx:
                tensor_idx = view_name_to_video_tensor_idx[view_name]
                # video is B C V T H W. Get tensor for view: B C T H W
                row_tensors.append(video[:, :, tensor_idx])
            else:
                # Use black view for None positions or missing views
                row_tensors.append(black_view)
        grid_rows.append(th.cat(row_tensors, dim=-1))  # Concat on W dimension

    # Concatenate rows on H dimension
    video = th.cat(grid_rows, dim=-2)  # Concat on H dimension

    return video

def append_dynamic(
    mv_video: th.Tensor,
    n_views: int,
    filenames: list[str],
    save_dir: str,
    fps: float = 10.0,
) -> None:
    """
    Append new video chunks to existing view files.

    Args:
        mv_video: New video chunk tensor with shape (C, V*T, H, W)
        n_views: Number of camera views
        filenames: List of camera names/filenames (without .mp4 extension)
        save_dir: Directory containing the existing view videos
        fps: Frames per second for saved videos
    """
    from cosmos_transfer2._src.imaginaire.utils.easy_io import easy_io

    C, VT, H, W = mv_video.shape
    T = VT // n_views

    # Reshape to separate views: (C, V*T, H, W) -> (C, V, T, H, W)
    video_views = mv_video.view(C, n_views, T, H, W)

    # Process each view
    for view_idx, camera_name in enumerate(filenames):
        new_view_video = video_views[:, view_idx, :, :, :]  # (C, T, H, W)
        view_path = os.path.join(save_dir, f"{camera_name}.mp4")

        # Check if file exists
        if os.path.exists(view_path):
            # Load existing video - Returns (T, H, W, C) numpy array
            existing_frames, _ = easy_io.load(view_path)

            # Concatenate existing frames with new chunk along time dimension
            # Convert existing from (T, H, W, C) numpy to (C, T, H, W) tensor in [0, 1]
            existing_tensor = th.from_numpy(existing_frames).permute(3, 0, 1, 2).float() / 255.0

            # Combine with new frames (C, T, H, W)
            combined_frames = th.cat([
                existing_tensor,
                new_view_video.cpu().float().clamp(0, 1)
            ], dim=1)  # Concatenate on T dimension

            # Save combined video
            save_img_or_video(combined_frames, os.path.join(save_dir, camera_name), fps=fps)
            log.info(f"Appended to view {camera_name} at {view_path}")
        else:
            # File doesn't exist yet, just save the new chunk
            save_img_or_video(new_view_video, os.path.join(save_dir, camera_name), fps=fps)
            log.info(f"Created new view {camera_name} at {view_path}")

    
def create_dynamic(
    mv_video: th.Tensor,
    n_views: int,
    save_dir: str,
    fps: float = 10.0,
    prefix: str = "",
) -> list[str]:
    C, VT, H, W = mv_video.shape
    T = VT // n_views
    
    camera_order = [str(n) for n in range(n_views)]
    filenames = [f"{prefix}{camera_name}" if prefix else camera_name for camera_name in camera_order]
    view_batch = {
        "camera_keys_selection":[camera_order]
    }
    save_each_view_separately(
        mv_video, view_batch, save_dir, fps, prefix
    )

    return filenames

def save_each_view_separately(
    mv_video: th.Tensor,
    data_batch: dict,
    save_dir: str,
    fps: float = 10.0,
    prefix: str = "",
) -> None:
    """
    Save each camera view as a separate video file.

    Args:
        mv_video: Multi-view video tensor with shape (C, V*T, H, W) where V is number of views
        data_batch: Data batch containing camera_keys_selection with actual camera order
        save_dir: Directory to save individual view videos
        fps: Frames per second for saved videos
        prefix: Optional prefix for saved filenames (e.g., "video_" or "control_")
    """
    os.makedirs(save_dir, exist_ok=True)

    # Extract actual camera order from data_batch
    camera_order = data_batch["camera_keys_selection"][0]

    C, VT, H, W = mv_video.shape
    n_views = len(camera_order)
    T = VT // n_views

    # Reshape to separate views: (C, V*T, H, W) -> (C, V, T, H, W)
    video_views = mv_video.view(C, n_views, T, H, W)

    # Save each view
    for view_idx, camera_name in enumerate(camera_order):
        view_video = video_views[:, view_idx, :, :, :]  # (C, T, H, W)
        filename = f"{prefix}{camera_name}" if prefix else camera_name
        view_save_path = os.path.join(save_dir, filename)
        save_img_or_video(view_video, view_save_path, fps=fps)
        log.info(f"Saved view {camera_name} to {view_save_path}")

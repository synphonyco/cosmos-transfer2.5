"""
Multiview Training Wrapper - Main Orchestration Module

This script provides a unified interface for multiview post-training using JSON configurations.
It orchestrates validation, config generation, and training execution.
"""

import argparse
import os
import sys
import subprocess
from pathlib import Path

from scripts.multi_train_validation import (
    validate_inputs,
    validate_dataset_structure,
    validate_view_meta_schema,
    load_json,
    ConfigValidationError,
    DatasetValidationError,
)
from scripts.multi_train_config_gen import (
    generate_experiment_config,
    cleanup_temp_config,
)


def create_parser() -> argparse.ArgumentParser:
    """
    Create argument parser for multiview training wrapper.

    Returns:
        argparse.ArgumentParser: Configured argument parser
    """
    epilog = """
Example usage:
  python scripts/multi_train.py \\
    --input_root /path/to/dataset \\
    --ckpt_save_root /path/to/checkpoints \\
    --configs_dir /path/to/configs \\
    --base_ckpt /path/to/base_checkpoint.pt \\
    --nproc_per_node 8 \\
    --master_port 12341
"""

    parser = argparse.ArgumentParser(
        description="Wrapper for multiview post-training with JSON configs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=epilog
    )

    # Required arguments
    parser.add_argument(
        "--input_root",
        type=str,
        required=True,
        help="Dataset root containing videos/, captions/, control_input_*/, view_meta.json"
    )
    parser.add_argument(
        "--ckpt_save_root",
        type=str,
        required=True,
        help="Root directory for checkpoint saves (sets IMAGINAIRE_OUTPUT_ROOT)"
    )
    parser.add_argument(
        "--configs_dir",
        type=str,
        required=True,
        help="Directory containing checkpoint.json and training.json"
    )
    parser.add_argument(
        "--base_ckpt",
        type=str,
        required=True,
        help="Path to base checkpoint (.pt file) for initialization"
    )

    # Optional arguments
    parser.add_argument(
        "--view_meta",
        type=str,
        default=None,
        help="Path to view_meta.json (defaults to {input_root}/view_meta.json)"
    )
    parser.add_argument(
        "--nproc_per_node",
        type=int,
        default=8,
        help="Number of GPUs per node for distributed training"
    )
    parser.add_argument(
        "--master_port",
        type=int,
        default=12341,
        help="Master port for distributed training"
    )
    parser.add_argument(
        "--job_name",
        type=str,
        default=None,
        help="Override job name from training.json"
    )

    return parser


def run_training_subprocess(config_name: str, args: argparse.Namespace) -> None:
    """
    Execute the training subprocess using torchrun.

    Args:
        config_name: Name of the generated experiment config
        args: Parsed command-line arguments

    Raises:
        RuntimeError: If training subprocess fails
    """
    # Build torchrun command
    cmd = [
        "torchrun",
        f"--nproc_per_node={args.nproc_per_node}",
        f"--master_port={args.master_port}",
        "-m",
        "scripts.train",
        "--config=cosmos_transfer2/_src/transfer2_multiview/configs/vid2vid_transfer/config.py",
        "--",
        f"experiment={config_name}",
        "job.wandb_mode=disabled"
    ]

    # Print execution header
    print("\n" + "=" * 80)
    print("LAUNCHING TRAINING SUBPROCESS")
    print("=" * 80)
    print(f"Command: {' '.join(cmd)}")
    print(f"IMAGINAIRE_OUTPUT_ROOT: {os.environ.get('IMAGINAIRE_OUTPUT_ROOT')}")
    print(f"Experiment name: {config_name}")
    print("=" * 80 + "\n")

    # Execute subprocess
    result = subprocess.run(
        cmd,
        stdout=sys.stdout,
        stderr=sys.stderr,
        env=os.environ.copy()
    )

    # Check return code
    if result.returncode != 0:
        raise RuntimeError(
            f"Training subprocess failed with return code {result.returncode}"
        )

    # Print success message
    checkpoint_path = os.path.join(
        args.ckpt_save_root,
        "checkpoints"
    )
    print("\n" + "=" * 80)
    print("TRAINING COMPLETED SUCCESSFULLY")
    print("=" * 80)
    print(f"Checkpoints saved to: {checkpoint_path}")
    print("=" * 80 + "\n")


def main():
    """Main entry point for multiview training wrapper."""
    parser = create_parser()
    args = parser.parse_args()

    try:
        # Validate inputs
        print("Validating inputs...")
        validate_inputs(args)

        # Load JSON configs
        print("Loading JSON configs...")
        view_meta_path = args.view_meta or os.path.join(args.input_root, "view_meta.json")
        view_meta = load_json(view_meta_path)
        checkpoint_config = load_json(os.path.join(args.configs_dir, "checkpoint.json"))
        training_config = load_json(os.path.join(args.configs_dir, "training.json"))

        # Validate view_meta schema
        print("Validating view_meta schema...")
        validate_view_meta_schema(view_meta)

        # Validate dataset structure
        print("Validating dataset structure...")
        validate_dataset_structure(args.input_root, view_meta)

        # Override job name if provided
        if args.job_name:
            training_config["job"]["name"] = args.job_name

        # Generate experiment config
        print("Generating experiment config...")
        config_path, config_name = generate_experiment_config(
            view_meta,
            checkpoint_config,
            training_config,
            args
        )
        print(f"Generated config: {config_path}")

        # Set environment variable for checkpoint save root
        os.environ["IMAGINAIRE_OUTPUT_ROOT"] = args.ckpt_save_root

        # Run training subprocess
        run_training_subprocess(config_name, args)

        # Cleanup temporary config
        cleanup_temp_config(config_path)

    except ConfigValidationError as e:
        print(f"\nConfiguration validation error: {e}", file=sys.stderr)
        sys.exit(1)
    except DatasetValidationError as e:
        print(f"\nDataset validation error: {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"\nValue error: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user", file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        print(f"\nUnexpected error occurred: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

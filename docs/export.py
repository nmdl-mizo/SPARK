"""Export the eight trusted local epoch-50 checkpoints for publication.

Run with the research environment, not with checkpoints downloaded from others.
Only the resulting weights and portable metadata belong in the release.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch


SEEDS = [27, 42, 123, 777, 1227, 1995, 2024, 2025]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export(source: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    manifest = {"version": "1.0.0", "checkpoint_selection": "last.pt", "epoch": 50,
                "active_learning_round": 5, "training_mode": "from_scratch_on_cumulative_labels",
                "num_members": 8, "members": []}
    baseline = None
    for seed in SEEDS:
        run = f"csiro_active_max_uncertainty_r005_final_init_seed_{seed}"
        path = source / run / "checkpoints/last.pt"
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        args = checkpoint["args"]
        comparison = {
            "model_config": checkpoint["model_config"],
            "dataset_config": checkpoint["dataset_config"],
            "splits": checkpoint["splits"],
            "training_args": {k: v for k, v in args.items() if k not in {"init_seed", "run_id"}},
        }
        if baseline is None:
            baseline = comparison
            manifest["model_config"] = checkpoint["model_config"]
            manifest["graph"] = {k: checkpoint["dataset_config"][k]
                                 for k in ("cutoff", "max_num_neighbors", "triplet_threshold")}
            manifest["reconstruction"] = {"energy_points": 512, "trim_tail_ev": 1.0,
                "adaptive_broadening": True, "sigma_scale": 1.0, "sigma_min_ev": 0.3, "sigma_max_ev": 1.0,
                "std_correction": 0, "grid": "union_of_member_samples_per_atom"}
            manifest["splits"] = {k: {"structures": len(v), "sha256": hashlib.sha256(
                ("\n".join(v) + "\n").encode()).hexdigest()} for k, v in checkpoint["splits"].items()}
            manifest["training"] = {k: args[k] for k in (
                "seed", "epochs", "batch_size", "lr", "weight_decay", "decay_mode", "final_lr_factor",
                "warmup_mode", "warmup_ratio", "warmup_start_factor", "samples_loss_type", "area_weight",
                "init_checkpoint")}
        elif comparison != baseline:
            raise ValueError(f"Ensemble conditions differ beyond initialization seed: {run}")
        if checkpoint["epoch"] != 50 or args["init_seed"] != seed or args["init_checkpoint"] is not None:
            raise ValueError(f"Unexpected training provenance: {run}")
        compact = {"format_version": 1, "model_state_dict": checkpoint["model_state_dict"],
                   "model_config": checkpoint["model_config"], "init_seed": seed, "epoch": 50}
        destination = output / f"seed_{seed}.pt"
        torch.save(compact, destination)
        restored = torch.load(destination, map_location="cpu", weights_only=True)
        for key, tensor in checkpoint["model_state_dict"].items():
            if not torch.equal(tensor, restored["model_state_dict"][key]):
                raise ValueError(f"Export changed parameter {key}")
        manifest["members"].append({"init_seed": seed, "file": destination.name, "epoch": 50,
            "sha256": digest(destination), "source_checkpoint_sha256": digest(path),
            "source_run": run, "bytes": destination.stat().st_size})
    manifest["consistency_check"] = {"passed": True, "only_training_difference": "init_seed",
        "same_model_config": True, "same_dataset_config": True, "same_splits": True,
        "same_training_args": True, "exported_tensors_equal_source": True}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Exported {len(SEEDS)} epoch-50 parameter sets to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Research runs/train directory")
    parser.add_argument("--output", type=Path, default=Path("checkpoints"))
    args = parser.parse_args()
    export(args.source, args.output)

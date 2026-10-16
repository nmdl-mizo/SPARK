from __future__ import annotations

import csv
import hashlib
import json
import tempfile
from pathlib import Path

import numpy as np
import torch

from . import __version__
from .data import StructureData, build_graph_pbc, build_graph_xyz, energy_reference_for_z
from .model import SPARK
from .utils import reconstruct_from_samples


ELEMENTS = {78: "Pt", 27: "Co"}


def resource(name: str) -> Path:
    """Find resources in either a source checkout or an installed wheel."""
    package = Path(__file__).resolve().parent
    path = package / name
    return path if path.is_dir() else package.parent / name


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_members(directory: Path, member: int | None = None) -> tuple[dict, list[dict]]:
    manifest = json.loads((directory / "manifest.json").read_text())
    records = manifest["members"]
    if len(records) != 8 or len({r["init_seed"] for r in records}) != 8:
        raise ValueError("The publication ensemble requires eight distinct initialization seeds.")
    if member is not None:
        records = [r for r in records if r["init_seed"] == member]
        if not records:
            raise ValueError(f"Unknown initialization seed: {member}")
    loaded = []
    for record in records:
        path = directory / record["file"]
        if sha256(path) != record["sha256"]:
            raise ValueError(f"Checkpoint checksum mismatch: {path}")
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if checkpoint["model_config"] != manifest["model_config"]:
            raise ValueError(f"Model configuration mismatch: {path}")
        if checkpoint["init_seed"] != record["init_seed"] or checkpoint["epoch"] != 50:
            raise ValueError(f"Checkpoint provenance mismatch: {path}")
        loaded.append(checkpoint)
    return manifest, loaded


def build_model(config: dict) -> SPARK:
    config = dict(config)
    config["out_channels"] = config.pop("n_samples")
    return SPARK(**config)


def graph(path: Path, config: dict) -> StructureData:
    settings = {k: config[k] for k in ("cutoff", "max_num_neighbors")}
    if path.suffix.lower() == ".xyz":
        values = build_graph_xyz(path, **settings)
    else:
        values = build_graph_pbc(path, triplet_threshold=config["triplet_threshold"], **settings)
    unsupported = set(values["z"].tolist()) - ELEMENTS.keys()
    if unsupported:
        raise ValueError(f"Supported elements are Pt and Co; found atomic numbers {sorted(unsupported)}")
    if not torch.isfinite(values["dists"]).all() or (values["dists"] <= 0).any():
        raise ValueError(f"Structure contains invalid interatomic distances: {path}")
    return StructureData(**values, num_nodes=len(values["z"]))


def grid(samples: torch.Tensor, points: int = 512, trim: float = 1.0) -> torch.Tensor:
    lower = samples.amin(dim=(0, 2))
    upper = torch.maximum(samples.amax(dim=(0, 2)) - trim, lower + 1e-3)
    fraction = torch.linspace(0, 1, points, device=samples.device)
    return lower[:, None] + (upper - lower)[:, None] * fraction


def reconstruct(energy: torch.Tensor, samples: torch.Tensor, areas: torch.Tensor) -> torch.Tensor:
    """Bound peak memory by reconstructing eight atoms at a time."""
    members, atoms, _ = samples.shape
    result = torch.empty((members, atoms, energy.shape[1]), dtype=samples.dtype, device=samples.device)
    for start in range(0, atoms, 8):
        stop = min(start + 8, atoms)
        for model in range(members):
            result[model, start:stop] = reconstruct_from_samples(
                energy_grid=energy[start:stop],
                samples=samples[model, start:stop],
                sigma=0.3,
                areas=areas[model, start:stop],
                adaptive_broadening=True,
                sigma_scale=1.0,
                sigma_min=0.3,
                sigma_max=1.0,
            )
    return result


def uncertainty(energy: torch.Tensor, spectra: torch.Tensor, areas: torch.Tensor) -> dict[str, torch.Tensor]:
    std = spectra.std(dim=0, unbiased=False)
    area_mean = areas.mean(dim=0)
    return {
        "unc_mean_std": std.mean(dim=1),
        "unc_rms_std": std.square().mean(dim=1).sqrt(),
        "unc_max_std": std.amax(dim=1),
        "unc_integrated_var": torch.trapezoid(std.square(), energy, dim=1),
        "area_mean": area_mean,
        "area_unc_std": areas.std(dim=0, unbiased=False),
        "area_rel_unc_std": areas.std(dim=0, unbiased=False) / area_mean.abs().clamp_min(1e-12),
    }


def write_atoms(path: Path, energy: torch.Tensor, values: torch.Tensor) -> None:
    columns = []
    names = []
    for atom in range(len(energy)):
        columns.extend([energy[atom], values[atom]])
        names.extend([f"atom_{atom}_energy", f"atom_{atom}_mu"])
    np.savetxt(path, torch.stack(columns, dim=1).cpu().numpy(), header=" ".join(names), comments="", fmt="%.9g")


def write_outputs(directory: Path, sample: str, z: torch.Tensor, samples: torch.Tensor,
                  areas: torch.Tensor, normalized: torch.Tensor, manifest: dict,
                  seeds: list[int]) -> list[Path]:
    energy = grid(samples)
    spectra = reconstruct(energy, samples, areas)
    mean = spectra.mean(dim=0)
    std = spectra.std(dim=0, unbiased=False)
    paths = [directory / f"{sample}_prediction.dat"]
    write_atoms(paths[0], energy, mean)
    ensemble = len(seeds) > 1
    if ensemble:
        paths.append(directory / f"{sample}_std.dat")
        write_atoms(paths[-1], energy, std)

    # Average atoms within each model first, then compute model-to-model spread.
    # This preserves correlations between the sites in an element-average spectrum.
    columns, names = [], []
    for number, symbol in ELEMENTS.items():
        mask = z == number
        if not mask.any():
            continue
        element_samples = samples[:, mask]
        element_grid = grid(element_samples.reshape(len(seeds), 1, -1)).expand(int(mask.sum()), -1)
        per_model = reconstruct(element_grid, element_samples, areas[:, mask]).mean(dim=1)
        columns.extend([element_grid[0], per_model.mean(dim=0)])
        names.extend([f"{symbol}_energy", f"{symbol}_mu"])
        if ensemble:
            columns.append(per_model.std(dim=0, unbiased=False))
            names.append(f"{symbol}_std")
    paths.append(directory / f"{sample}_mean.dat")
    np.savetxt(paths[-1], torch.stack(columns, dim=1).cpu().numpy(), header=" ".join(names), comments="", fmt="%.9g")

    if ensemble:
        metrics = uncertainty(energy, spectra, areas)
        # Shape-only uncertainty removes each member's predicted area.
        base = reconstruct(energy, samples, torch.ones_like(areas))
        norm = torch.trapezoid(base, energy.unsqueeze(0), dim=2).clamp_min(1e-12)
        shape_std = (base / norm[:, :, None]).std(dim=0, unbiased=False)
        metrics["samples_unc_rms_std"] = shape_std.square().mean(dim=1).sqrt()
        metrics["combined_samples_area_unc"] = (
            metrics["samples_unc_rms_std"].square() + metrics["area_rel_unc_std"].square()
        ).sqrt()
        paths.append(directory / f"{sample}_uncertainty.csv")
        with paths[-1].open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["sample_id", "atom_idx", "element", "num_models", *metrics])
            writer.writeheader()
            for atom, number in enumerate(z.tolist()):
                writer.writerow({"sample_id": sample, "atom_idx": atom, "element": ELEMENTS[number],
                                 "num_models": len(seeds), **{k: float(v[atom]) for k, v in metrics.items()}})

    paths.append(directory / f"{sample}_samples.npz")
    np.savez_compressed(paths[-1], samples=samples.cpu().numpy(),
                        normalized_samples=normalized.cpu().numpy(),
                        areas=areas.cpu().numpy(), atomic_numbers=z.cpu().numpy(), init_seeds=np.array(seeds))
    paths.append(directory / f"{sample}_prediction.json")
    metadata = {
        "version": __version__, "sample_id": sample, "atoms": len(z), "init_seeds": seeds,
        "num_models": len(seeds), "checkpoint_selection": manifest["checkpoint_selection"],
        "epoch": 50, "graph": manifest["graph"], "reconstruction": manifest["reconstruction"],
        "structure_sha256": manifest["structure_sha256"],
        "checkpoint_sha256": {str(r["init_seed"]): r["sha256"] for r in manifest["members"] if r["init_seed"] in seeds},
        "files": [p.name for p in paths],
    }
    paths[-1].write_text(json.dumps(metadata, indent=2) + "\n")
    return paths


def output_paths(directory: Path, sample: str, ensemble: bool) -> list[Path]:
    suffixes = ["prediction.dat", "mean.dat", "samples.npz", "prediction.json"]
    if ensemble:
        suffixes.extend(["std.dat", "uncertainty.csv"])
    return [directory / f"{sample}_{suffix}" for suffix in suffixes]


def predict(root: str | Path, *, ids: list[str] | None = None, member: int | None = None,
            device: str = "cpu", overwrite: bool = False, checkpoints: str | Path | None = None,
            threads: int = 4) -> list[Path]:
    """Predict structures without reading any reference or target XANES files."""
    root = Path(root).resolve()
    directory = root / "structure"
    paths = sorted(directory.glob("*.xyz")) + sorted(directory.glob("*.vasp"))
    by_id = {}
    for path in paths:
        if path.stem in by_id:
            raise ValueError(f"Ambiguous structure ID (both XYZ and VASP): {path.stem}")
        by_id[path.stem] = path
    if ids is not None:
        if len(set(ids)) != len(ids):
            raise ValueError("Structure IDs must be unique.")
        missing = set(ids) - by_id.keys()
        if missing:
            raise ValueError(f"Missing structure IDs: {sorted(missing)}")
        paths = [by_id[sample] for sample in ids]
    if not paths:
        raise ValueError(f"No .xyz or .vasp structures found in {directory}")
    if threads < 1:
        raise ValueError("CPU thread count must be positive.")
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable. Use --device cpu.")
    torch.set_num_threads(threads)
    output = root / "xanes"
    output.mkdir(parents=True, exist_ok=True)
    # Reserve all output names before writing, including artifacts from a prior
    # ensemble when switching to single-member inference.
    if not overwrite:
        existing = [p for path in paths for p in output_paths(output, path.stem, True) if p.exists()]
        if existing:
            raise FileExistsError(f"Prediction outputs already exist: {existing[0]}. Use --overwrite to replace predictions.")
    manifest, members = load_members(Path(checkpoints) if checkpoints else resource("checkpoints"), member)
    model = build_model(manifest["model_config"]).to(device).eval()
    results = []
    for path in paths:
        data = graph(path, manifest["graph"]).to(device)
        centers, scales = energy_reference_for_z(data.z)
        samples, areas, normalized = [], [], []
        with torch.inference_mode():
            for checkpoint in members:
                model.load_state_dict(checkpoint["model_state_dict"], strict=True)
                shape, area = model(data)
                normalized.append(shape)
                shape = shape * scales.to(device) + centers.to(device)
                if not torch.isfinite(shape).all() or not torch.isfinite(area).all():
                    raise ValueError(f"Non-finite model output for {path.name}")
                samples.append(shape)
                areas.append(area.view(-1))
            config = {**manifest, "structure_sha256": sha256(path)}
            with tempfile.TemporaryDirectory(prefix=".spark-", dir=output) as staging:
                generated = write_outputs(Path(staging), path.stem, data.z, torch.stack(samples),
                    torch.stack(areas), torch.stack(normalized), config, [c["init_seed"] for c in members])
                current = []
                for generated_path in generated:
                    destination = output / generated_path.name
                    generated_path.replace(destination)
                    current.append(destination)
            if overwrite:
                for obsolete in set(output_paths(output, path.stem, True)) - set(current):
                    obsolete.unlink(missing_ok=True)
        results.extend(current)
        print(f"{path.name}: {data.num_nodes} atoms, {len(members)} model(s) -> {current[0]}")
    return results

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import torch
from pymatgen.core import Molecule, Structure
from torch import Tensor
from torch_geometric.data import Data, Dataset
from torch_geometric.nn import radius_graph
from torch_geometric.typing import SparseTensor

try:
    from .utils import interpolate_spectrum, inverse_transform
except ImportError:
    from utils import interpolate_spectrum, inverse_transform


ATOM_COLUMN_PATTERN = re.compile(r"atom_(\d+)_(energy|mu)$")
CACHE_VERSION = "v3"
ENERGY_REFERENCE_BY_Z = {
    27: (7745.0, 35.0),   # Co K-edge window center and half-width.
    78: (11585.0, 35.0),  # Pt L3-edge window center and half-width.
}



class StructureData(Data):
    """PyG Data with correct batching offsets for cached DimeNet indices."""

    def __inc__(self, key, value, *args, **kwargs):
        if key == "edge_index":
            return int(self.num_nodes)
        if key in {"i", "j"}:
            return int(self.num_nodes)
        if key in {"idx_kj", "idx_ji"}:
            return int(self.edge_index.size(1))
        return super().__inc__(key, value, *args, **kwargs)


def energy_reference_for_z(z: Tensor) -> tuple[Tensor, Tensor]:
    centers = torch.zeros_like(z, dtype=torch.float32)
    scales = torch.ones_like(z, dtype=torch.float32)
    for atomic_number, (center, scale) in ENERGY_REFERENCE_BY_Z.items():
        mask = z == atomic_number
        centers[mask] = float(center)
        scales[mask] = float(scale)
    return centers.view(-1, 1), scales.view(-1, 1)


def scale_energy_samples(samples: Tensor, z: Tensor) -> Tensor:
    centers, scales = energy_reference_for_z(z)
    return (samples - centers.to(samples.device)) / scales.to(samples.device)


def unscale_energy_samples(samples: Tensor, centers: Tensor, scales: Tensor) -> Tensor:
    return samples * scales.to(samples.device) + centers.to(samples.device)


class StructureXANESDataset(Dataset):
    """Dataset for cached structure graphs and atom-resolved XANES targets.

    Expected layout:
        root/
          structure/{id}.xyz or {id}.vasp
          xanes/{id}.dat
          processed/{cache_key}/{id}.pt
    """

    def __init__(
        self,
        root: str | Path = "dataset",
        cutoff: float = 5.0,
        max_num_neighbors: int = 32,
        n_samples: int = 256,
        triplet_threshold: float = 1.0e-6,
        ids: Optional[Iterable[str]] = None,
        cache: bool = True,
        rebuild: bool = False,
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ):
        self.root_dir = Path(root)
        self.structure_dir = self.root_dir / "structure"
        self.xanes_dir = self.root_dir / "xanes"
        self.cutoff = float(cutoff)
        self.max_num_neighbors = int(max_num_neighbors)
        self.n_samples = int(n_samples)
        self.triplet_threshold = float(triplet_threshold)
        self.cache = bool(cache)
        self.rebuild = bool(rebuild)
        self.ids = list(ids) if ids is not None else self._discover_ids()
        self._memory_cache: Dict[int, StructureData] = {}
        self._cache_key = self._build_cache_key()
        self._processed_dir = self.root_dir / "processed" / self._cache_key
        self._processed_dir.mkdir(parents=True, exist_ok=True)

        super().__init__(str(self.root_dir), transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return [f"{sample_id}.pt" for sample_id in self.ids]

    def download(self):
        return None

    def _process(self) -> None:
        """Skip PyG's eager metadata path; samples are cached lazily in get()."""

        return None

    def process(self):
        return None

    def len(self) -> int:
        return len(self.ids)

    def get(self, idx: int) -> StructureData:
        if self.cache and not self.rebuild and idx in self._memory_cache:
            return self._memory_cache[idx].clone()

        sample_id = self.ids[idx]
        processed_path = self.processed_path(sample_id)

        if self.cache and not self.rebuild and processed_path.exists():
            try:
                data = torch.load(
                    processed_path,
                    map_location="cpu",
                    weights_only=False,
                )
            except (EOFError, OSError, RuntimeError):
                pass
            else:
                self._memory_cache[idx] = data.clone()
                return data

        data = self._build_sample(sample_id)
        if self.cache:
            temporary = processed_path.with_name(
                f".{processed_path.name}.{os.getpid()}.tmp"
            )
            try:
                torch.save(data, temporary)
                temporary.replace(processed_path)
            finally:
                temporary.unlink(missing_ok=True)
            self._memory_cache[idx] = data.clone()
        return data

    @property
    def cache_key(self) -> str:
        return self._cache_key

    @property
    def processed_dir_path(self) -> Path:
        return self._processed_dir

    def processed_path(self, sample_id: str) -> Path:
        return self._processed_dir / f"{sample_id}.pt"

    def clear_memory_cache(self) -> None:
        self._memory_cache.clear()

    def _build_cache_key(self) -> str:
        config = {
            "version": CACHE_VERSION,
            "cutoff": self.cutoff,
            "max_num_neighbors": self.max_num_neighbors,
            "n_samples": self.n_samples,
            "triplet_threshold": self.triplet_threshold,
        }
        raw = json.dumps(config, sort_keys=True)
        digest = hashlib.md5(raw.encode("utf-8")).hexdigest()[:12]
        return (
            f"{CACHE_VERSION}_cutoff_{self.cutoff:g}_"
            f"nbr_{self.max_num_neighbors}_ns_{self.n_samples}_"
            f"thr_{self.triplet_threshold:.1e}_{digest}"
        )

    def _discover_ids(self) -> list[str]:
        if not self.structure_dir.exists():
            return []
        structure_paths = sorted(self.structure_dir.glob("*.xyz")) + sorted(
            self.structure_dir.glob("*.vasp")
        )
        ids: list[str] = []
        seen: set[str] = set()
        for path in structure_paths:
            if path.stem not in seen:
                ids.append(path.stem)
                seen.add(path.stem)
        return ids

    def _find_structure_path(self, sample_id: str) -> Optional[Path]:
        xyz_path = self.structure_dir / f"{sample_id}.xyz"
        if xyz_path.exists():
            return xyz_path

        vasp_path = self.structure_dir / f"{sample_id}.vasp"
        if vasp_path.exists():
            return vasp_path

        return None

    def _build_sample(self, sample_id: str) -> StructureData:
        structure_path = self._find_structure_path(sample_id)
        if structure_path is None:
            raise FileNotFoundError(f"No structure file found for sample '{sample_id}'")

        if structure_path.suffix.lower() == ".xyz":
            graph = build_graph_xyz(
                structure_path=structure_path,
                cutoff=self.cutoff,
                max_num_neighbors=self.max_num_neighbors,
            )
        elif structure_path.suffix.lower() == ".vasp":
            graph = build_graph_pbc(
                structure_path=structure_path,
                cutoff=self.cutoff,
                max_num_neighbors=self.max_num_neighbors,
                triplet_threshold=self.triplet_threshold,
            )
        else:
            raise ValueError(f"Unsupported structure format: {structure_path.suffix}")

        samples, area, mask = self._load_xanes(sample_id, z=graph["z"])
        energy_center, energy_scale = energy_reference_for_z(graph["z"])

        return StructureData(
            z=graph["z"],
            edge_index=graph["edge_index"],
            dists=graph["dists"],
            i=graph["i"],
            j=graph["j"],
            idx_kj=graph["idx_kj"],
            idx_ji=graph["idx_ji"],
            angle=graph["angle"],
            samples=samples,
            area=area,
            mask=mask,
            energy_center=energy_center,
            energy_scale=energy_scale,
            num_nodes=int(graph["z"].size(0)),
            sample_id=sample_id,
            source_format=structure_path.suffix.lower(),
        )

    def _load_xanes(self, sample_id: str, z: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        n_atoms = int(z.size(0))
        samples = torch.zeros((n_atoms, self.n_samples), dtype=torch.float32)
        area = torch.zeros((n_atoms, 1), dtype=torch.float32)
        mask = torch.zeros((n_atoms, 1), dtype=torch.bool)

        xanes_path = self.xanes_dir / f"{sample_id}.dat"
        if not xanes_path.exists():
            return samples, area, mask

        with xanes_path.open("r", encoding="utf-8") as handle:
            lines = [line.strip() for line in handle if line.strip()]

        if len(lines) < 2:
            return samples, area, mask

        header = lines[0].split()
        values = np.loadtxt(str(xanes_path), skiprows=1)
        if values.ndim == 1:
            values = values.reshape(1, -1)

        atom_columns = parse_atom_columns(header)
        for atom_idx, columns in atom_columns.items():
            if atom_idx >= n_atoms:
                continue

            energy_col = columns.get("energy")
            mu_col = columns.get("mu")
            if energy_col is None or mu_col is None:
                continue
            if energy_col >= values.shape[1] or mu_col >= values.shape[1]:
                continue

            energy = torch.as_tensor(values[:, energy_col], dtype=torch.float32)
            mu = torch.as_tensor(values[:, mu_col], dtype=torch.float32)
            valid = torch.isfinite(energy) & torch.isfinite(mu)
            if valid.sum() < 2:
                continue

            energy = energy[valid]
            mu = mu[valid]
            order = torch.argsort(energy)
            energy = energy[order]
            mu = mu[order]
            energy, mu = interpolate_spectrum(energy, mu, dtype=torch.float32)

            sampled_energy, sampled_area = inverse_transform(
                energy,
                mu,
                n_samples=self.n_samples,
            )
            samples[atom_idx] = scale_energy_samples(
                sampled_energy.to(samples.dtype).view(1, -1),
                z[atom_idx:atom_idx + 1],
            ).view(-1)
            area[atom_idx, 0] = sampled_area.to(area.dtype)
            mask[atom_idx, 0] = True

        return samples, area, mask


def parse_atom_columns(header: list[str]) -> dict[int, dict[str, int]]:
    atom_columns: dict[int, dict[str, int]] = {}
    for column_idx, column_name in enumerate(header):
        match = ATOM_COLUMN_PATTERN.match(column_name)
        if match is None:
            continue
        atom_idx = int(match.group(1))
        field = match.group(2)
        atom_columns.setdefault(atom_idx, {})[field] = column_idx
    return atom_columns


def build_graph_xyz(
    structure_path: str | Path,
    cutoff: float,
    max_num_neighbors: int,
) -> dict[str, Tensor]:
    molecule = Molecule.from_file(str(structure_path))
    pos = torch.tensor(np.asarray(molecule.cart_coords), dtype=torch.float32)
    z = torch.tensor([site.specie.Z for site in molecule], dtype=torch.long)

    edge_index = radius_graph(
        x=pos,
        r=cutoff,
        batch=None,
        max_num_neighbors=max_num_neighbors,
    )
    if edge_index.numel() == 0:
        raise ValueError(
            f"No edges were found with cutoff={cutoff}. Increase cutoff or inspect {structure_path}."
        )

    i, j, _, _, _, idx_kj, idx_ji = triplets(edge_index=edge_index, num_nodes=z.size(0))
    dists = (pos[i] - pos[j]).pow(2).sum(dim=-1).sqrt()

    vec_kj = pos[edge_index[0, idx_kj]] - pos[edge_index[1, idx_kj]]
    vec_ji = pos[edge_index[0, idx_ji]] - pos[edge_index[1, idx_ji]]
    angle = compute_angle_from_vectors(vec_kj=vec_kj, vec_ji=vec_ji)

    return {
        "z": z,
        "edge_index": edge_index,
        "dists": dists,
        "i": i,
        "j": j,
        "idx_kj": idx_kj,
        "idx_ji": idx_ji,
        "angle": angle,
    }


def build_graph_pbc(
    structure_path: str | Path,
    cutoff: float,
    max_num_neighbors: int,
    triplet_threshold: float,
) -> dict[str, Tensor]:
    structure = Structure.from_file(str(structure_path))
    z = torch.tensor(structure.atomic_numbers, dtype=torch.long)
    edge_index, edge_vec = radius_graph_pbc(
        structure=structure,
        cutoff=cutoff,
        max_num_neighbors=max_num_neighbors,
    )
    dists = edge_vec.norm(dim=-1)

    i, j, _, _, _, idx_kj, idx_ji = triplets_pbc(edge_index=edge_index, num_nodes=z.size(0))
    idx_kj, idx_ji, angle = compute_pbc_angle(
        edge_vec=edge_vec,
        idx_kj=idx_kj,
        idx_ji=idx_ji,
        threshold=triplet_threshold,
    )

    return {
        "z": z,
        "edge_index": edge_index,
        "dists": dists,
        "i": i,
        "j": j,
        "idx_kj": idx_kj,
        "idx_ji": idx_ji,
        "angle": angle,
    }


def triplets(
    edge_index: Tensor,
    num_nodes: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    row, col = edge_index

    value = torch.arange(row.size(0), device=row.device)
    adj_t = SparseTensor(row=col, col=row, value=value, sparse_sizes=(num_nodes, num_nodes))
    adj_t_row = adj_t[row]
    num_triplets = adj_t_row.set_value(None).sum(dim=1).to(torch.long)

    idx_i = col.repeat_interleave(num_triplets)
    idx_j = row.repeat_interleave(num_triplets)
    idx_k = adj_t_row.storage.col()
    mask = idx_i != idx_k
    idx_i = idx_i[mask]
    idx_j = idx_j[mask]
    idx_k = idx_k[mask]

    idx_kj = adj_t_row.storage.value()[mask]
    idx_ji = adj_t_row.storage.row()[mask]

    return col, row, idx_i, idx_j, idx_k, idx_kj, idx_ji


def triplets_pbc(
    edge_index: Tensor,
    num_nodes: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    row, col = edge_index  # j->i

    value = torch.arange(row.size(0), device=row.device)
    adj_t = SparseTensor(row=col, col=row, value=value,
                         sparse_sizes=(num_nodes, num_nodes))
    adj_t_row = adj_t[row]
    num_triplets = adj_t_row.set_value(None).sum(dim=1).to(torch.long)

    # Node indices (k->j->i) for triplets.
    idx_i = col.repeat_interleave(num_triplets)
    idx_j = row.repeat_interleave(num_triplets)
    idx_k = adj_t_row.storage.col()
    # mask = idx_i != idx_k  # Remove i == k triplets.
    # idx_i, idx_j, idx_k = idx_i[mask], idx_j[mask], idx_k[mask]

    # Edge indices (k-j, j->i) for triplets.
    idx_kj = adj_t_row.storage.value()
    idx_ji = adj_t_row.storage.row()

    return col, row, idx_i, idx_j, idx_k, idx_kj, idx_ji


def compute_angle_from_vectors(vec_kj: Tensor, vec_ji: Tensor) -> Tensor:
    a = (vec_ji * vec_kj).sum(dim=-1)
    b = torch.cross(vec_ji, vec_kj, dim=1).norm(dim=-1)
    return torch.atan2(b, a)


def compute_angle(edge_vec: Tensor, idx_kj: Tensor, idx_ji: Tensor) -> Tensor:
    vec_kj = edge_vec[idx_kj]
    vec_ji = edge_vec[idx_ji]
    a = (vec_ji * vec_kj).sum(dim=-1)
    b = torch.cross(vec_ji, vec_kj, dim=1).norm(dim=-1)
    return torch.atan2(b, a)


def compute_pbc_angle(
    edge_vec: Tensor,
    idx_kj: Tensor,
    idx_ji: Tensor,
    threshold: float,
) -> tuple[Tensor, Tensor, Tensor]:
    vec_kj = edge_vec[idx_kj]
    vec_ji = edge_vec[idx_ji]
    keep = (vec_kj + vec_ji).norm(dim=1) > threshold
    idx_kj = idx_kj[keep]
    idx_ji = idx_ji[keep]
    vec_kj = vec_kj[keep]
    vec_ji = vec_ji[keep]

    a = (vec_ji * vec_kj).sum(dim=-1)
    b = torch.cross(vec_ji, vec_kj, dim=1).norm(dim=-1)
    angle = torch.atan2(b, a)
    return idx_kj, idx_ji, angle


def radius_graph_pbc(
    structure: Structure,
    cutoff: float,
    max_num_neighbors: int,
) -> tuple[Tensor, Tensor]:
    edge_pairs: list[list[int]] = []
    edge_vectors: list[np.ndarray] = []

    all_neighbors = structure.get_all_neighbors(cutoff, include_index=True)
    for center_idx, site in enumerate(structure):
        neighbors = sorted(all_neighbors[center_idx], key=lambda item: item.nn_distance)
        neighbors = neighbors[:max_num_neighbors]
        for neighbor in neighbors:
            neighbor_idx = int(neighbor.index)
            edge_pairs.append([neighbor_idx, center_idx])
            edge_vectors.append(np.asarray(neighbor.coords - site.coords, dtype=np.float64))

    if not edge_pairs:
        raise ValueError(
            f"No edges were found with cutoff={cutoff}. Increase cutoff or inspect structure."
        )

    edge_index = torch.tensor(edge_pairs, dtype=torch.long).t().contiguous()
    edge_vec = torch.tensor(np.asarray(edge_vectors), dtype=torch.float32)
    return edge_index, edge_vec

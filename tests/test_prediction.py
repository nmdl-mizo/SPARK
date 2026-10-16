from __future__ import annotations

import json
import shutil

import numpy as np
import pytest
import torch
from pymatgen.core import Lattice, Structure

from spark import cli
from spark.predict import graph, load_members, predict, resource, write_outputs


@pytest.fixture(autouse=True)
def cpu_threads():
    torch.set_num_threads(2)


def test_periodic_graph(tmp_path):
    structure = Structure(Lattice.cubic(3.9), ["Pt", "Pt", "Co", "Co"],
                          [[0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]])
    path = tmp_path / "cell.vasp"
    structure.to(filename=path, fmt="poscar")
    data = graph(path, {"cutoff": 5.0, "max_num_neighbors": 48, "triplet_threshold": 1e-4})
    assert data.num_nodes == 4
    assert len(data.dists) > 4 * 3  # Periodic images contribute neighbors.
    assert len(data.idx_kj) > 0
    assert torch.isfinite(data.angle).all()
    assert data.edge_index.max() == 3


def test_manifest_consistency():
    manifest, members = load_members(resource("checkpoints"))
    assert len(members) == 8
    assert manifest["consistency_check"]["passed"]
    assert len({r["sha256"] for r in manifest["members"]}) == 8
    assert {m["epoch"] for m in members} == {50}
    assert all("optimizer_state_dict" not in m for m in members)
    with pytest.raises(ValueError, match="Unknown initialization seed"):
        load_members(resource("checkpoints"), 0)


def test_existing_prediction_is_protected(tmp_path):
    (tmp_path / "structure").mkdir()
    (tmp_path / "xanes").mkdir()
    shutil.copy2(resource("example") / "structure/104795.xyz", tmp_path / "structure/104795.xyz")
    target = tmp_path / "xanes/104795_prediction.dat"
    target.write_text("existing prediction")
    with pytest.raises(FileExistsError, match="already exist"):
        predict(tmp_path)
    assert target.read_text() == "existing prediction"


def test_demo_does_not_replace_reference(tmp_path, monkeypatch):
    (tmp_path / "xanes").mkdir()
    reference = tmp_path / "xanes/104795_feff_reference.dat"
    reference.write_text("user reference")
    monkeypatch.setattr("sys.argv", ["spark", "demo", "--output", str(tmp_path), "--overwrite"])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2
    assert reference.read_text() == "user reference"


def test_element_average_uncertainty(tmp_path):
    # Opposite site shifts cancel in the structure mean. Averaging site-level
    # standard deviations would incorrectly produce a nonzero result.
    first = torch.linspace(7710, 7780, 512)
    second = first + 2
    samples = torch.stack([torch.stack([first, second]), torch.stack([second, first])])
    manifest = json.loads((resource("checkpoints") / "manifest.json").read_text())
    manifest["structure_sha256"] = "synthetic"
    paths = write_outputs(tmp_path, "synthetic", torch.tensor([27, 27]), samples,
                          torch.ones(2, 2), samples, manifest, [27, 42])
    table = np.loadtxt(tmp_path / "synthetic_mean.dat", skiprows=1)
    assert np.max(np.abs(table[:, 2])) == 0
    assert np.max(np.loadtxt(tmp_path / "synthetic_std.dat", skiprows=1)[:, 1]) > 0
    assert len(paths) == 6


def test_single_member_replaces_ensemble_outputs(tmp_path, monkeypatch):
    import importlib
    module = importlib.import_module("spark.predict")
    (tmp_path / "structure").mkdir()
    (tmp_path / "structure/triangle.xyz").write_text("3\nfinite example\nCo 0 0 0\nPt 2 0 0\nPt 1 2 0\n")
    output = tmp_path / "xanes"
    output.mkdir()
    reference = output / "triangle_feff_reference.dat"
    reference.write_text("preserve this answer")
    for name in ("triangle_std.dat", "triangle_uncertainty.csv"):
        (output / name).write_text("old ensemble output")

    class Model(torch.nn.Module):
        def load_state_dict(self, state, strict=True):
            return None

        def forward(self, data):
            return torch.linspace(-1, 1, 512).expand(data.num_nodes, -1), torch.ones(data.num_nodes, 1)

    manifest = json.loads((resource("checkpoints") / "manifest.json").read_text())
    monkeypatch.setattr(module, "build_model", lambda config: Model())
    monkeypatch.setattr(module, "load_members", lambda directory, member: (
        manifest, [{"init_seed": member, "model_state_dict": {}}]))
    predict(tmp_path, member=42, overwrite=True, threads=2)
    assert not (output / "triangle_std.dat").exists()
    assert not (output / "triangle_uncertainty.csv").exists()
    assert reference.read_text() == "preserve this answer"
    assert json.loads((output / "triangle_prediction.json").read_text())["num_models"] == 1

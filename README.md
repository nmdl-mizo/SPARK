# SPARK

**Spectral Prediction from Atomistic Representations frameworK** predicts
atom-resolved X-ray absorption near-edge structure (XANES) directly from atomic
structures. This publication release provides the trained Pt/Co model, eight
independent parameter sets, and ensemble uncertainty estimation. It covers
**Pt L3-edge** and **Co K-edge** spectra for Pt–Co structures.

The release includes one 105-atom CSIRO nanoparticle and its independently
calculated FEFF spectra. The full research datasets, training caches, optimizer
states, and experiment logs are excluded. No FEFF installation, W&B account, or
access to the original training workspace is required for prediction.

## Install and run

The supported installation uses **Python 3.11 or 3.12 on Linux x86-64**.
CPU inference is the default. The eight trained models are included in the
repository and the installed Python package; no model download is needed after
installation. Allow about 100 MB for model parameters plus the Python runtime
and dependencies.

```bash
git clone https://github.com/nmdl-mizo/SPARK.git
cd SPARK
python3.11 -m venv .venv
source .venv/bin/activate
bash install.sh
spark demo
```

`spark demo` copies the bundled example to `structure/104795.xyz` and the FEFF
answer to `xanes/104795_feff_reference.dat`, then **runs all eight models** and
creates new predictions directly in `xanes/`. The FEFF answer is never used as
input to the predictor. The command preserves any existing reference file and
refuses to replace prediction files unless `--overwrite` is supplied.

```text
structure/
└── 104795.xyz
xanes/
├── 104795_feff_reference.dat   # included FEFF answer, 105 sites
├── 104795_prediction.dat       # computed ensemble-mean spectrum at every site
├── 104795_std.dat              # computed pointwise ensemble standard deviation
├── 104795_mean.dat             # computed element averages and their uncertainty
├── 104795_uncertainty.csv      # computed scalar uncertainty for every site
├── 104795_samples.npz          # predicted samples and areas for every model/site
└── 104795_prediction.json      # settings, model checksums, and output inventory
```

To rerun or choose another destination:

```bash
spark demo --overwrite
spark demo --output my_example
```

For NVIDIA GPU installation and execution, see [GPU inference](#gpu-inference).
The installation script pins PyTorch 2.8.0 and PyG 2.7.0 and
installs matching compiled graph extensions from the
[official PyG wheel index](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html).
Use the script for a new environment so these extensions match your PyTorch
build. The Python dependencies are listed in `pyproject.toml`.

## Predict your own structures

Create a directory with `structure/` and, optionally, `xanes/`:

```text
my_data/
├── structure/
│   ├── particle.xyz
│   └── crystal.vasp
└── xanes/                     # created automatically if absent
```

```bash
spark predict --root my_data
spark predict --root my_data --ids particle
spark predict --root my_data --ids particle --overwrite
```

Each structure produces matching `<structure_id>_*` files in `my_data/xanes/`.
The predictor reads only `structure/` and the bundled model weights. It does
not need experimental spectra or FEFF labels. Structure IDs are filename stems;
preserve leading zeros, and use a different ID for each structure. Providing
both `particle.xyz` and `particle.vasp` is ambiguous and raises an error.

- **XYZ:** atom count, comment, then `element x y z`, with Cartesian coordinates
  in Å. XYZ is treated as a finite, nonperiodic object, including extended XYZ.
- **VASP/POSCAR:** save as `<id>.vasp`; lattice and periodic images are used.
- **Elements:** only Pt and Co are supported by this trained release.
- **Indices:** atom indices are zero-based and retain the input file's atom
  order. Do not reorder atoms when comparing with reference columns.

For a single model prediction:

```bash
spark predict --root my_data --member 42
```

`--member` selects an initialization seed from
`27, 42, 123, 777, 1227, 1995, 2024, 2025`. Omitting it uses all eight models.
A single member writes its spectrum, element mean, samples/areas, and metadata;
uncertainty files require the ensemble. Use a separate directory to compare
single-member and ensemble results, or explicitly replace predictions with
`--overwrite`. `--threads` controls CPU threads (default: 4).

Python entry point:

```python
from spark.predict import predict

files = predict("my_data", device="cpu", threads=4)
```

## GPU inference

GPU inference uses an NVIDIA GPU and a driver compatible with the CUDA 12.8
PyTorch build. **One GPU runs all eight ensemble members sequentially**;
`--device cuda` enables both model inference and spectrum reconstruction on
that GPU. Structure graph construction and file writing use the CPU.

### Install the GPU environment

From the repository root, create a separate GPU environment so PyTorch and the
compiled graph extensions all use the CUDA 12.8 build:

```bash
python3.11 -m venv .venv-gpu
source .venv-gpu/bin/activate
bash install.sh cu128
```

This installs PyTorch 2.8.0 with CUDA 12.8 and matching PyG extensions, following
the [PyTorch binary installation options](https://pytorch.org/get-started/previous-versions/)
and [PyG 2.7 installation guide](https://pytorch-geometric.readthedocs.io/en/2.7.0/install/installation.html).

Check GPU availability in the active environment with:

```bash
nvidia-smi
python - <<'PY'
import torch

print("PyTorch:", torch.__version__)
print("CUDA build:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
assert torch.cuda.is_available(), "Check the NVIDIA driver and active GPU environment."
print("GPU:", torch.cuda.get_device_name(0))
PY
```

### Run predictions and uncertainty

```bash
# Bundled CSIRO example: all eight models plus ensemble uncertainty.
spark demo --device cuda --output gpu_example

# Your own structures: all eight models plus ensemble uncertainty.
spark predict --root my_data --device cuda

# One selected structure, or one selected model.
spark predict --root my_data --ids particle --device cuda --overwrite
spark predict --root my_data --ids particle --member 42 --device cuda --overwrite
```

The demo writes to `gpu_example/xanes/`; your data writes to `my_data/xanes/`.
Output filenames and FEFF-reference protection follow the same rules as CPU
prediction. Add `--overwrite` to replace existing prediction files. Selecting
one member produces a single-model prediction; omitting `--member` computes
the eight-model uncertainty.

Python usage:

```python
from spark.predict import predict

files = predict("my_data", device="cuda", threads=4)
```

## Model

SPARK adapts the directional message-passing architecture of DimeNet++ to
atom-resolved spectra. Atomic-number embeddings, interatomic distances, and
angles describe local environments. Bessel radial and spherical angular basis
functions feed interaction blocks. Each output stage contributes to two heads:
one predicts spectral shape as energy samples, and the other predicts spectral
area. The outputs remain atom-resolved rather than being pooled into a single
graph-level vector.

| Setting | Publication model |
| --- | --- |
| Cutoff / maximum neighbors | 5.0 Å / 48 |
| Interaction blocks / hidden channels | 4 / 64 |
| Interaction / basis embedding size | 32 / 16 |
| Output embedding size | 256 |
| Spherical / radial basis functions | 10 / 12 |
| Samples per atom | 512 |
| Area output scale | 50 |
| Training dropout | 0.1; disabled during prediction |
| Periodic triplet filter threshold | 0.0001 Å |

The architecture, graph construction, and parameter tensors preserve the
research implementation. Periodic structures include image vectors in their
distances and angles. Finite XYZ graphs use the same radius-graph convention
used during training.

## How `samples` and `areas` represent a spectrum

The model predicts energy positions and one integral per atom. Its `samples`
are deterministic spectral quantiles, not randomly drawn ensemble members or
intensity values on an energy grid.

For target preparation, `spark.utils.inverse_transform`:

1. Interpolates a site's energy/intensity pairs to a uniform grid in the
   dataset loader (three times the original point count).
2. Clips negative intensity values to zero and computes the target area
   `A = trapezoid(mu, E)`.
3. Forms discrete probabilities `p_j = mu_j / sum(mu)` and their cumulative
   sum, then linearly inverts this CDF at the 512 fixed mid-bin quantiles
   `q_k = (k + 0.5) / 512`, for `k = 0, ..., 511`.
4. Scales each sampled energy as `s = (E - center) / scale` for training.

The code's discrete CDF convention and trapezoidal area are distinct
operations; use the supplied helpers when reproducing the representation.

| Element / edge | Center (eV) | Scale (eV) |
| --- | --- | --- |
| Co K | 7745 | 35 |
| Pt L3 | 11585 | 35 |

During prediction, the sample head produces 512 scaled energies per atom.
They are converted back with `E_k = s_k * scale + center`. The area head
returns `A = raw_area * 50`. The training area loss is
`((A_pred - A_target) / 50)^2`; this scale stabilizes optimization and is
already applied by the model during inference. Do not multiply the exported
`areas` by 50 again.

A continuous spectrum is reconstructed by Gaussian kernel density estimation:

```text
mu(E) = A / 512 * sum_k Normal(E; E_k, sigma_k)
```

The publication defaults use adaptive widths derived from adjacent sorted
sample spacings, with scale 1.0 and bounds **0.3–1.0 eV**. Sample positions and
widths use the same order as the research reconstruction implementation.
The 512-point output grid runs from the smallest predicted sample to the
largest predicted sample minus a **1 eV high-energy trim**. Every ensemble
member uses the same grid for a given atom. Each element-average output uses
a grid shared by all sites and models of that element. No additional tail
trim is applied to the element mean in this release.

`A` is the integral of the model's full Gaussian mixture; the integral over
the finite exported window may be smaller because of boundary clipping and
the tail trim. Reconstruction does not renormalize the full predicted
spectrum on that finite window. Shape-only uncertainty separately normalizes
the area-free reconstruction on the common finite grid.

In `<id>_samples.npz`, loaded with `numpy.load(..., allow_pickle=False)`:

| Array | Shape | Meaning |
| --- | --- | --- |
| `samples` | `(models, atoms, 512)` | Predicted energy positions in eV |
| `normalized_samples` | `(models, atoms, 512)` | Dimensionless sample-head outputs |
| `areas` | `(models, atoms)` | Predicted integrated intensities, already scaled |
| `atomic_numbers` | `(atoms,)` | Input atom identities |
| `init_seeds` | `(models,)` | Model order in the arrays |

## Ensemble uncertainty

All eight checkpoints are evaluated independently in `eval()` mode.
Reconstruct each member before computing the ensemble mean and spread:

```text
mean_mu(E) = (1/8) * sum_m mu_m(E)
std_mu(E)  = sqrt((1/8) * sum_m (mu_m(E) - mean_mu(E))^2)
unc_rms_std = sqrt(mean_E(std_mu(E)^2))
```

This uses the **population** standard deviation (`correction=0`, denominator
8), matching the research uncertainty definition. It measures variation
between independently initialized training trajectories. It is not a
calibrated confidence interval or experimental measurement noise estimate.
Small spread does not guarantee small prediction error outside the trained
chemical and structural domain.

`<id>_uncertainty.csv` includes these per-atom values:

| Column | Definition |
| --- | --- |
| `unc_mean_std` | Mean pointwise spectrum standard deviation |
| `unc_rms_std` | RMS pointwise spectrum standard deviation; primary metric |
| `unc_max_std` | Maximum pointwise spectrum standard deviation |
| `unc_integrated_var` | Trapezoidal integral of pointwise variance over energy |
| `area_mean`, `area_unc_std` | Mean and population spread of predicted areas |
| `area_rel_unc_std` | Area standard deviation divided by absolute mean area |
| `samples_unc_rms_std` | RMS spread of area-normalized spectral shapes |
| `combined_samples_area_unc` | Root-sum-square of shape spread and relative area spread |

`<id>_mean.dat` averages the sites of an element **within each model first**,
then takes the mean and standard deviation across those eight element means.
This includes correlations between sites. Its standard deviation is not the
average of the per-atom standard deviations. The combination metric above is
a heuristic descriptor, not a probabilistic error bound.

## Checkpoints and training provenance

`checkpoints/` contains eight separate **epoch-50 `last.pt` parameter sets**
from the final round-5 active-learning campaign. After acquiring labels, all
eight models were trained **from scratch** on the same cumulative training
split. Each file contains one final parameter set, its architecture, seed,
and epoch. There is no averaging of parameters between models, epochs, or
active-learning rounds.

- Training structures: 5,723 (4,748 original training structures plus 975
  CSIRO structures with 1,063 acquired site labels).
- Validation / test structures: 593 / 593, preserving the original splits.
- Final training: 50 epochs, batch size 4, AdamW, learning rate `1e-4` with
  cosine decay to `1e-6`, weight decay `1e-6`, sample MSE plus scaled area MSE.
- Only `init_seed` differs among the eight training configurations; training
  seed, architecture, full splits, and other training settings were checked.

`checkpoints/manifest.json` records SHA-256 hashes, split hashes, training
settings, and the successful consistency/export checks. The predictor verifies
the parameter-file hashes before loading them with `weights_only=True`.
Optimizer states, scheduler states, and training IDs are omitted from the
inference files.

The research audit used validation-selected `best.pt` checkpoints. This
publication package deliberately uses the **final epoch** requested for this
release. The historical audit's numerical coverage and maximum-uncertainty
results therefore must not be attributed to these `last.pt` weights without a
new full audit. `docs/export.py` provides the export recipe for the trusted
local research checkpoints. This repository distributes inference; the full
training dataset and active-learning execution environment are not bundled.

## Data and file formats

The bundled source example is in `example/structure/104795.xyz` and
`example/xanes/104795_feff_reference.dat`. It contains 105 atoms: 52 Pt and 53 Co.
The reference has FEFF energy/absorption pairs for **all 105 sites**, not an
element-average prediction. The example demonstrates execution and comparison;
it is not presented as an independent held-out benchmark.

Atom-resolved reference, prediction, and standard-deviation `.dat` files are
whitespace-separated tables with one header row:

```text
atom_0_energy atom_0_mu atom_1_energy atom_1_mu ...
```

Energy is in eV and `mu` uses the FEFF absorption convention of the training
data. In `_std.dat`, the `atom_*_mu` columns contain standard deviations. Use
each atom's own energy column: Pt and Co have different edge energies, and
individual sites may have different grids. Reference files may contain NaN
padding; filter finite pairs independently for each atom. To compare a FEFF
answer with a prediction, interpolate both onto their overlapping energy
range. Do not subtract row numbers from different energy grids directly.

Element averages have columns `Pt_energy Pt_mu Pt_std Co_energy Co_mu Co_std`
when both elements are present. A single-member result omits the `_std`
columns. The FEFF answer is kept under `_feff_reference.dat`; generated files
always use the suffixes documented above. Existing `<id>.dat` target files
are also left untouched.

For training-data compatibility, `StructureXANESDataset` in `spark/data.py`
recognizes paired `structure/<id>.xyz` or `.vasp` and `xanes/<id>.dat` files.
It converts the labeled atom columns to samples/areas and masks unlabeled
atoms. This loader is separate from inference, which reads no target files
and computes a spectrum for every supported input atom.

CSIRO source structures: Jonathan Ting, with contributors Amanda Barnard and
George Opletal, *PtCo Nanoparticle Data Set*, CSIRO,
[DOI: 10.25919/cg83-jv56](https://doi.org/10.25919/cg83-jv56),
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The selected structure
bytes and atom order are unchanged. SPARK's FEFF calculations supply the
separate reference spectra. See `example/README.md` for checksums and source
metadata. Full datasets are not included in this repository.

## Verification and citation

```bash
python -m pip install pytest
python -m pytest -q
spark demo --output /tmp/spark-demo
```

Tests check uncertainty with synthetic spectra, sample/area behavior, periodic
graphs, model consistency, and file protection. `docs/validation.md` records
the release validation results. Cite this version using `CITATION.cff` and
cite the CSIRO source data when using the example structure.

The code and model parameters use the repository's MIT license. The CSIRO
structure retains its separate CC BY 4.0 license. DimeNet++/PyTorch Geometric
attribution and its MIT notice are in `THIRD_PARTY_NOTICES.md`.

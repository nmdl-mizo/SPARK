# Release validation

Validated on 2026-10-15 for SPARK 1.0.0. The validation used CPU inference;
the CUDA installation path is provided but was not exercised for this release.

## Environment and installation

- Linux x86-64, Python 3.11.15.
- Clean virtual environment, installed with `bash install.sh cpu`.
- PyTorch 2.8.0+cpu, PyG 2.7.0, torch-cluster 1.6.3+pt28cpu,
  torch-sparse 0.6.18+pt28cpu, torch-scatter 2.1.2+pt28cpu.
- `python -m pip check`: no broken requirements.
- The installed package was executed from `/tmp`, outside the source checkout.
  Model and example resources resolved inside the installed `spark` package.

Package files came from PyPI, the PyTorch CPU index, and the PyG wheel index.

## Automated checks

`python -m pytest -q`: **10 passed** in the source checkout and in
the clean CPU package environment. Checks cover population uncertainty,
sample/area scaling, energy references, bounded reconstruction, periodic graph
construction, distinct ensemble members, input/output protection, element-mean
uncertainty with correlated sites, and removal of stale ensemble uncertainty
files when explicitly replacing results with a single-member prediction.

PyG emits one upstream distributed-module deprecation warning on import.

## End-to-end example

```bash
spark demo --output /tmp/spark-demo
```

- 105 sites: 52 Pt and 53 Co; all eight models executed.
- Atom-resolved mean prediction: 512 rows × 210 columns, all values finite.
- Saved samples: `(8, 105, 512)`; saved areas: `(8, 105)`.
- Scalar uncertainty table: 105 rows, each recording eight models.
- Mean / maximum `unc_rms_std`: `0.0085282811` / `0.0167673845`.
- FEFF reference SHA-256 preserved:
  `dd0200cb06e4186524afa8ffbf20b0a8d560260d689d725506a8f25235e23272`.
- Repeating the command without `--overwrite` correctly returned exit code 2
  and preserved the reference and previous prediction files.

These values describe the bundled functional example, not a held-out accuracy
estimate or the historical full-dataset convergence audit.

## Comparison with research code

The original `sparknet` model and graph code separately loaded all eight
trusted research `last.pt` files. Their unnormalized sample-head values and
scaled area-head values were compared with the installed publication package
for the 105-atom XYZ example. All values were **bitwise identical** across all
eight members (maximum absolute difference 0).

The research `batched_uncertainty_rows` routine separately reconstructed the
same predictions on the standard 512-point grids. Across all 105 atoms, the
maximum absolute differences were zero for:

```text
unc_mean_std
unc_rms_std
unc_max_std
unc_integrated_var
area_unc_std
samples_unc_rms_std
```

A generated four-site periodic Pt–Co FCC cell was also predicted with
`spark predict --root /tmp/spark-periodic --member 42`. Its sample and area
outputs were bitwise identical to the original periodic research path.
The single-member output contained predictions, element means, samples/areas,
and metadata, with no ensemble uncertainty files.

## Checkpoint and artifact checks

The export checks all training arguments except `init_seed` and run identifier,
all split contents, model configurations, and dataset configurations for
equality. Every member is a scratch-trained round-5 final model at epoch 50.
Every exported tensor was compared exactly with its source tensor. These
checks passed; their results and hashes are in `checkpoints/manifest.json`.

The built wheel contains exactly eight parameter files, one XYZ structure,
and one FEFF reference `.dat` file. It contains no generated prediction data,
research datasets, optimizer states, or training-run directories. Each
checkpoint is about 12.1 MB; the wheel is about 88.8 MB. Weights are regular
repository files, so Git LFS or a separate credentialed download is unnecessary.

The release uses final-epoch parameters. Historical coverage figures based on
validation-selected `best.pt` parameters are not claimed for this release.

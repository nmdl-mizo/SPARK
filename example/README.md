# CSIRO example

`structure/104795.xyz` is one original structure from the CSIRO **PtCo
Nanoparticle Data Set**, identified in its source comment as
`PtCo20DH50RCS3`. It contains 105 atoms: 52 Pt and 53 Co. Coordinates are in Å;
the input atom order and file bytes are unchanged.

Attribution: Jonathan Ting; contributors Amanda Barnard and George Opletal;
CSIRO, *PtCo Nanoparticle Data Set*,
https://doi.org/10.25919/cg83-jv56. The source collection record is retained in
`source.xml`. The source structure is licensed under
[Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/).
This distribution selects one structure from the collection and adds separate
FEFF spectra; it does not change the structure.

`xanes/104795_feff_reference.dat` contains the project's FEFF calculations for
all 105 absorbers, copied from the research workspace's `dataset_csiro105`
labels. It is an example reference answer. Its 210 columns are paired
`atom_<index>_energy atom_<index>_mu`, with zero-based indices matching the XYZ
file and energy in eV. The original table values are unchanged; the filename
identifies the file as a FEFF reference. No FEFF executable is redistributed.
The project-generated reference spectra are supplied under the repository's
MIT license; the separate CSIRO structure license still applies.

Run from a source checkout:

```bash
spark predict --root example
```

Or, after installing the Python package, run `spark demo` in any writable
directory. This copies the same structure and reference and executes the
models. Prediction only reads the structure. The full dataset is excluded.

SHA-256 checksums of the bundled inputs:

```text
a357690802a8c77e15b0e0abcee436e5fd40e4c79b3eb446d1945806288bd7f6  structure/104795.xyz
dd0200cb06e4186524afa8ffbf20b0a8d560260d689d725506a8f25235e23272  xanes/104795_feff_reference.dat
```

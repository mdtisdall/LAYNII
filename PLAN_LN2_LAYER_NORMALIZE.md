# Plan: LN2_LAYER_NORMALIZE

Planning file for branch `feature/layer-normalize`. That branch deletes this
file when the command is finished; the lasting content goes into the
program's `show_help()` text.

## Goal

A new program, `LN2_LAYER_NORMALIZE`, that takes the same kind of input as
`LN2_LAYER_SMOOTH` (an intensity image and a label image with layers or
columns). It does not smooth. It treats the intensities of all voxels that
share a label as one vector and divides each one by that vector's L2 norm,
so each label's vector has unit length:

    out_i = x_i / sqrt( sum_{j in label(i)} x_j^2 )

The main use case is a column file: the intensities inside each column are
scaled so the column has unit length.

## Decisions (confirmed)

| Topic | Decision |
|---|---|
| Vector scope | All voxels with the same label value. No FWHM/radius option. |
| `-NoKissing` | Optional. Each spatially connected piece (26-neighborhood) of a label is its own vector. |
| 4D input | Each volume (time point) is normalized on its own. |
| Voxels with label 0 | Copied unchanged. `-mask` sets them to 0 (same as `LN2_LAYER_SMOOTH`). |
| Name / output suffix | `LN2_LAYER_NORMALIZE`, suffix `layer_normalized`. |
| Zero-norm vector | Output 0 for that vector (no division by zero). Not tested. |

## CLI

    LN2_LAYER_NORMALIZE -input data.nii.gz -layer_file columns.nii.gz
        [-NoKissing] [-mask] [-output out.nii.gz]

Options: `-help`, `-input`, `-layer_file`, `-NoKissing`, `-mask`, `-output`.
`-FWHM` and `-twodim` from `LN2_LAYER_SMOOTH` are left out because they have
no meaning without smoothing.

## Implementation

New file `src/LN2_LAYER_NORMALIZE.cpp`. It reuses the structure and
conventions of `src/LN2_LAYER_SMOOTH.cpp` (argument parsing, `show_help`,
`log_welcome`, `log_nifti_descriptives`, `copy_nifti_as_float32` /
`copy_nifti_as_int32`, `save_output_nifti`). It does not copy the smoothing
loops.

1. Read input and layer file. Check that x/y/z dimensions match; error if not.
   Number of volumes `nt = nii_input->nt` (at least 1).
2. Make a "vector id" image (int32, one value per voxel, 0 = no vector):
   - Default: vector id = label value.
   - `-NoKissing`: a breadth-first flood fill over the 26-neighborhood
     gives each connected component of each label a new id 1..K. Only
     voxels with the same label join a component. This replaces the
     `hairy_brain` code of `LN2_LAYER_SMOOTH`, which is slow and has known
     problems (the `jy <= jz_stop` loop bound, a mask that is never reset,
     and a debug file that is always written).
3. For each volume t:
   - Accumulate `sum_sq[id] += x^2` in double precision (array of size K+1).
   - `norm[id] = sqrt(sum_sq[id])`.
   - For each voxel with id > 0: `out = norm > 0 ? x / norm : 0`.
   - For each voxel with id == 0: `out = -mask ? 0 : x`.
4. Save as float32 with `save_output_nifti(fout, "layer_normalized", ...)`.
   When `-output` is not given, the name comes from the input (same as
   `LN2_LAYER_SMOOTH`).

Use 64-bit indexing (`nr_voxels * nt` can be larger than `INT_MAX` for big
4D data).

## Build and documentation

- `Makefile`: add `LN2_LAYER_NORMALIZE` to the `LAYNII2` list and add a
  build rule, the same as `LN2_RECIPROCAL`.
- `test_data/tests.sh` and `test_data/tests_on_windows.sh`: add one line
  that runs the program (the current smoke-test convention):
  `../LN2_LAYER_NORMALIZE -input lo_BOLD_act.nii.gz -layer_file lo_columns.nii.gz`
- `show_help()` is the user documentation. Nothing else in `README.md`
  lists the LN2 programs.

## Correctness tests (minimal)

The repository has no correctness tests; `tests.sh` only checks that the
programs run. numpy/nibabel are not available, so the test script is
`test_data/test_LN2_LAYER_NORMALIZE.py`, which uses only the Python
standard library (`struct`, `gzip`) to write and read small NIfTI-1 files.
Python 3 is in the devShell today only by accident (it comes in with
codespell), so add `pkgs.python3` to `packages` in `flake.nix` to make it
explicit. `scripts/check` runs the script after the smoke tests, so CI runs
it too. The script runs the compiled binary on synthetic data:

1. **Per-label unit norm.** A 3D 6x6x6 image with two column labels and
   label-0 background, random non-zero intensities. Check: for each label,
   the sum of output^2 is 1 (tolerance 1e-5); each output voxel equals
   input / (norm computed in Python); background voxels are unchanged.
2. **`-mask`.** Same input with `-mask`: background voxels are 0.
3. **`-NoKissing`.** One label made of two blobs that do not touch. Without
   the flag, the two blobs together have norm 1. With the flag, each blob
   has norm 1 on its own.
4. **4D.** A 2-volume input where volume 2 = 3 x volume 1. Each volume is
   normalized on its own, so the two output volumes are equal and each has
   unit norm per label.

Run:

    nix develop --command make LN2_LAYER_NORMALIZE
    nix develop --command python3 test_data/test_LN2_LAYER_NORMALIZE.py ./LN2_LAYER_NORMALIZE

## Steps

1. Write `src/LN2_LAYER_NORMALIZE.cpp`; add the Makefile entries.
2. Build; run the smoke line on `lo_BOLD_act` / `lo_columns`.
3. Write the Python test script; make all four tests pass.
4. Add the smoke-test lines to `tests.sh` / `tests_on_windows.sh`; add
   `pkgs.python3` to `flake.nix`; call the test script from `scripts/check`.
5. `nix develop --command scripts/check` passes.
6. Delete this file in the same PR as the command; ship.

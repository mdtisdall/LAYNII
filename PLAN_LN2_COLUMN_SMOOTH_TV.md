# Plan: LN2_COLUMN_SMOOTH_TV

Planning file for branch `feature/column-smooth-tv`. That branch deletes this
file when the command is finished; the lasting content goes into the
program's `show_help()` text.

## Goal

A new program, `LN2_COLUMN_SMOOTH_TV`. It is related to `LN2_LAYER_SMOOTH`,
but it smooths **across** columns, not within them, and it uses an L2,1
(group total variation) penalty, not a Gaussian-weighted mean (which is the
L2 case).

A **cell** (k, j) is the set of voxels that have column label k in the
column file and layer label j in the layer file. Thus a column is a stack
of L cells, one for each layer, and each cell has a small number of voxels.

Each column k has a layer profile x_k ∈ R^L: entry j is the mean input in
cell (k, j). The distance between two columns is the L2
norm of the difference of their profiles. The program solves

    min_u  ½ Σ_{k,j} n_kj (u_kj − x_kj)²  +  λ Σ_{k<l} w_kl ||u_k − u_l||₂

- n_kj: number of voxels in cell (k, j).
- w_kl = the number of face-adjacent (6-neighbourhood) voxel pairs with one
  voxel in column k and one in column l. Only touching columns have an
  edge. The weight approximates the area of the shared boundary, so the
  penalty is a discrete form of continuous total variation (the coarea
  formula), and columns with different sizes or contact areas are handled
  correctly. λ is in units of intensity × voxels per face. Faces are
  counted, not measured in mm², so anisotropic voxels are not corrected for.

A neighbourhood width is not necessary. TV is a global optimization, so its
effect spreads across the whole graph through the iterations, and λ alone
sets the amount of smoothing. Touching edges also keep the degree small
(about 6–12), which gives large solver steps, and they cause fewer links
across a sulcus than a distance radius would.

The sum over edges of an L2 norm (not squared) is the L1 norm of the edge
norms. Thus neighbouring columns move to the same profile as a group, or
keep a step between them (edges are kept). This is different from a
per-layer L1 penalty, which would make the layers change independently.

There are two output rules. Both use the same solution u; only the
mapping back to the voxels is different.

- **Shift (default).** Each voxel gets the change of its cell:
  `voxel_out = voxel_in + (u_kj − x_kj)`. The variation between the voxels
  inside a cell is kept; only the cell mean moves. With this rule, the data
  term equals ½ ||voxel_out − voxel_in||² summed over voxels.
- **Replace (`-replace`).** Each voxel gets its cell's smoothed mean:
  `voxel_out = u_kj`. The output is constant in each cell. The voxel-level
  squared change is then n_kj (u_kj − x_kj)² + Σ (voxel_in − x_kj)², and
  the second term does not depend on u. Thus the same u is optimal for
  this rule too, and the solver is the same for the two rules.

## Decisions (confirmed)

| Topic | Decision |
|---|---|
| Formulation | Group total variation (L2,1) denoising over a column graph. Global optimization, not a local filter. |
| Column vector | Layer profile. Needs a column file and a layer file. |
| Neighbours / weights | Touching columns only. Weight = number of shared voxel faces (6-neighbourhood). No `-FWHM`. |
| Data term | Voxel-count weighted (n_kj). Empty cells (n = 0) have no data term; the TV term sets their value, and no voxel uses it. |
| Output | Default: shift each voxel by the change of its cell mean. `-replace`: set each voxel to the smoothed cell mean. |
| 4D input | Default: each volume is solved separately. `-joint4D`: the profile of each column has all volumes (length L × T), and one problem is solved for all of them. |
| Voxels outside columns or layers | Copied unchanged. `-mask` sets them to 0 (the same as `LN2_LAYER_SMOOTH`). |
| Name / output suffix | `LN2_COLUMN_SMOOTH_TV`, suffix `column_smoothed_TV`. |

## CLI

    LN2_COLUMN_SMOOTH_TV -input data.nii.gz -column_file columns.nii.gz
        -layer_file layers.nii.gz -lambda 10
        [-replace] [-joint4D] [-iter 500] [-mask] [-output out.nii.gz]

| Option | Meaning |
|---|---|
| `-input` | Data to smooth (3D or 4D). |
| `-column_file` | Column labels (int, 0 = not a column). |
| `-layer_file` | Layer labels (int, 0 = not a layer). Same x/y/z as `-input`. |
| `-lambda` | TV strength. 0 gives the input back. |
| `-replace` | (Optional) Set each voxel to its cell's smoothed mean. Default: shift each voxel by the change of the cell mean. |
| `-joint4D` | (Optional) One problem with all volumes, not one per volume. |
| `-iter` | (Optional) Maximum number of iterations. Default 500. |
| `-mask` | (Optional) Set voxels outside the column ∩ layer cells to 0. |
| `-output` | (Optional) Output file name. |

## Implementation

New file `src/LN2_COLUMN_SMOOTH_TV.cpp`. It uses the conventions of
`src/LN2_LAYER_SMOOTH.cpp` (argument parsing, `show_help`, `log_welcome`,
`log_nifti_descriptives`, `copy_nifti_as_float32` / `copy_nifti_as_int32`,
`save_output_nifti`). Use 64-bit indexing for the voxel × volume index.

1. **Read and check.** Read the three files. Error if x/y/z of the column or
   layer file is different from the input. K = max column label, L = max
   layer label, T = `nt` (at least 1).
2. **Cells.** For each voxel with column k > 0 and layer j > 0: add 1 to
   n_kj and add the value to sum_kj (one sum for each volume). Then
   x_kj = sum_kj / n_kj when n_kj > 0, else 0. Use double precision.
3. **Graph.** One pass over the voxels of the column file. For each voxel
   with column k > 0, look at its +x, +y and +z neighbours (each face is
   counted one time). If the neighbour has column l > 0 and l ≠ k, add 1 to
   the count of the pair (min(k,l), max(k,l)) in a
   `std::unordered_map<uint64_t, int>` (key = min·(K+1) + max). Then copy
   the map to an edge list (k, l, w_kl). This is O(voxels), whatever the
   number of columns. Only the column file is used here; a voxel's layer
   label does not matter for adjacency.
4. **Solver: Chambolle–Pock (primal-dual).** The variables are u (K × M,
   where M = L, or L × T with `-joint4D`) and one dual vector p_e ∈ R^M
   for each edge. D is the unweighted difference operator (Du)_e = u_k − u_l.
   The weights go into the dual constraint ||p_e|| ≤ λ w_e.
   - Step sizes: τ = σ = 0.99 / sqrt(2 · max_degree), because
     ||D||² ≤ 2 · max_degree.
   - Each iteration:
     - p_e ← proj_{||·|| ≤ λ w_e}(p_e + σ (D ū)_e) (scale p_e down to the
       ball radius when it is outside)
     - u_new = prox_τF(u − τ Dᵀp), where for each cell
       prox(v) = (v + τ n_kj x_kj) / (1 + τ n_kj). When n_kj = 0, this is
       u = v. Thus empty cells need no special case.
     - ū = 2·u_new − u; u = u_new.
   - Stop when max |u_new − u| < 1e-6 · max(1, max |x|), or at `-iter`.
     Write the number of iterations and the final change to the log.
   - Initialize u = x, p = 0.
   - Without `-joint4D`, run the solver once for each volume (M = L).
5. **Output.** float32 copy of the input. For each voxel in a cell (k, j)
   with n_kj > 0, at each volume t: out += u_kj − x_kj, or out = u_kj with
   `-replace` (use the volume's u and x, or the (j, t) entry with
   `-joint4D`). Other voxels: unchanged,
   or 0 with `-mask`. Save with
   `save_output_nifti(fout, "column_smoothed_TV", ...)`.

## Build and documentation

- `Makefile`: add `LN2_COLUMN_SMOOTH_TV` to the `LAYNII2` list and add a
  build rule, the same as `LN2_RECIPROCAL`.
- `test_data/tests.sh` and `test_data/tests_on_windows.sh`: add one smoke
  line:
  `../LN2_COLUMN_SMOOTH_TV -input lo_BOLD_act.nii.gz -column_file lo_columns.nii.gz -layer_file lo_layers.nii.gz -lambda 10`
  (Before this line is added, check that `lo_columns` and `lo_layers` are
  on the same grid and that the run takes only a few seconds.)
- `show_help()` is the user documentation: give the objective, the output
  rule, and the units of `-lambda` there.

## Correctness tests (minimal)

`test_data/test_LN2_COLUMN_SMOOTH_TV.py` uses only the Python standard
library (`struct`, `gzip`) to write and read small NIfTI-1 files, the same
method as the `LN2_LAYER_NORMALIZE` plan. If that branch merges first, use
its NIfTI helper and its `pkgs.python3` in `flake.nix`. If it does not,
this branch adds them. `scripts/check` runs the script after the smoke
tests.

All the tests use two columns of 2 layers, so the exact solution is known.
For two nodes with weights n_1 = n_2 = n in each cell, and difference
δ = x_1 − x_2:

- If λw ≥ n·||δ||/2: u_1 = u_2 = (x_1 + x_2)/2 (the columns join).
- Else: u_1 = x_1 − (λw/n)·δ/||δ||, u_2 = x_2 + (λw/n)·δ/||δ|| (group
  shrinkage along δ).

Geometry: a 4×2×2 image. Column 1 is x = 0–1 and column 2 is x = 2–3;
layer 1 is z = 0 and layer 2 is z = 1. Thus n = 4 in each cell, and the
columns share 4 faces at the x = 1 | x = 2 boundary, so w = 4. This also
checks the face count. Tolerance: 1e-4 (relative).

1. **λ = 0.** The output is equal to the input.
2. **Shrinkage (λ under the threshold).** The column profiles move toward
   each other along δ by the amount above. This checks the L2,1 group
   behaviour: both layers move by the same fraction, not each by the same
   amount as a per-layer L1 penalty would give. Also check that the
   variation of the voxels inside a cell does not change (shift output).
   Run the same input again with `-replace`: each voxel equals its cell's
   u, and u is the same as in the shift run.
3. **Join (λ over the threshold).** The two profiles become equal to their
   mean.
4. **`-joint4D`.** Two volumes, one layer, where volume 1 has a large δ
   and volume 2 has a small δ. Choose λ so that, per volume, volume 2
   joins and volume 1 does not. With `-joint4D`, neither volume joins and
   both shrink along the joint 2-vector δ. Compare both runs with the
   exact solutions.

Empty cells, columns with no neighbours, and `-mask` are not tested.

Run:

    nix develop --command make LN2_COLUMN_SMOOTH_TV
    nix develop --command python3 test_data/test_LN2_COLUMN_SMOOTH_TV.py ./LN2_COLUMN_SMOOTH_TV

## Steps

1. Write `src/LN2_COLUMN_SMOOTH_TV.cpp`; add the Makefile entries.
2. Build; run the smoke line on `lo_BOLD_act` / `lo_columns` / `lo_layers`.
3. Write the Python test script; make all four tests pass.
4. Add the smoke-test lines to `tests.sh` / `tests_on_windows.sh`; call the
   test script from `scripts/check` (and add `pkgs.python3` if it is not
   there yet).
5. `nix develop --command scripts/check` passes.
6. Delete this file in the same PR as the command; ship.

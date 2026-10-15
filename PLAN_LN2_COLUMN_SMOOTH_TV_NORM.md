# Plan: LN2_COLUMN_SMOOTH_TV_NORM

Planning file for branch `feature/column-smooth-tv-normalized`. That branch
deletes this file when the command is finished; the lasting content goes
into the program's `show_help()` text.

## Goal

A new program, `LN2_COLUMN_SMOOTH_TV_NORM`. It combines two commands:

- `LN2_COLUMN_SMOOTH_TV` (see `PLAN_LN2_COLUMN_SMOOTH_TV.md`): L2,1 (group
  total variation) smoothing of column layer profiles over the graph of
  touching columns.
- `LN2_COLUMN_NORMALIZE` (`src/LN2_COLUMN_NORMALIZE.cpp`): each column's
  layer profile (the vector of its cell means) is divided by its L2 norm.

The TV smoothing is applied with the condition that every column profile
has unit L2 norm at each solver iteration and in the output. Thus the
output is a fixed point of `LN2_COLUMN_NORMALIZE`.

The terms *cell*, *profile*, n_kj, w_kl, x_kj and the column graph are the
same as in `PLAN_LN2_COLUMN_SMOOTH_TV.md`.

### Problem

Normalize the input profiles first: x̂_k = x_k / ||x_k||, where the norm
uses only the observed cells (n_kj > 0), as in `LN2_COLUMN_NORMALIZE`.
Then solve

    min_u  ½ Σ_{k,j} n_kj (u_kj − x̂_kj)²  +  λ Σ_{k<l} w_kl ||u_k − u_l||₂
    s.t.   Σ_{j : n_kj > 0} u_kj² = 1   for each column k

This is TV on a product of unit spheres. The problem is not convex, so the
solver finds a stationary point, starting from u = x̂. Because the data are
unit vectors, λ does not depend on the intensity scale of the input. Its
unit is voxels per face.

Empty cells (n_kj = 0) are free TV variables, the same as in
`LN2_COLUMN_SMOOTH_TV`. They are not part of the norm, so a column's
coverage does not change its norm, and the output satisfies the
`LN2_COLUMN_NORMALIZE` definition (layers that a column does not reach
add 0).

### Output rules

- **Shift (default).** `voxel_out = voxel_in / ||x_k|| + (u_kj − x̂_kj)`.
  This is `LN2_COLUMN_NORMALIZE` followed by the TV change of the cell mean.
  The cell mean of the output is u_kj, so the output profile norm is 1.
- **Replace (`-replace`).** `voxel_out = u_kj`.

## Decisions (confirmed)

| Topic | Decision |
|---|---|
| Unit-norm enforcement | Exact prox of (data term + sphere constraint) in each iteration (see Solver). The alternative, a plain prox followed by a rescale, costs the same per iteration and converges at about the same rate, but its fixed point is biased when n_kj differs between the layers of a column. The output quality decides, so the exact prox is used. The two are identical when a column's cells have equal n_kj. |
| Empty cells | Free TV variables, not part of the norm. |
| 4D | Default: each volume is solved separately. `-joint4D`: one problem whose profile has all volumes (L × T), with a separate unit-norm constraint for each (column, volume). This is the per-volume rule of `LN2_COLUMN_NORMALIZE`. |
| `-NoKissing` | Kept. Each connected piece (26-neighbourhood) of a column is a separate graph node and is normalized separately (the same flood fill as `LN2_COLUMN_NORMALIZE`). Adjacency between two pieces of the same label counts as an edge. |
| Zero-norm columns | Output 0 (as in `LN2_COLUMN_NORMALIZE`); the column is not a graph node. Not tested. |
| Voxels outside cells | Copied unchanged; `-mask` sets them to 0. |
| Name / suffix | `LN2_COLUMN_SMOOTH_TV_NORM`, suffix `column_smoothed_TV_normalized`. |

## CLI

    LN2_COLUMN_SMOOTH_TV_NORM -input data.nii.gz -column_file columns.nii.gz
        -layer_file layers.nii.gz -lambda 2
        [-replace] [-joint4D] [-NoKissing] [-iter 500] [-mask] [-output out.nii.gz]

The options are the union of the options of the two parent commands, with
the same meaning.

## Dependency

The implementation copies the solver and the graph construction from
`src/LN2_COLUMN_SMOOTH_TV.cpp`, and the `-NoKissing` flood fill from
`src/LN2_COLUMN_NORMALIZE.cpp` (each command is one self-contained file).
Thus the implementation starts after `feature/column-smooth-tv` has merged.
This plan does not depend on it.

## Implementation

New file `src/LN2_COLUMN_SMOOTH_TV_NORM.cpp`. Steps 1–3 are the same as in
`LN2_COLUMN_SMOOTH_TV`. The node id comes from the column label, or from
the flood fill with `-NoKissing`.

1. **Read and check.** The same as `LN2_COLUMN_SMOOTH_TV`.
2. **Nodes and cells.** Node ids as in `LN2_COLUMN_NORMALIZE`. Then n_kj and
   x_kj for each node k.
3. **Graph.** The +x/+y/+z face count over the node-id image (not the raw
   column label, so that `-NoKissing` pieces are separate nodes).
4. **Normalize.** For each node k (and each volume t):
   ||x_k|| over the observed cells; x̂_k = x_k / ||x_k||. A node with
   norm 0 is marked inactive; its voxels get 0 and its edges are removed.
5. **Solver.** Chambolle–Pock, the same as `LN2_COLUMN_SMOOTH_TV` (the same
   τ, σ, dual projection, and ū extrapolation), with x̂ in place of x. Only
   the primal prox changes. For each node block (one volume) with
   v = u − τ Dᵀp:
   - Empty cells: u_j = v_j.
   - Observed cells: u_j(μ) = (v_j + τ n_j x̂_j) / (1 + τ n_j + μ). μ is the
     root of g(μ) = Σ_j u_j(μ)² − 1 on μ > −(1 + τ n_min), where g
     decreases monotonically. A safeguarded Newton solve (bisection
     fallback, up to 50 steps, tolerance 1e-12) finds μ. When all n_j in the
     block are equal, μ has a closed form: u = a / ||a|| with
     a_j = v_j + τ n x̂_j. Use this shortcut.
   - With `-joint4D`, apply this to each volume's L-block of the node
     separately.
   - The stopping rule and the log output are the same as in
     `LN2_COLUMN_SMOOTH_TV`, with max |x̂| = 1 in the tolerance.
6. **Output.** float32 copy of the input. For each voxel in an observed
   cell of an active node, apply the shift or replace rule above. Inactive
   nodes: 0. Other voxels: unchanged, or 0 with `-mask`. Save with
   `save_output_nifti(fout, "column_smoothed_TV_normalized", ...)`.

## Build and documentation

- `Makefile`: add `LN2_COLUMN_SMOOTH_TV_NORM` to the `LAYNII2` list and
  add a build rule.
- `test_data/tests.sh` and `test_data/tests_on_windows.sh`: one smoke line:
  `../LN2_COLUMN_SMOOTH_TV_NORM -input lo_BOLD_act.nii.gz -column_file lo_columns.nii.gz -layer_file lo_layers.nii.gz -lambda 2`
- `show_help()`: the objective, the unit-norm constraint, the output rules,
  and the units of `-lambda`.

## Correctness tests (minimal)

`test_data/test_LN2_COLUMN_SMOOTH_TV_NORM.py`. It uses the standard-library
NIfTI helpers of `test_data/test_LN2_COLUMN_NORMALIZE.py`. `scripts/check`
finds it automatically (`test_data/test_*.py`).

Geometry: the same as the `LN2_COLUMN_SMOOTH_TV` tests. A 4×2×2 image with
two columns (x = 0–1, x = 2–3) and two layers (z = 0, 1). Thus n = 4 in each
cell and w = 4.

Exact solution for two nodes. Let 2α be the angle between x̂_1 and x̂_2,
b = (x̂_1 + x̂_2) / ||x̂_1 + x̂_2|| the bisector, and
e = (x̂_1 − x̂_2) / ||x̂_1 − x̂_2||. By symmetry, u_{1,2} = cos θ · b ± sin θ · e.
The objective is 2n(1 − cos(α − θ)) + 2λw sin θ, so

- if λw ≥ n sin α: θ = 0 (the columns join at b);
- else: tan θ = (n sin α − λw) / (n cos α).

Tolerance: 1e-4.

1. **λ = 0.** The output is equal to the `LN2_COLUMN_NORMALIZE` rule
   (voxel / ||x_k||). Also, with `-NoKissing` on one column label that has two
   pieces with different scales, each piece is normalized separately.
2. **Shrink (λw < n sin α).** The cell means of the output equal u from the
   formula above, and each output profile has norm 1. In each cell, the
   variation between voxels is the input variation / ||x_k|| (shift). The
   same input with `-replace` gives voxel = u_kj.
3. **Join (λw ≥ n sin α).** Both profiles equal b.
4. **`-joint4D`.** Two volumes, with volume 2 = 3 × volume 1. After
   normalization the volumes are the same, and the joint problem is
   equal to the separate problem with λ/√2. Choose λ so that
   n sin α ≤ λw < √2 · n sin α. Without `-joint4D` the columns join in each
   volume; with `-joint4D` they do not, and they match the shrink formula
   with λ/√2. This also checks that the norm is computed per volume.

Not tested: empty cells, unequal n_kj in a column (the Newton path),
zero-norm columns, and `-mask`.

Run:

    nix develop --command make LN2_COLUMN_SMOOTH_TV_NORM
    nix develop --command python3 test_data/test_LN2_COLUMN_SMOOTH_TV_NORM.py ./LN2_COLUMN_SMOOTH_TV_NORM

## Steps

1. Wait until `feature/column-smooth-tv` has merged; rebase this branch on
   `origin/master`.
2. Write `src/LN2_COLUMN_SMOOTH_TV_NORM.cpp`; add the Makefile entries.
3. Build; run the smoke line.
4. Write the test script; make the four tests pass.
5. Add the smoke lines to `tests.sh` / `tests_on_windows.sh`.
6. `nix develop --command scripts/check` passes.
7. Delete this file in the same PR as the command; ship.

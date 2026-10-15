# Plan: LN2_COLUMN_SMOOTH_TV_NORM

Planning file for branch `feature/column-smooth-tv-normalized`. That branch
deletes this file when the command is finished; the lasting content goes
into the program's `show_help()` text.

## Goal

A new program, `LN2_COLUMN_SMOOTH_TV_NORM`. It combines two commands:

- `LN2_COLUMN_SMOOTH_TV` (`src/LN2_COLUMN_SMOOTH_TV.cpp`): L2,1 (group
  total variation) smoothing of column layer profiles over the graph of
  touching columns.
- `LN2_COLUMN_NORMALIZE` (`src/LN2_COLUMN_NORMALIZE.cpp`): each column's
  layer profile (the vector of its cell means) is divided by its L2 norm.

The TV smoothing is applied with the condition that every column profile
has unit L2 norm at each solver iteration and in the output. Thus the
output is a fixed point of `LN2_COLUMN_NORMALIZE`.

The terms *cell*, *profile*, n_kj, w_kl, x_kj and the column graph are the
same as in `LN2_COLUMN_SMOOTH_TV`. As in that program, cells without voxels
(n_kj = 0) are not part of the problem, and the difference of an edge
uses only the layers where both columns have voxels.

### Problem

Normalize the input profiles first: x̂_k = x_k / ||x_k||, over the cells
of column k that have voxels (the `LN2_COLUMN_NORMALIZE` rule). Then solve

    min_u  ½ Σ_{k,j} n_kj (u_kj − x̂_kj)²  +  λ Σ_{k<l} w_kl ||u_k − u_l||₂
    s.t.   Σ_{j : n_kj > 0} u_kj² = 1   for each column k

This is TV on a product of unit spheres. The problem is not convex, so the
solver finds a stationary point, starting from u = x̂. Because the data are
unit vectors, λ does not depend on the intensity scale of the input. Its
unit is voxels per face.

### Output rules

- **Shift (default).** `voxel_out = voxel_in / ||x_k|| + (u_kj − x̂_kj)`.
  This is `LN2_COLUMN_NORMALIZE` followed by the TV change of the cell mean.
  The cell mean of the output is u_kj, so the output profile norm is 1.
- **Replace (`-replace`).** `voxel_out = u_kj`.

## Decisions (confirmed)

| Topic | Decision |
|---|---|
| Unit-norm enforcement | Exact prox of (data term + sphere constraint) in each iteration (see Solver). The alternative, a plain prox followed by a rescale, costs the same per iteration and converges at about the same rate, but its fixed point is biased when the per-entry steps or n_kj differ in a column. The output quality decides, so the exact prox is used. |
| Empty cells | Not part of the problem or of the norm (the same as `LN2_COLUMN_SMOOTH_TV`). |
| 4D | Default: each volume is solved separately. `-joint4D`: one problem whose profile has all volumes (L × T), with a separate unit-norm constraint for each (column, volume). This is the per-volume rule of `LN2_COLUMN_NORMALIZE`. |
| `-NoKissing` | Kept. Each connected piece (26-neighbourhood) of a column is a separate graph node and is normalized separately (the same flood fill as `LN2_COLUMN_NORMALIZE`). Adjacency between two pieces of the same label counts as an edge. |
| Zero-norm columns | Output 0 (as in `LN2_COLUMN_NORMALIZE`); the column is not a graph node. Not tested. |
| Voxels outside cells | Copied unchanged; `-mask` sets them to 0. |
| Name / suffix | `LN2_COLUMN_SMOOTH_TV_NORM`, suffix `column_smoothed_TV_normalized`. |

## CLI

    LN2_COLUMN_SMOOTH_TV_NORM -input data.nii.gz -column_file columns.nii.gz
        -layer_file layers.nii.gz -lambda 2
        [-replace] [-joint4D] [-NoKissing] [-tol 1e-4] [-iter 100000]
        [-mask] [-output out.nii.gz]

The options are the union of the options of the two parent commands, with
the same meaning, except `-tol` (see Solver: the duality-gap bound of
`LN2_COLUMN_SMOOTH_TV` does not apply).

## Implementation

New file `src/LN2_COLUMN_SMOOTH_TV_NORM.cpp`. Copy `src/LN2_COLUMN_SMOOTH_TV.cpp`
and change it. Take the `-NoKissing` flood fill from
`src/LN2_COLUMN_NORMALIZE.cpp`.

1. **Read and check.** No change.
2. **Nodes and cells.** A node-id image: the column label, or the flood-fill
   piece id with `-NoKissing`. `nr_nodes`, `cell_of` and `cell_count` use
   the node id in place of the column label.
3. **Graph.** The +x/+y/+z face count over the node-id image (not the raw
   column label, so that `-NoKissing` pieces are separate nodes).
4. **Normalize.** After the cell means x are made (for each problem): for
   each node and volume, ||x|| over the entries with wt > 0, and
   x̂ = x / ||x||. A (node, volume) with norm 0 is inactive: its wt is set
   to 0 (so it leaves the problem and its edges are unused) and its voxels
   get 0.
5. **Solver.** `solve_tv` with x̂ in place of x. The preconditioned steps
   (τ_i = γ / degree_i, σ = 1 / (2γ)), the dual projection, the ū
   extrapolation and the adaptive γ do not change. Two parts change:
   - **Primal prox.** For each (node, volume) block, with
     v_i = u_i − τ_i (Dᵀp)_i over the entries with wt > 0:
     u_i(μ) = (v_i + τ_i n_i x̂_i) / (1 + τ_i (n_i + μ)).
     μ is the root of g(μ) = Σ_i u_i(μ)² − 1 on
     μ > −min_i (1/τ_i + n_i), where g decreases monotonically. A
     safeguarded Newton solve (bisection fallback, up to 50 steps,
     tolerance 1e-12) finds μ. When 1/τ_i + n_i is the same for all
     entries of the block, u = a / ||a|| with a_i = v_i + τ_i n_i x̂_i;
     use this shortcut. Entries with τ_i = 0 (no used edge) keep their
     value; the other entries of the block share the remaining norm,
     1 − Σ(kept entries)².
   - **Stop test.** The duality gap is not a valid bound for a problem
     that is not convex. Stop when the RMS primal residual and the RMS
     dual residual (the ones that the adaptive γ already computes) are
     both ≤ tol × scale, where scale is the n-weighted RMS of x̂.
     Default `-tol 1e-4`. The log gives the iterations and the final
     residuals, and warns if `-iter` stops the solver first.
6. **Output.** For each voxel in a cell of an active node, apply the shift
   or replace rule above. Inactive nodes: 0. Other voxels: unchanged, or 0
   with `-mask`. Save with
   `save_output_nifti(fout, "column_smoothed_TV_normalized", ...)`.

## Build and documentation

- `Makefile`: add `LN2_COLUMN_SMOOTH_TV_NORM` to the `LAYNII2` list after
  `LN2_COLUMN_SMOOTH_TV`, and add a build rule.
- `test_data/tests.sh` and `test_data/tests_on_windows.sh`: one smoke line
  after the `LN2_COLUMN_SMOOTH_TV` line:
  `../LN2_COLUMN_SMOOTH_TV_NORM -input lo_BOLD_act.nii.gz -column_file lo_columns.nii.gz -layer_file lo_layers.nii.gz -lambda 2`
- `show_help()`: the objective, the unit-norm constraint, the output rules,
  the units of `-lambda`, and what `-tol` means.

## Correctness tests (minimal)

`test_data/test_LN2_COLUMN_SMOOTH_TV_NORM.py`. Copy the NIfTI helpers, the
geometry (`SHAPE`, `COLUMNS`, `W`, `OFFSET`, `coords`) and `two_layer_data`
from `test_data/test_LN2_COLUMN_SMOOTH_TV.py`. `run` passes a tight `-tol`
(1e-8). `scripts/check` finds the script automatically
(`test_data/test_*.py`).

Geometry: two columns (x = 0–1, x = 2–3) and two layers (z = 0, 1), so
n = 4 in each cell and w = 4.

Exact solution for two nodes. Let 2α be the angle between x̂_1 and x̂_2,
b = (x̂_1 + x̂_2) / ||x̂_1 + x̂_2|| the bisector, and
e = (x̂_1 − x̂_2) / ||x̂_1 − x̂_2||. By symmetry, u_{1,2} = cos θ · b ± sin θ · e.
The objective is 2n(1 − cos(α − θ)) + 2λw sin θ, so

- if λw ≥ n sin α: θ = 0 (the columns join at b);
- else: tan θ = (n sin α − λw) / (n cos α).

(Checked against a brute-force minimization over θ.) Tolerance: 1e-4.

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

Not tested: empty cells (tested in `LN2_COLUMN_SMOOTH_TV`), unequal steps
in a block (the Newton path), zero-norm columns, and `-mask`.

Run:

    nix develop --command make LN2_COLUMN_SMOOTH_TV_NORM
    nix develop --command python3 test_data/test_LN2_COLUMN_SMOOTH_TV_NORM.py ./LN2_COLUMN_SMOOTH_TV_NORM

## Steps

1. Write `src/LN2_COLUMN_SMOOTH_TV_NORM.cpp`; add the Makefile entries.
2. Build; run the smoke line.
3. Write the test script; make the four tests pass.
4. Add the smoke lines to `tests.sh` / `tests_on_windows.sh`.
5. `nix develop --command scripts/check` passes.
6. Delete this file in the same PR as the command; ship.

# Plan: laminar refinement filter (working name `layer_refine`)

Planning file for branch `docs/layer-refine-plan`. The first version is a
Python prototype in the ex vivo example project (`example/`, a separate
repository with its own flake). A LayNii C++ port (working name
`LN2_LAYER_REFINE`) follows when the method and its weights are settled; the
port gets its own plan, and this file is deleted then. The lasting content
goes into the program's `show_help()` text.

## Goal

A filter that refines geometric layers with the observed laminar contrast,
for 100 µm ex vivo data. It balances three pressures:

1. **Within-layer total variation.** TV between neighbouring columns, inside
   each layer, on the filtered intensity and on the layer boundary depths.
2. **Bottom layer.** For each region, the deepest depth (into the
   `-extend_wm` layers: u-fibres, superficial WM) down to which the region
   keeps both its areal borders and its laminar transitions.
3. **Nudged layers.** Each column's layer boundaries move so that layers are
   homogeneous inside and distinct from their neighbours, pooled over each
   region.

The regions are estimated together with the layers, by merging columns.
Laminar contrast is normalized per column, over small units, whatever the
size of the region.

## Data and terms

Input, per patch (V1V2, PCC_SPL, M1S1 of `pipeline_L25_L13_wm`):

- FLASH crop f.
- Columns c (`LN2_COLUMNS -extend_wm 0.5`, about 0.7 mm apart). Column
  graph: columns that share voxel faces; w_cc' = number of shared faces (as
  in `LN2_COLUMN_SMOOTH_TV`).
- Continuous depth m(v): the equivolume metric of `LN2_LAYERS`, in [0, 1]
  in the cortex (0 = WM border) and −x/T in the extended WM. The pipeline
  does not crop it yet (step 0 below).
- Geometric layers: the 25- or 13-layer equivolume bins.

Measured on the patches (columns at least 2 voxels from the crop faces):

| | V1V2 | PCC_SPL | M1S1 |
|---|---|---|---|
| Cortex layers present, 25 bins (median) | 22 | 22 | 22 |
| Columns with an interior gap, 25 / 13 bins | 2.6 % / 0.1 % | 1.8 % / 0 % | 1.7 % / 0 % |
| Columns with no WM layer | 27 % | 12 % | 10 % |
| Voxels per cell, 25 bins (median, 10th pct) | 16, 6 | 16, 6 | 18, 7 |

Layers are missing almost only at the ends. Layer 1 (25 bins) is less than
one voxel thick (V1V2: about 2,000 voxels, against 37,000 in layer 2), layer
25 is empty and layer 24 is sparse (the pial cut, equidistant d ≤ 21/23).

Terms:

- *Region* R: a connected set of columns. *Template* μ_R: the region's
  layer profile (one value for each layer), zero mean and unit norm over the
  region's common layers.
- *Common layers* of a region: the layers that at least `p_common` (default
  0.8) of its columns have.
- *Boundaries* z_c = (z_c,1 < … < z_c,K−1): the depths, on the m axis, that
  separate the K layers of column c. Layer of voxel v: k_c(v) = the number of
  boundaries below m(v), plus 1.

## Model

For voxel v in column c of region R:

    f(v) = a_c + s_c · μ_R(k_c(v)) + noise

a_c (offset) and s_c (gain) normalize column c: they remove its intensity
level and contrast, so only the profile **shape** separates regions. They
form a smooth field over the column graph, which is also smooth across
region borders (a bias field does not stop at an areal border).

Energy, for a fixed partition into regions:

    E = Σ_c Σ_{v ∈ c} (f(v) − a_c − s_c μ_R(k_c(v)))² / σ_c²           data (pressure 3)
      + κ Σ_{c~c'} w_cc' [(a_c − a_c')² + (s_c − s_c')²] / σ²             gain/offset field
      + λ_z Σ_k Σ_{c~c'} w_cc' |z_c,k − z_c',k|                           TV on boundaries (pressure 1)
      + η Σ_c Σ_k (z_c,k − z⁰_c,k)²                                       pull to geometry
      s.t. z_c,k+1 − z_c,k ≥ t_min · (z⁰_c,k+1 − z⁰_c,k)                  minimum thickness

σ_c is the noise level of column c, from the variance inside its cells; it
makes the data term dimensionless. z⁰ are the geometric (equivolume)
boundaries. Minimizing the within-layer residual is the same as maximizing
the between-layer / within-layer (Fisher, Otsu) ratio for the fixed total
variance, so this is pressure 3. It is scored on the raw data, not on the
filtered u, so the TV on u cannot make the layers look homogeneous.

TV on the intensity (pressure 1) is a separate step on the normalized cell
means x̂_c,k = (cell mean − a_c) / s_c in the nudged layers:

    min_u  ½ Σ_c,k n_c,k (u_c,k − x̂_c,k)²
         + λ_g Σ_{c~c'} w_cc' ||u_c − u_c'||₂          group term (as LN2_COLUMN_SMOOTH_TV_NORM)
         + λ_l Σ_k Σ_{c~c'} w_cc' |u_c,k − u_c',k|      per-layer term

over the cells where both columns have voxels. λ_l = 0 is the group form;
λ_l > 0 the group + per-layer form. The two are compared (see Evaluation).

## Solver

### 0. Data preparation (pipeline change)

- Crop `metric_equivol` (float32) for each patch, next to the bins.
- Merge sparse edge layers: a geometric layer with fewer voxels than
  `f_sparse` (default 0.25) × the median layer count in the patch is merged
  into its inner neighbour. For 25 bins this merges 1 into 2 and 24, 25
  into 23; for 13 bins, 13 into 12.

### 1. Regions: agglomerative merging

The merge loop is the outer loop; it gives the full hierarchy.

1. Start: each column is a region.
2. Merge cost: the increase of the data term when two touching regions share
   one template (Ward linkage on the normalized column profiles, with the
   column graph as connectivity), using the current gain/offset and
   boundaries.
3. Merge the cheapest pair; update the costs of its neighbours.
4. **Re-fit levels.** A full re-fit (steps 2–4 below) after each merge is too
   expensive. The re-fit runs at the levels where the number of regions
   reaches N₀ / 2^j (and at 2, 3, 4, 5 regions), and the merge costs are then
   recomputed. Before the first re-fit, the boundaries are geometric and the
   gain/offset is the per-column mean and RMS.
5. Stop at one region. Nothing chooses a level: the output is the merge tree
   plus the full results at each re-fit level.

### 2. Gain/offset field (fixed regions, templates, boundaries)

Quadratic: a sparse linear system in 2 unknowns per column (data term plus
graph Laplacian), solved with conjugate gradients. κ = 0 gives the
independent per-column least squares.

### 3. Templates (fixed gain/offset, boundaries)

μ_R,k = Σ s_c (f(v) − a_c) / Σ s_c², over the voxels of region R in layer k.
Then centre and scale μ_R to zero mean and unit norm over the common layers,
and change a_c and s_c to match, so the model is unchanged.

### 4. Boundaries: optimal surfaces (fixed gain/offset, templates)

Exact min-cut for the K−1 surfaces of all columns together (Li, Wu, Chen &
Sonka 2006; Ishikawa 2003). The data term splits into one term for each
surface: with e_k(v) = (f(v) − a_c − s_c μ_R,k)² / σ_c²,

    Σ_v e_{k_c(v)}(v) = Σ_v e_1(v) + Σ_k Σ_{v : m(v) ≥ z_c,k} (e_k+1(v) − e_k(v)),

so the cost of surface k at depth d in column c is a cumulative sum over the
column's voxels sorted by depth. The prior η(z − z⁰)² is a unary term;
TV on z is a convex pairwise term between columns; the minimum thickness is
a hard constraint between neighbouring surfaces of one column. All of these
fit the construction, so the cut gives the global optimum of this step.

TV edges between columns of different regions have weight ρ λ_z (default
ρ = 0), so boundaries can jump at a region border.

**Coarse to fine.** The graph has columns × depth samples × surfaces nodes
(V1V2 at full resolution: about 4,500 × 250 × 24 = 27 million). First solve
with the 13-layer bins and a depth step of 1/50 of the thickness; then
solve the full layer count in a band of ± one coarse layer around the
coarse result, with a step of 1/200.

### 5. Bottom (pressure 2), for each region at each re-fit level

The profiles are extended into the WM layers −1, −2, … (normalized with
the column's own a_c, s_c). For each candidate bottom b (the WM layers down
to b included):

- **Test A, areal border.** For each neighbouring region R', over the
  columns within `D` (default 2 column widths) of the R–R' border: the
  separation of the two groups of profiles (between-group / within-group
  variance). It fails when it drops by more than `δ_A` (default 0.1) from its
  value with no WM layers.
- **Test B, laminar transitions.** The fraction of variance of the region's
  columns that the extended template explains. It fails when it drops by
  more than `δ_B` (default 0.1) from its value with no WM layers.

b_R is the deepest b for which every layer from 0 to b passes both tests.
Only columns that have WM layers down to b vote; columns without WM layers
take their region's b_R.

### 6. Intensity TV (for each output level)

Chambolle–Pock with diagonal preconditioning and adaptive steps, as in
`LN2_COLUMN_SMOOTH_TV_NORM`, with the per-layer term added (its prox is
soft thresholding of each entry of the dual). u does not feed back into
steps 1–5.

## Weights (normalized units)

| Weight | Unit | Default (first guess) |
|---|---|---|
| λ_g, λ_l (TV on u) | voxels per shared face, on normalized profiles (as TV_NORM) | sweep |
| λ_z (TV on z) | per shared face, z in fractions of local thickness | sweep |
| η (pull to geometry) | per column, z in fractions of local thickness | sweep |
| κ (gain/offset smoothness) | per shared face, a and s relative to σ | sweep |
| t_min | fraction of the geometric layer thickness | 0.3 |
| δ_A, δ_B | relative drop | 0.1 |
| p_common, f_sparse | fractions | 0.8, 0.25 |

## Outputs (per patch)

- Merge tree (parent, cost, size for each merge), and per re-fit level:
  - region labels (column and voxel images);
  - nudged layer labels and the continuous boundaries z_c (one value per
    column and boundary, CSV, and a nudged layer image);
  - gain and offset per column;
  - bottom b_R per region, and a bottom-depth image;
  - filtered image: each voxel shifted by the change of its cell mean, as
    the TV_NORM default (u in normalized units).
- A JSON with parameters, energies per term and solver statistics.

## Evaluation

For λ_l / λ_g ∈ {0, 0.1, 0.3, 1} over a λ_g sweep, at the 2-region level and
at the level nearest to the earlier k = 2 split:

- The measures of `analyze_dylan_laynii_filters_dancoordinates.py`:
  roughness, contrast, isolated, border, dist_dan.
- Laminar measures: the pooled within-layer residual fraction per region
  (data term / total variance), and, in V1V2, whether the stria of Gennari
  stays inside its layers: the width (in layers) of the stria's dip in the
  V1 template and in u, against the unfiltered profile.
- Whether the nudge helps: the same measures with η → ∞ (geometric layers).

## Correctness tests (prototype)

1. Optimal surfaces against brute force on a tiny problem (3 columns,
   2 surfaces, 10 depths).
2. Gain/offset with κ = 0 equals per-column least squares; with κ → ∞ it
   equals one global gain and offset.
3. Intensity TV: λ = 0 gives x̂; λ_g large gives the mean profile over each
   connected component; λ_l = 0 equals `LN2_COLUMN_SMOOTH_TV_NORM`'s problem
   (before its normalization).
4. Merging with re-fits off and equal column weights equals scikit-learn
   Ward clustering with the same connectivity.
5. A column copied with a gain and an offset gets the same boundaries and
   template.

## Prototype layout and dependencies

In `example/`:

- `layer_refine.py`: the filter (library and command line);
- `layer_refine_eval.py`: the evaluation sweep;
- `test_layer_refine.py`: the tests above;
- `run_pipeline.py`: step 0 (metric crop).

Dependencies: numpy, scipy, nibabel, scikit-learn (in nixpkgs), PyMaxflow
(Boykov–Kolmogorov min-cut; **not in nixpkgs**: package it from PyPI in the
example flake).

## Decided

| Topic | Decision |
|---|---|
| Pressure 2 boundaries | Both the areal borders and the laminar transitions. |
| Regions | Estimated jointly, by merging columns; the full hierarchy is the output. |
| Normalization | Smooth per-column gain/offset field against the region template; shape only. |
| Nudge | Per column, with TV smoothness across columns and a pull to geometry. |
| TV | On the intensity and on the boundary depths. Intensity TV: compare group norm against group + per-layer. |
| Bottom tests | Relative drop. |
| Size | Coarse to fine. |
| Edge layers | Merged into their neighbours. |
| Weights | Normalized units. |
| Evaluation | Existing measures plus laminar ones. |
| Implementation | Python prototype in `example/`; LayNii C++ port when settled. |

## Choices made in this plan (not yet confirmed)

- u does not feed back into the regions, boundaries or bottom.
- Re-fit only at the levels N₀ / 2^j and 2–5 regions; Ward costs in between.
- TV on z is cut at region borders (ρ = 0); the gain/offset field is not.
- The tests A and B and their defaults (D, δ_A, δ_B).
- The coarse-to-fine schedule and depth steps.
- The sparse-layer rule (f_sparse) and the common-layer rule (p_common).

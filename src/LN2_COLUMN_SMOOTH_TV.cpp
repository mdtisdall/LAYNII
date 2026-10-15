#include "../dep/laynii_lib.h"
#include <unordered_map>

int show_help(void) {
    printf(
    "LN2_COLUMN_SMOOTH_TV : Smooth layer profiles across columns with total\n"
    "                       variation (L2,1 norm).\n"
    "\n"
    "    For each column, this program averages the voxels in each layer of\n"
    "    the column (one 'cell' for each column and layer). The cell means\n"
    "    of a column make its layer profile x_k, a vector with one value for\n"
    "    each layer. The program smooths across columns, not within them:\n"
    "    it finds the profiles u that minimize\n"
    "\n"
    "        1/2 sum_kj n_kj (u_kj - x_kj)^2 + lambda sum_kl w_kl ||u_k - u_l||\n"
    "\n"
    "    where n_kj is the number of voxels in cell (k, j), w_kl is the\n"
    "    number of voxel faces that columns k and l share (only touching\n"
    "    columns are neighbours), and ||.|| is the L2 norm of the difference\n"
    "    of two profiles over the layers where both columns have voxels.\n"
    "    Cells without voxels are not part of the problem. Because the norm is not squared, neighbouring\n"
    "    columns either move to the same profile as a group or keep a step\n"
    "    between them, so edges are kept. lambda is in units of intensity x\n"
    "    voxels per shared face; lambda = 0 gives the input back.\n"
    "\n"
    "    By default, each voxel is shifted by the change of its cell mean,\n"
    "    out = x + (u_kj - x_kj), so the detail inside each cell is kept and\n"
    "    the data term equals 1/2 ||out - in||^2 over the voxels. With\n"
    "    -replace, each voxel is set to u_kj. The same u is optimal for both.\n"
    "\n"
    "    Only voxels with both a column and a layer label are smoothed.\n"
    "\n"
    "Usage:\n"
    "    LN2_COLUMN_SMOOTH_TV -input activity_map.nii -column_file columns.nii -layer_file layers.nii -lambda 10\n"
    "    ../LN2_COLUMN_SMOOTH_TV -input lo_BOLD_act.nii.gz -column_file lo_columns.nii.gz -layer_file lo_layers.nii.gz -lambda 10\n"
    "\n"
    "Options:\n"
    "    -help        : Show this help.\n"
    "    -input       : Nifti (.nii) file that should be smoothed. This\n"
    "                   can be a 4D nifti.\n"
    "    -column_file : Nifti (.nii) file that contains column labels\n"
    "                   (0 = not a column). Same x, y, z dimensions as the\n"
    "                   input.\n"
    "    -layer_file  : Nifti (.nii) file that contains layer labels\n"
    "                   (0 = not a layer). Same x, y, z dimensions as the\n"
    "                   input.\n"
    "    -lambda      : Strength of the total variation term.\n"
    "    -replace     : (Optional) Set each voxel to its smoothed cell mean,\n"
    "                   in place of shifting it by the change of the mean.\n"
    "    -joint4D     : (Optional) For 4D input, make each profile from all\n"
    "                   volumes (layers x volumes) and solve one problem.\n"
    "                   Without this option, each volume is smoothed\n"
    "                   separately.\n"
    "    -tol         : (Optional) Accuracy. The solver stops when the RMS,\n"
    "                   over the smoothed voxels, of the difference from\n"
    "                   the exact optimal output is at most tol x scale,\n"
    "                   where scale is the RMS of the cell means over the\n"
    "                   voxels. The bound is guaranteed (it comes from the\n"
    "                   duality gap). Single small cells can differ by more\n"
    "                   than this. Default 0.01.\n"
    "    -iter        : (Optional) Maximum number of iterations. Default\n"
    "                   100000. The program warns when it stops here before\n"
    "                   it reaches -tol.\n"
    "    -mask        : (Optional) Set voxels outside of the columns or\n"
    "                   layers to 0. Without this option they are copied\n"
    "                   unchanged.\n"
    "    -output      : (Optional) Output filename, including .nii or\n"
    "                   .nii.gz, and path if needed. Overwrites existing files.\n"
    "\n");
    return 0;
}

// Minimize P(u) = 1/2 sum_i wt_i (u_i - x_i)^2 + lambda sum_e w_e ||(Du)_e||
// with Chambolle-Pock. u, x and wt have nr_nodes * M entries (node-major).
// Entries with wt = 0 (empty cells) are not part of the problem, and
// (Du)_e has only the entries m where both columns of edge e have data.
// u is the start value and the result. Returns the number of iterations.
//
// The steps are diagonally preconditioned (Pock & Chambolle 2011): entry i
// has primal step gamma / degree(i) and every edge entry has dual step
// 1 / (2 gamma). gamma is adapted to balance the primal and dual residuals
// (Goldstein et al. 2015, "Adaptive primal-dual splitting methods for
// statistical learning and image processing").
//
// Stop test: P is strongly convex with modulus wt, so for the optimum u*,
// sum_i wt_i (u_i - u*_i)^2 <= 2 (P(u) - P(u*)) <= 2 gap, where
// gap = P(u) - D(p) for the dual objective
// D(p) = <Dx, p> - 1/2 sum_i (D^T p)_i^2 / wt_i. Each voxel's output
// changes by u_i - u*_i of its cell, so sqrt(2 gap / sum_i wt_i) bounds the
// RMS error of the output over the voxels. The solver stops when that bound
// is <= max_rms_error, and gives the bound in *error_bound.
static int solve_tv(vector<double>& u, const vector<double>& x,
                    const vector<double>& wt, int64_t nr_nodes, int64_t M,
                    const vector<int32_t>& edge_k, const vector<int32_t>& edge_l,
                    const vector<double>& edge_w, double lambda, int max_iter,
                    double max_rms_error, double* error_bound) {
    const int64_t nr_edges = edge_k.size();
    const int64_t nr_entries = u.size();
    *error_bound = 0;
    if (nr_edges == 0 || lambda <= 0) return 0;

    // Edge entry (e, m) is used when both of its cells have data
    vector<char> used(nr_edges * M);
    for (int64_t e = 0; e < nr_edges; ++e) {
        for (int64_t m = 0; m < M; ++m) {
            used[e * M + m] = wt[edge_k[e] * M + m] > 0 && wt[edge_l[e] * M + m] > 0;
        }
    }

    vector<double> inv_degree(nr_entries, 0.0);
    for (int64_t e = 0; e < nr_edges; ++e) {
        for (int64_t m = 0; m < M; ++m) {
            if (!used[e * M + m]) continue;
            inv_degree[edge_k[e] * M + m]++;
            inv_degree[edge_l[e] * M + m]++;
        }
    }
    double sum_wt = 0;
    for (int64_t i = 0; i < nr_entries; ++i) {
        if (inv_degree[i] > 0) inv_degree[i] = 1 / inv_degree[i];
        sum_wt += wt[i];
    }

    // D^T q for a vector q with one entry for each edge entry
    auto apply_dt = [&](const vector<double>& q, vector<double>& out) {
        fill(out.begin(), out.end(), 0.0);
        for (int64_t e = 0; e < nr_edges; ++e) {
            for (int64_t m = 0; m < M; ++m) {
                out[edge_k[e] * M + m] += q[e * M + m];
                out[edge_l[e] * M + m] -= q[e * M + m];
            }
        }
    };
    auto primal_objective = [&](const vector<double>& v) {
        double data = 0, tv = 0;
        for (int64_t i = 0; i < nr_entries; ++i) {
            data += wt[i] * (v[i] - x[i]) * (v[i] - x[i]);
        }
        for (int64_t e = 0; e < nr_edges; ++e) {
            double norm2 = 0;
            for (int64_t m = 0; m < M; ++m) {
                if (!used[e * M + m]) continue;
                double d = v[edge_k[e] * M + m] - v[edge_l[e] * M + m];
                norm2 += d * d;
            }
            tv += edge_w[e] * sqrt(norm2);
        }
        return 0.5 * data + lambda * tv;
    };

    double gamma = 1, alpha = 0.5;
    const double eta = 0.95, delta = 1.5;
    const int check_every = 50;
    vector<double> p(nr_edges * M, 0.0), dp(nr_edges * M);
    vector<double> u_bar(nr_entries), du(nr_entries), dtp(nr_entries);
    vector<double> dtdp(nr_entries), u_dual(nr_entries);
    int iter = 0;
    while (iter < max_iter) {
        iter++;
        const double sigma = 0.5 / gamma;

        // Primal step: u <- prox of the data term at u - tau * D^T p
        apply_dt(p, dtp);
        for (int64_t i = 0; i < nr_entries; ++i) {
            double tau_i = gamma * inv_degree[i];
            double v = u[i] - tau_i * dtp[i];
            double u_new = (v + tau_i * wt[i] * x[i]) / (1 + tau_i * wt[i]);
            du[i] = u[i] - u_new;
            u_bar[i] = 2 * u_new - u[i];
            u[i] = u_new;
        }

        // Dual step: p_e <- projection onto the ball of radius lambda * w_e
        double dual_res2 = 0;
        for (int64_t e = 0; e < nr_edges; ++e) {
            double* p_e = &p[e * M];
            double* dp_e = &dp[e * M];
            const double* uk = &u_bar[edge_k[e] * M];
            const double* ul = &u_bar[edge_l[e] * M];
            double norm2 = 0;
            for (int64_t m = 0; m < M; ++m) {
                dp_e[m] = p_e[m];
                if (!used[e * M + m]) continue;
                p_e[m] += sigma * (uk[m] - ul[m]);
                norm2 += p_e[m] * p_e[m];
            }
            double radius = lambda * edge_w[e];
            double scale = norm2 > radius * radius ? radius / sqrt(norm2) : 1;
            for (int64_t m = 0; m < M; ++m) {
                p_e[m] *= scale;
                dp_e[m] -= p_e[m];
                double r = dp_e[m] / sigma
                           - (du[edge_k[e] * M + m] - du[edge_l[e] * M + m]);
                if (used[e * M + m]) dual_res2 += r * r;
            }
        }

        // Stop test with the duality gap
        if (iter % check_every == 0 || iter == max_iter) {
            apply_dt(p, dtp);
            double dual_obj = 0;
            for (int64_t e = 0; e < nr_edges; ++e) {
                for (int64_t m = 0; m < M; ++m) {
                    dual_obj += p[e * M + m]
                                * (x[edge_k[e] * M + m] - x[edge_l[e] * M + m]);
                }
            }
            for (int64_t i = 0; i < nr_entries; ++i) {
                if (wt[i] == 0) {
                    u_dual[i] = u[i];
                    continue;
                }
                dual_obj -= 0.5 * dtp[i] * dtp[i] / wt[i];
                u_dual[i] = x[i] - dtp[i] / wt[i];  // Primal point of p
            }
            double primal_u = primal_objective(u);
            double primal_dual = primal_objective(u_dual);
            double gap = min(primal_u, primal_dual) - dual_obj;
            *error_bound = sqrt(2 * max(gap, 0.0) / sum_wt);
            if (*error_bound <= max_rms_error || iter == max_iter) {
                if (primal_dual < primal_u) u = u_dual;
                break;
            }
        }

        // Balance the residuals; alpha decays so that the steps settle
        apply_dt(dp, dtdp);
        double primal_res2 = 0;
        for (int64_t i = 0; i < nr_entries; ++i) {
            if (inv_degree[i] == 0) continue;
            double r = du[i] / (gamma * inv_degree[i]) - dtdp[i];
            primal_res2 += r * r;
        }
        if (primal_res2 > delta * delta * dual_res2) {
            gamma /= 1 - alpha;
            alpha *= eta;
        } else if (primal_res2 * delta * delta < dual_res2) {
            gamma *= 1 - alpha;
            alpha *= eta;
        }
    }
    return iter;
}

int main(int argc, char* argv[]) {
    bool use_outpath = false, do_masking = false, replace = false, joint4d = false;
    char *fout = NULL, *f_input = NULL, *f_column = NULL, *f_layer = NULL;
    double lambda = -1;
    int max_iter = 100000;
    double tol = 0.01;
    int ac;
    if (argc < 3) return show_help();

    for (ac = 1; ac < argc; ac++) {
        if (!strncmp(argv[ac], "-h", 2)) {
            return show_help();
        } else if (!strcmp(argv[ac], "-column_file")) {
            if (++ac >= argc) {
                fprintf(stderr, "** missing argument for -column_file\n");
                return 1;
            }
            f_column = argv[ac];
        } else if (!strcmp(argv[ac], "-layer_file")) {
            if (++ac >= argc) {
                fprintf(stderr, "** missing argument for -layer_file\n");
                return 1;
            }
            f_layer = argv[ac];
        } else if (!strcmp(argv[ac], "-input")) {
            if (++ac >= argc) {
                fprintf(stderr, "** missing argument for -input\n");
                return 1;
            }
            f_input = argv[ac];
        } else if (!strcmp(argv[ac], "-lambda")) {
            if (++ac >= argc) {
                fprintf(stderr, "** missing argument for -lambda\n");
                return 1;
            }
            lambda = atof(argv[ac]);
        } else if (!strcmp(argv[ac], "-iter")) {
            if (++ac >= argc) {
                fprintf(stderr, "** missing argument for -iter\n");
                return 1;
            }
            max_iter = atoi(argv[ac]);
        } else if (!strcmp(argv[ac], "-tol")) {
            if (++ac >= argc) {
                fprintf(stderr, "** missing argument for -tol\n");
                return 1;
            }
            tol = atof(argv[ac]);
        } else if (!strcmp(argv[ac], "-replace")) {
            replace = true;
            cout << "Replace each voxel with its smoothed cell mean."  << endl;
        } else if (!strcmp(argv[ac], "-joint4D")) {
            joint4d = true;
            cout << "Smooth all volumes together."  << endl;
        } else if (!strcmp(argv[ac], "-mask")) {
            do_masking = true;
            cout << "Set voxels to zero outside columns (mask option)"  << endl;
        } else if (!strcmp(argv[ac], "-output")) {
            if (++ac >= argc) {
                fprintf(stderr, "** missing argument for -output\n");
                return 1;
            }
            use_outpath = true;
            fout = argv[ac];
        } else {
            fprintf(stderr, "** invalid option, '%s'\n", argv[ac]);
            return 1;
        }
    }

    if (!f_input) {
        fprintf(stderr, "** missing option '-input'\n");
        return 1;
    }
    if (!f_column) {
        fprintf(stderr, "** missing option '-column_file'\n");
        return 1;
    }
    if (!f_layer) {
        fprintf(stderr, "** missing option '-layer_file'\n");
        return 1;
    }
    if (lambda < 0) {
        fprintf(stderr, "** missing or negative option '-lambda'\n");
        return 1;
    }

    // Read inputs including data
    nifti_image* nii1 = nifti_image_read(f_input, 1);
    if (!nii1) {
        fprintf(stderr, "** failed to read NIfTI from '%s'\n", f_input);
        return 2;
    }

    nifti_image* nii2 = nifti_image_read(f_column, 1);
    if (!nii2) {
        fprintf(stderr, "** failed to read NIfTI from '%s'\n", f_column);
        return 2;
    }

    nifti_image* nii3 = nifti_image_read(f_layer, 1);
    if (!nii3) {
        fprintf(stderr, "** failed to read NIfTI from '%s'\n", f_layer);
        return 2;
    }

    log_welcome("LN2_COLUMN_SMOOTH_TV");
    log_nifti_descriptives(nii1);
    log_nifti_descriptives(nii2);
    log_nifti_descriptives(nii3);

    if (nii1->nx != nii2->nx || nii1->ny != nii2->ny || nii1->nz != nii2->nz) {
        fprintf(stderr, "** input and column file x, y, z dimensions differ\n");
        return 2;
    }
    if (nii1->nx != nii3->nx || nii1->ny != nii3->ny || nii1->nz != nii3->nz) {
        fprintf(stderr, "** input and layer file x, y, z dimensions differ\n");
        return 2;
    }

    // Get dimensions of input
    const int64_t size_x = nii2->nx;
    const int64_t size_y = nii2->ny;
    const int64_t size_z = nii2->nz;
    const int64_t size_time = max(static_cast<int64_t>(nii1->nt), static_cast<int64_t>(1));
    const int64_t nxy = size_x * size_y;
    const int64_t nr_voxels = nxy * size_z;

    // ========================================================================
    // Fix datatype issues
    nifti_image* nii_smooth = copy_nifti_as_float32_with_scl_slope_and_scl_inter(nii1);
    float* nii_smooth_data = static_cast<float*>(nii_smooth->data);
    nifti_image* nii_column = copy_nifti_as_int32(nii2);
    int32_t* nii_column_data = static_cast<int32_t*>(nii_column->data);
    nifti_image* nii_layer = copy_nifti_as_int32(nii3);
    int32_t* nii_layer_data = static_cast<int32_t*>(nii_layer->data);

    // Voxels with both a column and a layer label take part
    auto in_cell = [&](int64_t i) {
        return *(nii_column_data + i) > 0 && *(nii_layer_data + i) > 0;
    };

    int32_t nr_cols = 0, nr_layers = 0;
    for (int64_t i = 0; i < nr_voxels; ++i) {
        nr_cols = max(nr_cols, *(nii_column_data + i));
        if (in_cell(i)) nr_layers = max(nr_layers, *(nii_layer_data + i));
    }
    const int64_t nr_nodes = nr_cols + 1;  // Node 0 is unused

    // Cell index: column * nr_layers + (layer - 1)
    auto cell_of = [&](int64_t i) {
        return static_cast<int64_t>(*(nii_column_data + i)) * nr_layers
               + *(nii_layer_data + i) - 1;
    };
    vector<int64_t> cell_count(nr_nodes * nr_layers, 0);
    for (int64_t i = 0; i < nr_voxels; ++i) {
        if (in_cell(i)) cell_count[cell_of(i)]++;
    }

    // ========================================================================
    // Find touching columns: count the voxel faces each pair shares
    // ========================================================================
    unordered_map<uint64_t, int32_t> face_count;
    for (int64_t iz = 0; iz < size_z; ++iz) {
        for (int64_t iy = 0; iy < size_y; ++iy) {
            for (int64_t ix = 0; ix < size_x; ++ix) {
                int64_t i = nxy * iz + size_x * iy + ix;
                int32_t k = *(nii_column_data + i);
                if (k <= 0) continue;
                // +x, +y, +z neighbours, so each face is counted one time
                int64_t next[3] = {ix + 1 < size_x ? i + 1 : -1,
                                   iy + 1 < size_y ? i + size_x : -1,
                                   iz + 1 < size_z ? i + nxy : -1};
                for (int n = 0; n < 3; ++n) {
                    if (next[n] < 0) continue;
                    int32_t l = *(nii_column_data + next[n]);
                    if (l <= 0 || l == k) continue;
                    uint64_t key = static_cast<uint64_t>(min(k, l)) * nr_nodes + max(k, l);
                    face_count[key]++;
                }
            }
        }
    }
    vector<int32_t> edge_k, edge_l;
    vector<double> edge_w;
    for (auto& f : face_count) {
        edge_k.push_back(f.first / nr_nodes);
        edge_l.push_back(f.first % nr_nodes);
        edge_w.push_back(f.second);
    }
    cout << "  There are " << nr_cols << " columns, " << nr_layers
         << " layers and " << edge_k.size() << " pairs of touching columns." << endl;

    // ========================================================================
    // Smooth the layer profiles: one problem for each volume, or one for all
    // ========================================================================
    const int64_t nr_vols_per_problem = joint4d ? size_time : 1;
    const int64_t M = nr_layers * nr_vols_per_problem;
    vector<double> x(nr_nodes * M), wt(nr_nodes * M), u;

    for (int64_t t0 = 0; t0 < size_time; t0 += nr_vols_per_problem) {
        // Profile entry m = (t - t0) * nr_layers + (layer - 1)
        fill(x.begin(), x.end(), 0.0);
        for (int64_t dt = 0; dt < nr_vols_per_problem; ++dt) {
            const float* data_t = nii_smooth_data + nr_voxels * (t0 + dt);
            for (int64_t i = 0; i < nr_voxels; ++i) {
                if (!in_cell(i)) continue;
                int64_t c = cell_of(i);
                x[(c / nr_layers) * M + dt * nr_layers + c % nr_layers] += *(data_t + i);
            }
        }
        for (int64_t k = 0; k < nr_nodes; ++k) {
            for (int64_t m = 0; m < M; ++m) {
                int64_t n = cell_count[k * nr_layers + m % nr_layers];
                wt[k * M + m] = n;
                if (n > 0) x[k * M + m] /= n;
            }
        }

        // Data scale for -tol: RMS of the voxels' cell means
        double sum_wx2 = 0, sum_w = 0;
        for (int64_t i = 0; i < nr_nodes * M; ++i) {
            sum_wx2 += wt[i] * x[i] * x[i];
            sum_w += wt[i];
        }
        const double max_rms_error = tol * sqrt(sum_wx2 / max(sum_w, 1.0));

        u = x;
        double error_bound;
        int iter = solve_tv(u, x, wt, nr_nodes, M, edge_k, edge_l, edge_w,
                            lambda, max_iter, max_rms_error, &error_bound);
        cout << "  Volume " << t0 + 1;
        if (nr_vols_per_problem > 1) cout << "-" << t0 + nr_vols_per_problem;
        cout << ": " << iter << " iterations, RMS error bound " << error_bound
             << " (target " << max_rms_error << ")" << endl;
        if (error_bound > max_rms_error) {
            cout << "  WARNING: target not reached; increase -iter or -tol." << endl;
        }

        for (int64_t dt = 0; dt < nr_vols_per_problem; ++dt) {
            float* data_t = nii_smooth_data + nr_voxels * (t0 + dt);
            for (int64_t i = 0; i < nr_voxels; ++i) {
                if (in_cell(i)) {
                    int64_t c = cell_of(i);
                    int64_t e = (c / nr_layers) * M + dt * nr_layers + c % nr_layers;
                    *(data_t + i) = replace ? u[e] : *(data_t + i) + (u[e] - x[e]);
                } else if (do_masking) {
                    *(data_t + i) = 0;
                }
            }
        }
    }
    cout << "  Smoothing is done. " <<  endl;

    if (!use_outpath) fout = f_input;
    save_output_nifti(fout, "column_smoothed_TV", nii_smooth, true, use_outpath);

    cout << "  Finished." << endl;
    return 0;
}

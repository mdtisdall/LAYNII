
#include "../dep/laynii_lib.h"

int show_help(void) {
    printf(
    "LN2_COLUMN_NORMALIZE : Normalize the layer profile of each column.\n"
    "\n"
    "    For each column, this program averages the voxels in each layer of\n"
    "    the column (one 'cell' for each column and layer). The cell means\n"
    "    of a column make its layer profile, a vector with one value for\n"
    "    each layer. Each voxel of the column is then divided by the L2 norm\n"
    "    of that profile:\n"
    "        norm = sqrt(sum over layers of (mean of cell)^2)\n"
    "        out  = x / norm\n"
    "    Because the profile uses cell means, the number of voxels in a\n"
    "    cell does not change the norm. Layers that a column does not reach\n"
    "    add 0 to the norm. 4D inputs are normalized for each volume\n"
    "    separately. A column whose norm is 0 gives 0.\n"
    "\n"
    "    Only voxels with both a column and a layer label are normalized.\n"
    "\n"
    "Usage:\n"
    "    LN2_COLUMN_NORMALIZE -input activity_map.nii -column_file columns.nii -layer_file layers.nii\n"
    "    ../LN2_COLUMN_NORMALIZE -input lo_BOLD_act.nii.gz -column_file lo_columns.nii.gz -layer_file lo_layers.nii.gz\n"
    "\n"
    "Options:\n"
    "    -help        : Show this help.\n"
    "    -input       : Nifti (.nii) file that should be normalized. This\n"
    "                   can be a 4D nifti.\n"
    "    -column_file : Nifti (.nii) file that contains column labels\n"
    "                   (0 = not a column). Same x, y, z dimensions as the\n"
    "                   input.\n"
    "    -layer_file  : Nifti (.nii) file that contains layer labels\n"
    "                   (0 = not a layer). Same x, y, z dimensions as the\n"
    "                   input.\n"
    "    -replace     : (Optional) Replace each voxel with its normalized\n"
    "                   cell mean (mean / norm), so the output of each column\n"
    "                   is exactly its unit-length layer profile.\n"
    "    -NoKissing   : (Optional) Use each spatially connected piece\n"
    "                   (26-neighborhood) of a column as its own column. Use\n"
    "                   this when a column touches itself across sulci. All\n"
    "                   voxels with the column label connect the pieces,\n"
    "                   also voxels without a layer label.\n"
    "    -mask        : (Optional) Set voxels outside of the columns or\n"
    "                   layers to 0. Without this option they are copied\n"
    "                   unchanged.\n"
    "    -output      : (Optional) Output filename, including .nii or\n"
    "                   .nii.gz, and path if needed. Overwrites existing files.\n"
    "\n");
    return 0;
}

int main(int argc, char* argv[]) {
    bool use_outpath = false, do_masking = false, nokissing = false, replace = false;
    char *fout = NULL, *f_input = NULL, *f_column = NULL, *f_layer = NULL;
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
        } else if (!strcmp(argv[ac], "-replace")) {
            replace = true;
            cout << "Replace each voxel with its normalized cell mean."  << endl;
        } else if (!strcmp(argv[ac], "-NoKissing")) {
            nokissing = true;
            cout << "Normalize each connected piece of a column separately."  << endl;
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

    log_welcome("LN2_COLUMN_NORMALIZE");
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
    nifti_image* nii_norm = copy_nifti_as_float32_with_scl_slope_and_scl_inter(nii1);
    float* nii_norm_data = static_cast<float*>(nii_norm->data);
    nifti_image* nii_column = copy_nifti_as_int32(nii2);
    int32_t* nii_column_data = static_cast<int32_t*>(nii_column->data);
    nifti_image* nii_layer = copy_nifti_as_int32(nii3);
    int32_t* nii_layer_data = static_cast<int32_t*>(nii_layer->data);

    // ========================================================================
    // Find the node (column, or connected piece) that each voxel belongs to
    // ========================================================================
    vector<int32_t> node(nr_voxels, 0);  // 0 = not a column
    int32_t nr_cols = 0;
    if (!nokissing) {
        for (int64_t i = 0; i < nr_voxels; ++i) {
            node[i] = max(*(nii_column_data + i), 0);
            nr_cols = max(nr_cols, node[i]);
        }
    } else {
        // Flood fill each connected piece of each column with a new id
        vector<int64_t> queue;
        for (int64_t i = 0; i < nr_voxels; ++i) {
            int32_t label_i = *(nii_column_data + i);
            if (label_i <= 0 || node[i] != 0) continue;

            nr_cols++;
            node[i] = nr_cols;
            queue.clear();
            queue.push_back(i);
            for (size_t q = 0; q < queue.size(); ++q) {
                int64_t j = queue[q];
                int64_t jz = j / nxy;
                int64_t jy = (j % nxy) / size_x;
                int64_t jx = j % size_x;
                for (int64_t kz = max(jz - 1, (int64_t)0); kz <= min(jz + 1, size_z - 1); ++kz) {
                    for (int64_t ky = max(jy - 1, (int64_t)0); ky <= min(jy + 1, size_y - 1); ++ky) {
                        for (int64_t kx = max(jx - 1, (int64_t)0); kx <= min(jx + 1, size_x - 1); ++kx) {
                            int64_t k = nxy * kz + size_x * ky + kx;
                            if (node[k] == 0 && *(nii_column_data + k) == label_i) {
                                node[k] = nr_cols;
                                queue.push_back(k);
                            }
                        }
                    }
                }
            }
        }
    }

    // Voxels with both a column and a layer label take part
    auto in_cell = [&](int64_t i) {
        return node[i] > 0 && *(nii_layer_data + i) > 0;
    };

    int32_t nr_layers = 0;
    for (int64_t i = 0; i < nr_voxels; ++i) {
        if (in_cell(i)) nr_layers = max(nr_layers, *(nii_layer_data + i));
    }
    cout << "  There are " << nr_cols << (nokissing ? " connected column pieces and " : " columns and ")
         << nr_layers << " layers." << endl;

    // Cell index: node * (nr_layers + 1) + layer
    const int64_t nr_cells = static_cast<int64_t>(nr_cols + 1) * (nr_layers + 1);
    auto cell_of = [&](int64_t i) {
        return static_cast<int64_t>(node[i]) * (nr_layers + 1) + *(nii_layer_data + i);
    };

    vector<int64_t> cell_count(nr_cells, 0);
    for (int64_t i = 0; i < nr_voxels; ++i) {
        if (in_cell(i)) cell_count[cell_of(i)]++;
    }

    // ========================================================================
    // Normalize the layer profile of each column, for each volume
    // ========================================================================
    vector<double> cell_mean(nr_cells);
    vector<double> col_norm(nr_cols + 1);
    for (int64_t t = 0; t < size_time; ++t) {
        float* data_t = nii_norm_data + nr_voxels * t;

        fill(cell_mean.begin(), cell_mean.end(), 0.0);
        for (int64_t i = 0; i < nr_voxels; ++i) {
            if (in_cell(i)) cell_mean[cell_of(i)] += *(data_t + i);
        }

        fill(col_norm.begin(), col_norm.end(), 0.0);
        for (int64_t c = 0; c < nr_cells; ++c) {
            if (cell_count[c] > 0) {
                cell_mean[c] /= cell_count[c];
                col_norm[c / (nr_layers + 1)] += cell_mean[c] * cell_mean[c];
            }
        }
        for (int32_t v = 1; v <= nr_cols; ++v) {
            col_norm[v] = sqrt(col_norm[v]);
        }

        for (int64_t i = 0; i < nr_voxels; ++i) {
            if (in_cell(i)) {
                double norm = col_norm[node[i]];
                double val = replace ? cell_mean[cell_of(i)] : *(data_t + i);
                *(data_t + i) = norm > 0 ? val / norm : 0;
            } else if (do_masking) {
                *(data_t + i) = 0;
            }
        }
    }
    cout << "  Normalizing is done. " <<  endl;

    if (!use_outpath) fout = f_input;
    save_output_nifti(fout, "column_normalized", nii_norm, true, use_outpath);

    cout << "  Finished." << endl;
    return 0;
}

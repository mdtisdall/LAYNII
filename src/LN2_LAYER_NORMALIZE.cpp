
#include "../dep/laynii_lib.h"

int show_help(void) {
    printf(
    "LN2_LAYER_NORMALIZE : Normalize data within layers or columns.\n"
    "\n"
    "    This program treats the voxel intensities that share a label in the\n"
    "    layer file as one vector and divides each intensity by the L2 norm\n"
    "    of that vector. After this, each label has unit length:\n"
    "        out_i = x_i / sqrt(sum of x_j^2 over all voxels j in the label)\n"
    "    This is mostly useful with a column file: the intensities within\n"
    "    each column are normalized to unit length. 4D inputs are\n"
    "    normalized for each volume separately. A label whose norm is 0\n"
    "    gives 0.\n"
    "\n"
    "Usage:\n"
    "    LN2_LAYER_NORMALIZE -layer_file columns.nii -input activity_map.nii\n"
    "    ../LN2_LAYER_NORMALIZE -input lo_BOLD_act.nii.gz -layer_file lo_columns.nii.gz\n"
    "\n"
    "Options:\n"
    "    -help       : Show this help.\n"
    "    -layer_file : Nifti (.nii) file that contains layer or column masks.\n"
    "    -input      : Nifti (.nii) file that should be normalized. It\n"
    "                  should have the same x, y, z dimensions as the layer\n"
    "                  file. This can be a 4D nifti.\n"
    "    -NoKissing  : (Optional) Normalize each spatially connected piece\n"
    "                  (26-neighborhood) of a label as its own vector. Use\n"
    "                  this when a label touches itself across sulci.\n"
    "    -mask       : (Optional) Set voxels outside of the layers to 0.\n"
    "                  Without this option they are copied unchanged.\n"
    "    -output     : (Optional) Output filename, including .nii or\n"
    "                  .nii.gz, and path if needed. Overwrites existing files.\n"
    "\n");
    return 0;
}

int main(int argc, char* argv[]) {
    bool use_outpath = false, do_masking = false, nokissing = false;
    char *fout = NULL, *f_input = NULL, *f_layer = NULL;
    int ac;
    if (argc < 3) return show_help();

    for (ac = 1; ac < argc; ac++) {
        if (!strncmp(argv[ac], "-h", 2)) {
            return show_help();
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
        } else if (!strcmp(argv[ac], "-NoKissing")) {
            nokissing = true;
            cout << "Normalize each connected piece of a label separately."  << endl;
        } else if (!strcmp(argv[ac], "-mask")) {
            do_masking = true;
            cout << "Set voxels to zero outside layers (mask option)"  << endl;
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

    nifti_image* nii2 = nifti_image_read(f_layer, 1);
    if (!nii2) {
        fprintf(stderr, "** failed to read NIfTI from '%s'\n", f_layer);
        return 2;
    }

    log_welcome("LN2_LAYER_NORMALIZE");
    log_nifti_descriptives(nii1);
    log_nifti_descriptives(nii2);

    if (nii1->nx != nii2->nx || nii1->ny != nii2->ny || nii1->nz != nii2->nz) {
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
    nifti_image* nii_layer = copy_nifti_as_int32(nii2);
    int32_t* nii_layer_data = static_cast<int32_t*>(nii_layer->data);

    // ========================================================================
    // Find the vector that each voxel belongs to (0 = none)
    // ========================================================================
    vector<int32_t> vec_id(nr_voxels, 0);
    int32_t nr_vectors = 0;

    if (!nokissing) {
        for (int64_t i = 0; i < nr_voxels; ++i) {
            if (*(nii_layer_data + i) > 0) {
                vec_id[i] = *(nii_layer_data + i);
                nr_vectors = max(nr_vectors, vec_id[i]);
            }
        }
        cout << "  There are " << nr_vectors << " labels to normalize within." << endl;
    } else {
        // Flood fill each connected piece of each label with a new id
        vector<int64_t> queue;
        for (int64_t i = 0; i < nr_voxels; ++i) {
            int32_t label_i = *(nii_layer_data + i);
            if (label_i <= 0 || vec_id[i] != 0) continue;

            nr_vectors++;
            vec_id[i] = nr_vectors;
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
                            if (vec_id[k] == 0 && *(nii_layer_data + k) == label_i) {
                                vec_id[k] = nr_vectors;
                                queue.push_back(k);
                            }
                        }
                    }
                }
            }
        }
        cout << "  There are " << nr_vectors << " connected pieces to normalize within." << endl;
    }

    // ========================================================================
    // Normalize each vector, for each volume
    // ========================================================================
    vector<double> sum_sq(nr_vectors + 1);
    for (int64_t t = 0; t < size_time; ++t) {
        float* data_t = nii_norm_data + nr_voxels * t;

        fill(sum_sq.begin(), sum_sq.end(), 0.0);
        for (int64_t i = 0; i < nr_voxels; ++i) {
            if (vec_id[i] > 0) {
                sum_sq[vec_id[i]] += static_cast<double>(*(data_t + i)) * *(data_t + i);
            }
        }
        for (int32_t v = 1; v <= nr_vectors; ++v) {
            sum_sq[v] = sqrt(sum_sq[v]);  // Now the norm
        }

        for (int64_t i = 0; i < nr_voxels; ++i) {
            if (vec_id[i] > 0) {
                double norm = sum_sq[vec_id[i]];
                *(data_t + i) = norm > 0 ? *(data_t + i) / norm : 0;
            } else if (do_masking) {
                *(data_t + i) = 0;
            }
        }
    }
    cout << "  Normalizing is done. " <<  endl;

    if (!use_outpath) fout = f_input;
    save_output_nifti(fout, "layer_normalized", nii_norm, true, use_outpath);

    cout << "  Finished." << endl;
    return 0;
}

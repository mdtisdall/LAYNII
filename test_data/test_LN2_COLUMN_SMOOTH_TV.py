#!/usr/bin/env python3
"""Correctness tests for LN2_COLUMN_SMOOTH_TV on small synthetic images.

Usage: python3 test_data/test_LN2_COLUMN_SMOOTH_TV.py ./LN2_COLUMN_SMOOTH_TV

Uses only the Python standard library: images are written as float32
NIfTI-1 and outputs are read as NIfTI-1 or NIfTI-2.

All tests use two touching columns, so the exact solution is known. For
two profiles x1, x2 with n voxels in each cell, w shared faces and
d = x1 - x2:
  - if lambda * w >= n * |d| / 2, both become (x1 + x2) / 2;
  - else u1 = x1 - (lambda * w / n) * d / |d| and u2 = x2 + (same).
"""
import math
import os
import struct
import subprocess
import sys
import tempfile

SHAPE = (4, 2, 2)  # x, y, z
TOL = 1e-4


def write_nifti(path, data, nt=1, shape=SHAPE):
    """Write a list of floats (x fastest, then y, z, t) as a float32 NIfTI-1."""
    hdr = bytearray(348)
    struct.pack_into("<i", hdr, 0, 348)
    struct.pack_into("<8h", hdr, 40, 4 if nt > 1 else 3, *shape, nt, 1, 1, 1)
    struct.pack_into("<hh", hdr, 70, 16, 32)  # NIFTI_TYPE_FLOAT32, bitpix
    struct.pack_into("<8f", hdr, 76, 1, 1, 1, 1, 1, 1, 1, 1)
    struct.pack_into("<fff", hdr, 108, 352, 1, 0)  # vox_offset, scl_slope, scl_inter
    hdr[344:348] = b"n+1\0"
    with open(path, "wb") as f:
        f.write(hdr + bytes(4) + struct.pack("<%df" % len(data), *data))


def read_nifti(path):
    """Read a float32 NIfTI-1 or NIfTI-2 file as a list of floats."""
    with open(path, "rb") as f:
        raw = f.read()
    if struct.unpack_from("<i", raw, 0)[0] == 348:
        dtype = struct.unpack_from("<h", raw, 70)[0]
        nvox = math.prod(struct.unpack_from("<7h", raw, 42)[:4])
        offset = int(struct.unpack_from("<f", raw, 108)[0])
    else:
        dtype = struct.unpack_from("<h", raw, 12)[0]
        nvox = math.prod(struct.unpack_from("<7q", raw, 24)[:4])
        offset = struct.unpack_from("<q", raw, 168)[0]
    assert dtype == 16, "expected float32 output, got datatype %d" % dtype
    return list(struct.unpack_from("<%df" % nvox, raw, offset))


def run(binary, tmp, data, columns, layers, *flags, nt=1, shape=SHAPE):
    f_in = os.path.join(tmp, "input.nii")
    f_column = os.path.join(tmp, "columns.nii")
    f_layer = os.path.join(tmp, "layers.nii")
    f_out = os.path.join(tmp, "output.nii")
    write_nifti(f_in, data, nt, shape)
    write_nifti(f_column, columns, 1, shape)
    write_nifti(f_layer, layers, 1, shape)
    # A tight -tol, so that the output is the exact optimum within TOL
    subprocess.run([binary, "-input", f_in, "-column_file", f_column,
                    "-layer_file", f_layer, "-output", f_out, "-tol", "1e-7",
                    *flags],
                   check=True, stdout=subprocess.DEVNULL)
    return read_nifti(f_out)


def two_column_solution(x1, x2, n, w, lam):
    d = [a - b for a, b in zip(x1, x2)]
    norm = math.sqrt(sum(v * v for v in d))
    if lam * w >= n * norm / 2:
        mean = [(a + b) / 2 for a, b in zip(x1, x2)]
        return mean, list(mean)
    s = lam * w / n / norm
    return ([a - s * v for a, v in zip(x1, d)],
            [b + s * v for b, v in zip(x2, d)])


def check(out, exp, name):
    for i, (a, b) in enumerate(zip(out, exp)):
        assert abs(a - b) <= TOL * max(1.0, abs(b)), \
            "%s: voxel %d is %g, expected %g" % (name, i, a, b)


def coords(shape=SHAPE):
    for z in range(shape[2]):
        for y in range(shape[1]):
            for x in range(shape[0]):
                yield x, y, z


# Column 1 is x = 0..1, column 2 is x = 2..3: they share 4 faces.
COLUMNS = [1 if x <= 1 else 2 for x, y, z in coords()]
W = 4
# Offsets from the cell mean; they sum to 0 in each cell of 4 voxels.
OFFSET = {(0, 0): -1.5, (1, 0): 0.5, (0, 1): 2.0, (1, 1): -1.0,
          (2, 0): 1.0, (3, 0): -0.5, (2, 1): 0.25, (3, 1): -0.75}


def two_layer_data():
    """Layer 1 is z = 0, layer 2 is z = 1, so n = 4 in each cell. Returns
    the data, the layers, the cell means and a function that gives the
    output for given profiles u1, u2."""
    layers = [z + 1 for x, y, z in coords()]
    means = {1: [10.0, 4.0], 2: [2.0, 8.0]}
    data = [means[c][l - 1] + OFFSET[(x, y)]
            for (x, y, z), c, l in zip(coords(), COLUMNS, layers)]

    def output(u, replace=False):
        out = []
        for v, c, l in zip(data, COLUMNS, layers):
            out.append(u[c][l - 1] if replace else v + u[c][l - 1] - means[c][l - 1])
        return out
    return data, layers, means, output


def test_lambda_zero(binary, tmp):
    data, layers, _, _ = two_layer_data()
    out = run(binary, tmp, data, COLUMNS, layers, "-lambda", "0")
    check(out, data, "lambda 0")


def test_shrink_and_replace(binary, tmp):
    # |d| = sqrt(80), so the columns join at lambda >= 4 * sqrt(80) / 8 = 4.47
    data, layers, means, output = two_layer_data()
    u1, u2 = two_column_solution(means[1], means[2], 4, W, 2.0)
    assert u1 != u2
    out = run(binary, tmp, data, COLUMNS, layers, "-lambda", "2")
    check(out, output({1: u1, 2: u2}), "shrink")
    out = run(binary, tmp, data, COLUMNS, layers, "-lambda", "2", "-replace")
    check(out, output({1: u1, 2: u2}, replace=True), "shrink -replace")


def test_join(binary, tmp):
    data, layers, means, output = two_layer_data()
    u1, u2 = two_column_solution(means[1], means[2], 4, W, 10.0)
    assert u1 == u2
    out = run(binary, tmp, data, COLUMNS, layers, "-lambda", "10")
    check(out, output({1: u1, 2: u2}), "join")


def test_empty_cell(binary, tmp):
    # Three columns in a row (x = 0..1, 2..3, 4..5); columns 1 and 3 do not
    # touch. Column 2 has no voxels in layer 2 (layer label 0 there), so
    # each pair is compared on layer 1 only, where all three columns are
    # equal: nothing changes. If the empty cell took part, it would link
    # the different layer 2 values of columns 1 and 3 and pull them together.
    shape = (6, 2, 2)
    columns = [x // 2 + 1 for x, y, z in coords(shape)]
    layers = [0 if c == 2 and z == 1 else z + 1
              for (x, y, z), c in zip(coords(shape), columns)]
    data = [5.0 if l == 1 else {1: 10.0, 2: 0.0, 3: 2.0}[c]
            for c, l in zip(columns, layers)]
    out = run(binary, tmp, data, columns, layers, "-lambda", "2", shape=shape)
    check(out, data, "empty cell")


def test_nokissing(binary, tmp):
    # Three blocks in a row (x = 0..1, 2..3, 4..5). The outer two have the
    # same label, so without -NoKissing they are one column. With
    # -NoKissing they are two pieces, and the result must be the same as
    # with three different labels.
    shape = (6, 2, 2)
    split = [{0: 1, 1: 2, 2: 1}[x // 2] for x, y, z in coords(shape)]
    three = [x // 2 + 1 for x, y, z in coords(shape)]
    layers = [z + 1 for x, y, z in coords(shape)]
    data = [{1: 10.0, 2: 4.0, 3: 0.0}[c] + (1 if z else -1) * (x % 2 - 0.5)
            for (x, y, z), c in zip(coords(shape), three)]
    exp = run(binary, tmp, data, three, layers, "-lambda", "2", shape=shape)
    out = run(binary, tmp, data, split, layers, "-lambda", "2", "-NoKissing",
              shape=shape)
    check(out, exp, "-NoKissing")
    out = run(binary, tmp, data, split, layers, "-lambda", "2", shape=shape)
    assert any(abs(a - b) > 1e-3 for a, b in zip(out, exp)), \
        "without -NoKissing the split label should be one column"


def test_joint4d(binary, tmp):
    # One layer, so n = 8 in each cell. Volume 1 has d = 4 and volume 2 has
    # d = 1. With lambda = 2, volume 2 joins on its own (threshold 1) and
    # volume 1 does not (threshold 4). Together, |d| = sqrt(17) > 2, so
    # neither joins.
    layers = [1] * len(COLUMNS)
    means = [{1: 6.0, 2: 2.0}, {1: 3.0, 2: 2.0}]
    vols = [[m[c] + OFFSET[(x, y)] for (x, y, z), c in zip(coords(), COLUMNS)]
            for m in means]

    def output(u):
        return [v + u[t][c] - means[t][c]
                for t in range(2) for v, c in zip(vols[t], COLUMNS)]

    sep = [two_column_solution([m[1]], [m[2]], 8, W, 2.0) for m in means]
    assert sep[1][0] == sep[1][1] and sep[0][0] != sep[0][1]
    out = run(binary, tmp, vols[0] + vols[1], COLUMNS, layers, "-lambda", "2", nt=2)
    check(out, output([{1: s[0][0], 2: s[1][0]} for s in sep]), "4D per volume")

    u1, u2 = two_column_solution([means[0][1], means[1][1]],
                                 [means[0][2], means[1][2]], 8, W, 2.0)
    assert u1 != u2
    out = run(binary, tmp, vols[0] + vols[1], COLUMNS, layers, "-lambda", "2",
              "-joint4D", nt=2)
    check(out, output([{1: u1[t], 2: u2[t]} for t in range(2)]), "-joint4D")


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    binary = os.path.abspath(sys.argv[1])
    tests = [test_lambda_zero, test_shrink_and_replace, test_join,
             test_empty_cell, test_nokissing, test_joint4d]
    with tempfile.TemporaryDirectory() as tmp:
        for test in tests:
            test(binary, tmp)
            print("ok    %s" % test.__name__)
    print("test_LN2_COLUMN_SMOOTH_TV: all %d tests passed" % len(tests))


if __name__ == "__main__":
    main()

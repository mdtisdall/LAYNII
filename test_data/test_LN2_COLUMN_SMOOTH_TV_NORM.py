#!/usr/bin/env python3
"""Correctness tests for LN2_COLUMN_SMOOTH_TV_NORM on small synthetic images.

Usage: python3 test_data/test_LN2_COLUMN_SMOOTH_TV_NORM.py ./LN2_COLUMN_SMOOTH_TV_NORM

Uses only the Python standard library: images are written as float32
NIfTI-1 and outputs are read as NIfTI-1 or NIfTI-2.

All tests use two touching columns with n voxels in each cell and w shared
faces, so the exact solution is known. Let xn1, xn2 be the unit profiles,
2a the angle between them, b the unit bisector and e the unit vector along
xn1 - xn2. The output profiles are cos(t) b +- sin(t) e, where
  - t = 0 if lambda * w >= n * sin(a) (the columns join);
  - else tan(t) = (n * sin(a) - lambda * w) / (n * cos(a)).
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


def run(binary, tmp, data, columns, layers, *flags, nt=1):
    f_in = os.path.join(tmp, "input.nii")
    f_column = os.path.join(tmp, "columns.nii")
    f_layer = os.path.join(tmp, "layers.nii")
    f_out = os.path.join(tmp, "output.nii")
    write_nifti(f_in, data, nt)
    write_nifti(f_column, columns)
    write_nifti(f_layer, layers)
    # A tight -tol, so that the output is the exact optimum within TOL
    subprocess.run([binary, "-input", f_in, "-column_file", f_column,
                    "-layer_file", f_layer, "-output", f_out, "-tol", "1e-7",
                    *flags],
                   check=True, stdout=subprocess.DEVNULL)
    return read_nifti(f_out)


def unit(v):
    norm = math.sqrt(sum(a * a for a in v))
    return [a / norm for a in v]


def two_column_solution(x1, x2, n, w, lam):
    """Output unit profiles for input profiles x1, x2 (any norm)."""
    xn1, xn2 = unit(x1), unit(x2)
    a = math.acos(max(-1.0, min(1.0, sum(p * q for p, q in zip(xn1, xn2))))) / 2
    b = unit([p + q for p, q in zip(xn1, xn2)])
    e = unit([p - q for p, q in zip(xn1, xn2)])
    t = 0.0
    if lam * w < n * math.sin(a):
        t = math.atan((n * math.sin(a) - lam * w) / (n * math.cos(a)))
    return ([math.cos(t) * p + math.sin(t) * q for p, q in zip(b, e)],
            [math.cos(t) * p - math.sin(t) * q for p, q in zip(b, e)])


def check(out, exp, name):
    for i, (a, b) in enumerate(zip(out, exp)):
        assert abs(a - b) <= TOL * max(1.0, abs(b)), \
            "%s: voxel %d is %g, expected %g" % (name, i, a, b)


def coords():
    for z in range(SHAPE[2]):
        for y in range(SHAPE[1]):
            for x in range(SHAPE[0]):
                yield x, y, z


# Column 1 is x = 0..1, column 2 is x = 2..3: they share 4 faces.
COLUMNS = [1 if x <= 1 else 2 for x, y, z in coords()]
W = 4
# Offsets from the cell mean; they sum to 0 in each cell of 4 voxels.
OFFSET = {(0, 0): -1.5, (1, 0): 0.5, (0, 1): 2.0, (1, 1): -1.0,
          (2, 0): 1.0, (3, 0): -0.5, (2, 1): 0.25, (3, 1): -0.75}
# Layer 1 is z = 0, layer 2 is z = 1, so n = 4 in each cell.
LAYERS = [z + 1 for x, y, z in coords()]
MEANS = {1: [10.0, 4.0], 2: [2.0, 8.0]}
# The columns join at lambda >= n * sin(a) / w = 0.455
DATA = [MEANS[c][l - 1] + OFFSET[(x, y)]
        for (x, y, z), c, l in zip(coords(), COLUMNS, LAYERS)]
NORM = {c: math.sqrt(sum(v * v for v in m)) for c, m in MEANS.items()}


def expected(u, replace=False, data=DATA, means=MEANS):
    """Output for the unit output profiles u (shift or replace rule)."""
    norm = {c: math.sqrt(sum(v * v for v in m)) for c, m in means.items()}
    out = []
    for v, c, l in zip(data, COLUMNS, LAYERS):
        un = u[c][l - 1]
        out.append(un if replace else v / norm[c] + un - means[c][l - 1] / norm[c])
    return out


def test_lambda_zero(binary, tmp):
    out = run(binary, tmp, DATA, COLUMNS, LAYERS, "-lambda", "0")
    check(out, [v / NORM[c] for v, c in zip(DATA, COLUMNS)], "lambda 0")

    # One column label in two pieces that do not touch (x = 0 and x = 3),
    # with different scales. -NoKissing normalizes each piece separately.
    columns = [1 if x in (0, 3) else 2 for x, y, z in coords()]
    data = [v * (5 if x == 3 else 1) for v, (x, y, z) in zip(DATA, coords())]
    out = run(binary, tmp, data, columns, LAYERS, "-lambda", "0", "-NoKissing")
    pieces = {}
    for v, (x, y, z), c in zip(data, coords(), columns):
        piece = x if c == 1 else -1
        pieces.setdefault(piece, {}).setdefault(z, []).append(v)
    norm = {p: math.sqrt(sum((sum(vs) / len(vs)) ** 2 for vs in cells.values()))
            for p, cells in pieces.items()}
    exp = [v / norm[x if c == 1 else -1]
           for v, (x, y, z), c in zip(data, coords(), columns)]
    check(out, exp, "lambda 0 -NoKissing")


def test_shrink_and_replace(binary, tmp):
    u1, u2 = two_column_solution(MEANS[1], MEANS[2], 4, W, 0.2)
    assert abs(u1[0] - u2[0]) > 0.1
    out = run(binary, tmp, DATA, COLUMNS, LAYERS, "-lambda", "0.2")
    check(out, expected({1: u1, 2: u2}), "shrink")
    out = run(binary, tmp, DATA, COLUMNS, LAYERS, "-lambda", "0.2", "-replace")
    check(out, expected({1: u1, 2: u2}, replace=True), "shrink -replace")


def test_join(binary, tmp):
    u1, u2 = two_column_solution(MEANS[1], MEANS[2], 4, W, 1.0)
    assert u1 == u2
    out = run(binary, tmp, DATA, COLUMNS, LAYERS, "-lambda", "1")
    check(out, expected({1: u1, 2: u2}), "join")


def test_joint4d(binary, tmp):
    # Volume 2 = 3 x volume 1, so the unit profiles of the two volumes are
    # the same, and the joint problem is the per-volume problem with
    # lambda / sqrt(2). lambda = 0.5 joins each volume on its own
    # (threshold 0.455) but not jointly (threshold 0.643).
    vols = [DATA, [3 * v for v in DATA]]
    out = run(binary, tmp, vols[0] + vols[1], COLUMNS, LAYERS, "-lambda", "0.5", nt=2)
    u1, u2 = two_column_solution(MEANS[1], MEANS[2], 4, W, 0.5)
    assert u1 == u2
    check(out, expected({1: u1, 2: u2}) * 2, "4D per volume")

    out = run(binary, tmp, vols[0] + vols[1], COLUMNS, LAYERS, "-lambda", "0.5",
              "-joint4D", nt=2)
    u1, u2 = two_column_solution(MEANS[1], MEANS[2], 4, W, 0.5 / math.sqrt(2))
    assert u1 != u2
    check(out, expected({1: u1, 2: u2}) * 2, "-joint4D")


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    binary = os.path.abspath(sys.argv[1])
    tests = [test_lambda_zero, test_shrink_and_replace, test_join, test_joint4d]
    with tempfile.TemporaryDirectory() as tmp:
        for test in tests:
            test(binary, tmp)
            print("ok    %s" % test.__name__)
    print("test_LN2_COLUMN_SMOOTH_TV_NORM: all %d tests passed" % len(tests))


if __name__ == "__main__":
    main()

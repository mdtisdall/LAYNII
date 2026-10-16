#!/usr/bin/env python3
"""Correctness tests for LN2_COLUMN_NORMALIZE on small synthetic images.

Usage: python3 test_data/test_LN2_COLUMN_NORMALIZE.py ./LN2_COLUMN_NORMALIZE

Uses only the Python standard library: images are written as float32
NIfTI-1 and outputs are read as NIfTI-1 or NIfTI-2.
"""
import math
import os
import random
import struct
import subprocess
import sys
import tempfile

N = 6  # Images are N x N x N (x N_t)
TOL = 1e-5


def write_nifti(path, data, nt=1):
    """Write a list of floats (x fastest, then y, z, t) as a float32 NIfTI-1."""
    hdr = bytearray(348)
    struct.pack_into("<i", hdr, 0, 348)
    struct.pack_into("<8h", hdr, 40, 4 if nt > 1 else 3, N, N, N, nt, 1, 1, 1)
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
    subprocess.run([binary, "-input", f_in, "-column_file", f_column,
                    "-layer_file", f_layer, "-output", f_out, *flags],
                   check=True, stdout=subprocess.DEVNULL)
    return read_nifti(f_out)


def expected(data, groups, layers, replace=False, mask=False):
    """Reference result for one volume. groups[i] is the column (or column
    piece) of voxel i, 0 for voxels that are not normalized."""
    cells = {}
    for i, g in enumerate(groups):
        if g:
            cells.setdefault((g, layers[i]), []).append(data[i])
    mean = {c: sum(v) / len(v) for c, v in cells.items()}
    norm = {}
    for (g, _), m in mean.items():
        norm[g] = norm.get(g, 0.0) + m * m
    out = []
    for i, g in enumerate(groups):
        if g:
            val = mean[(g, layers[i])] if replace else data[i]
            out.append(val / math.sqrt(norm[g]))
        else:
            out.append(0.0 if mask else data[i])
    return out


def check(out, exp, name):
    for i, (a, b) in enumerate(zip(out, exp)):
        assert abs(a - b) <= TOL * max(1.0, abs(b)), \
            "%s: voxel %d is %g, expected %g" % (name, i, a, b)


def coords():
    for z in range(N):
        for y in range(N):
            for x in range(N):
                yield x, y, z


def setup(rng):
    # Layers 1..5 along z (z = 0 has no layer). Columns along x with
    # different widths, so the cells have different voxel counts:
    # x = 0 -> column 1, x = 1..2 -> column 2, x = 3..4 -> column 3,
    # x = 5 -> no column.
    layers = [z for x, y, z in coords()]
    columns = [{0: 1, 1: 2, 2: 2, 3: 3, 4: 3}.get(x, 0) for x, y, z in coords()]
    data = [rng.uniform(1, 10) for _ in layers]
    groups = [c if l else 0 for c, l in zip(columns, layers)]
    return data, columns, layers, groups


def test_default_and_mask(binary, tmp, rng):
    data, columns, layers, groups = setup(rng)
    out = run(binary, tmp, data, columns, layers)
    check(out, expected(data, groups, layers), "default")
    out = run(binary, tmp, data, columns, layers, "-mask")
    check(out, expected(data, groups, layers, mask=True), "-mask")


def test_replace(binary, tmp, rng):
    data, columns, layers, groups = setup(rng)
    out = run(binary, tmp, data, columns, layers, "-replace")
    check(out, expected(data, groups, layers, replace=True), "-replace")
    # Each column's profile of output cell values has unit length
    for c in (1, 2, 3):
        profile = {layers[i]: out[i] for i, g in enumerate(groups) if g == c}
        n = math.sqrt(sum(v * v for v in profile.values()))
        assert abs(n - 1) <= TOL, "-replace: column %d has norm %g" % (c, n)


def test_nokissing(binary, tmp, rng):
    # Column 1 in two blobs that do not touch: x in 0..1 and x in 4..5.
    data, _, layers, _ = setup(rng)
    columns = [1 if x <= 1 or x >= 4 else 0 for x, y, z in coords()]
    whole = [c if l else 0 for c, l in zip(columns, layers)]
    pieces = [(1 if x <= 1 else 2) if g else 0
              for (x, y, z), g in zip(coords(), whole)]

    out = run(binary, tmp, data, columns, layers)
    check(out, expected(data, whole, layers), "no -NoKissing")
    out = run(binary, tmp, data, columns, layers, "-NoKissing")
    check(out, expected(data, pieces, layers), "-NoKissing")

    # A bridge of column 1 at x = 2..3 in z = 0, which has no layer label,
    # connects the two blobs, so -NoKissing keeps them as one column.
    bridged = [1 if x <= 1 or x >= 4 or z == 0 else 0 for x, y, z in coords()]
    out = run(binary, tmp, data, bridged, layers, "-NoKissing")
    check(out, expected(data, whole, layers), "-NoKissing with bridge")


def test_4d(binary, tmp, rng):
    # Volume 2 = 3 x volume 1: each volume is normalized on its own.
    vol1, columns, layers, groups = setup(rng)
    vol2 = [3 * v for v in vol1]
    out = run(binary, tmp, vol1 + vol2, columns, layers, nt=2)
    nvox = len(vol1)
    check(out[:nvox], expected(vol1, groups, layers), "4D volume 1")
    check(out[nvox:], expected(vol2, groups, layers), "4D volume 2")


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    binary = os.path.abspath(sys.argv[1])
    rng = random.Random(0)
    tests = [test_default_and_mask, test_replace, test_nokissing, test_4d]
    with tempfile.TemporaryDirectory() as tmp:
        for test in tests:
            test(binary, tmp, rng)
            print("ok    %s" % test.__name__)
    print("test_LN2_COLUMN_NORMALIZE: all %d tests passed" % len(tests))


if __name__ == "__main__":
    main()

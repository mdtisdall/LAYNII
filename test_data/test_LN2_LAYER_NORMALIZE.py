#!/usr/bin/env python3
"""Correctness tests for LN2_LAYER_NORMALIZE on small synthetic images.

Usage: python3 test_data/test_LN2_LAYER_NORMALIZE.py ./LN2_LAYER_NORMALIZE

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


def run(binary, tmp, data, labels, *flags, nt=1):
    f_in = os.path.join(tmp, "input.nii")
    f_layer = os.path.join(tmp, "layers.nii")
    f_out = os.path.join(tmp, "output.nii")
    write_nifti(f_in, data, nt)
    write_nifti(f_layer, labels)
    subprocess.run([binary, "-input", f_in, "-layer_file", f_layer,
                    "-output", f_out, *flags],
                   check=True, stdout=subprocess.DEVNULL)
    return read_nifti(f_out)


def norm(values):
    return math.sqrt(sum(v * v for v in values))


def close(a, b):
    return abs(a - b) <= TOL * max(1.0, abs(b))


def check_unit_norm(out, groups, name):
    for g, idx in groups.items():
        n = norm(out[i] for i in idx)
        assert close(n, 1.0), "%s: group %s has norm %g" % (name, g, n)


def coords():
    for z in range(N):
        for y in range(N):
            for x in range(N):
                yield x, y, z


def test_unit_norm_and_mask(binary, tmp, rng):
    # Two column labels (x < 3 and x >= 3) for z in 1..4; label 0 elsewhere.
    labels = [(1 if x < 3 else 2) if 1 <= z <= 4 else 0 for x, y, z in coords()]
    data = [rng.uniform(1, 10) for _ in labels]
    groups = {l: [i for i, v in enumerate(labels) if v == l] for l in (1, 2)}

    out = run(binary, tmp, data, labels)
    check_unit_norm(out, groups, "unit norm")
    for l, idx in groups.items():
        n = norm(data[i] for i in idx)
        for i in idx:
            assert close(out[i], data[i] / n), "unit norm: voxel %d" % i
    for i, l in enumerate(labels):
        if l == 0:
            assert close(out[i], data[i]), "unit norm: background voxel %d changed" % i

    out = run(binary, tmp, data, labels, "-mask")
    check_unit_norm(out, groups, "-mask")
    for i, l in enumerate(labels):
        if l == 0:
            assert out[i] == 0, "-mask: background voxel %d is %g" % (i, out[i])


def test_nokissing(binary, tmp, rng):
    # One label in two blobs that do not touch: x in 0..1 and x in 4..5.
    labels = [1 if x <= 1 or x >= 4 else 0 for x, y, z in coords()]
    data = [rng.uniform(1, 10) for _ in labels]
    x_of = [x for x, y, z in coords()]
    blobs = {
        "left": [i for i, l in enumerate(labels) if l and x_of[i] <= 1],
        "right": [i for i, l in enumerate(labels) if l and x_of[i] >= 4],
    }

    out = run(binary, tmp, data, labels)
    check_unit_norm(out, {"both": blobs["left"] + blobs["right"]}, "no -NoKissing")

    out = run(binary, tmp, data, labels, "-NoKissing")
    check_unit_norm(out, blobs, "-NoKissing")


def test_4d(binary, tmp, rng):
    # Volume 2 = 3 x volume 1: normalized volumes must be equal.
    labels = [1 if y < 3 else 2 for x, y, z in coords()]
    vol1 = [rng.uniform(1, 10) for _ in labels]
    data = vol1 + [3 * v for v in vol1]
    nvox = len(labels)
    groups = {l: [i for i, v in enumerate(labels) if v == l] for l in (1, 2)}

    out = run(binary, tmp, data, labels, nt=2)
    out1, out2 = out[:nvox], out[nvox:]
    check_unit_norm(out1, groups, "4D volume 1")
    check_unit_norm(out2, groups, "4D volume 2")
    for i in range(nvox):
        assert close(out1[i], out2[i]), "4D: voxel %d differs between volumes" % i


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    binary = os.path.abspath(sys.argv[1])
    rng = random.Random(0)
    tests = [test_unit_norm_and_mask, test_nokissing, test_4d]
    with tempfile.TemporaryDirectory() as tmp:
        for test in tests:
            test(binary, tmp, rng)
            print("ok    %s" % test.__name__)
    print("test_LN2_LAYER_NORMALIZE: all %d tests passed" % len(tests))


if __name__ == "__main__":
    main()

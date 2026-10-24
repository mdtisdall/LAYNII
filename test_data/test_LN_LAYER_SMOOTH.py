#!/usr/bin/env python3
"""Correctness tests for negative layers in LN_LAYER_SMOOTH on small synthetic images.

Usage: python3 test_data/test_LN_LAYER_SMOOTH.py ./LN_LAYER_SMOOTH

Uses only the Python standard library: images are written as float32 NIfTI-1 and
outputs are read as NIfTI-1 or NIfTI-2. A negative layer must be smoothed like any
other layer: giving each negative layer -k an unused positive label instead must give
the same output in every voxel, with and without -NoKissing and -mask.
"""
import math
import os
import random
import struct
import subprocess
import sys
import tempfile

DIMS = (12, 10, 8)


def write_nifti(path, data):
    """Write a list of floats (x fastest, then y, z) as a float32 NIfTI-1."""
    hdr = bytearray(348)
    struct.pack_into("<i", hdr, 0, 348)
    struct.pack_into("<8h", hdr, 40, 3, *DIMS, 1, 1, 1, 1)
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
        nvox = math.prod(struct.unpack_from("<7h", raw, 42)[:3])
        offset = int(struct.unpack_from("<f", raw, 108)[0])
    else:
        dtype = struct.unpack_from("<h", raw, 12)[0]
        nvox = math.prod(struct.unpack_from("<7q", raw, 24)[:3])
        offset = struct.unpack_from("<q", raw, 168)[0]
    assert dtype == 16, "expected float32 output, got datatype %d" % dtype
    return list(struct.unpack_from("<%df" % nvox, raw, offset))


def run(binary, tmp, data, layers, *flags):
    f_in, f_lay, f_out = (os.path.join(tmp, n) for n in ("input.nii", "layers.nii", "output.nii"))
    write_nifti(f_in, data)
    write_nifti(f_lay, layers)
    subprocess.run([binary, "-input", f_in, "-layer_file", f_lay, "-FWHM", "1.5", "-output", f_out, *flags],
                   check=True, stdout=subprocess.DEVNULL)
    return read_nifti(f_out)


def main():
    binary = os.path.abspath(sys.argv[1])
    rng = random.Random(1)
    nx, ny, nz = DIMS
    # Layers along z: -3 -2 -1 (WM) 0 (a gap) 1 2 3 4, and 0 at the x edges (outside)
    by_z = [-3, -2, -1, 0, 1, 2, 3, 4]
    layers = [0.0 if x in (0, nx - 1) else float(by_z[z]) for z in range(nz) for y in range(ny) for x in range(nx)]
    relabeled = [10.0 - v if v < 0 else v for v in layers]  # -k -> 10 + k, unused positive labels
    data = [rng.uniform(0, 100) for _ in layers]
    failures = []
    for flags in ([], ["-NoKissing"], ["-mask"]):
        with tempfile.TemporaryDirectory() as tmp:
            a = run(binary, tmp, data, layers, *flags)
            b = run(binary, tmp, data, relabeled, *flags)
        diff = max(abs(p - q) for p, q in zip(a, b))
        if diff > 1e-4:
            failures.append("%s: negative layers differ from the same layers with positive labels (max %g)"
                            % (" ".join(flags) or "default", diff))
        neg = [i for i, v in enumerate(layers) if v < 0]
        if all(abs(a[i] - data[i]) < 1e-6 for i in neg):
            failures.append("%s: negative layers were not smoothed" % (" ".join(flags) or "default"))
    if failures:
        print("\n".join(failures))
        sys.exit(1)
    print("All LN_LAYER_SMOOTH negative-layer tests passed")


if __name__ == "__main__":
    main()

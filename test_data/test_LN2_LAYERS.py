#!/usr/bin/env python3
"""Correctness tests for the -extend_wm option of LN2_LAYERS on small synthetic rims.

Usage: python3 test_data/test_LN2_LAYERS.py ./LN2_LAYERS

Uses only the Python standard library: rims are written as float32 NIfTI-1 with
1 mm voxels, and outputs are read as NIfTI-1 or NIfTI-2 (int16 or float32). The
geometries make every distance and thickness a whole number of mm and every
x / T exact in binary, so the expected labels are exact.
"""
import math
import os
import struct
import subprocess
import sys
import tempfile

OUTPUTS = ("layers_equidist", "metric_equidist", "layers_equivol", "metric_equivol")


def write_nifti(path, data, dims):
    """Write a list of floats (x fastest, then y, z) as a float32 NIfTI-1."""
    hdr = bytearray(348)
    struct.pack_into("<i", hdr, 0, 348)
    struct.pack_into("<8h", hdr, 40, 3, *dims, 1, 1, 1, 1)
    struct.pack_into("<hh", hdr, 70, 16, 32)  # NIFTI_TYPE_FLOAT32, bitpix
    struct.pack_into("<8f", hdr, 76, 1, 1, 1, 1, 1, 1, 1, 1)
    struct.pack_into("<fff", hdr, 108, 352, 1, 0)  # vox_offset, scl_slope, scl_inter
    hdr[344:348] = b"n+1\0"
    with open(path, "wb") as f:
        f.write(hdr + bytes(4) + struct.pack("<%df" % len(data), *data))


def read_nifti(path):
    """Read an int16 or float32 NIfTI-1 or NIfTI-2 file as a list of numbers."""
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
    fmt = {4: "h", 16: "f"}[dtype]
    return list(struct.unpack_from("<%d%s" % (nvox, fmt), raw, offset))


def run(binary, tmp, rim, dims, nr_layers, *flags):
    """Run LN2_LAYERS -equivol on a rim; return {output name: values}."""
    f_rim = os.path.join(tmp, "rim.nii")
    prefix = os.path.join(tmp, "out")
    write_nifti(f_rim, rim, dims)
    subprocess.run([binary, "-rim", f_rim, "-nr_layers", str(nr_layers), "-equivol",
                    "-output", prefix, *flags], check=True, stdout=subprocess.DEVNULL)
    out = {}
    for name in OUTPUTS:
        path = "%s_%s.nii" % (prefix, name)
        out[name] = read_nifti(path if os.path.exists(path) else path + ".gz")
    return out


def volume(dims, label_of):
    nx, ny, nz = dims
    return [label_of(x, y, z) for z in range(nz) for y in range(ny) for x in range(nx)]


def check_unchanged(rim, base, ext, failures, name):
    """Outputs of voxels not coded 2 must be identical with and without -extend_wm."""
    for out in OUTPUTS:
        diff = [i for i, r in enumerate(rim) if r != 2 and base[out][i] != ext[out][i]]
        if diff:
            failures.append("%s: %s changed in %d voxels not coded 2" % (name, out, len(diff)))
        if any(base[out][i] != 0 for i, r in enumerate(rim) if r == 2):
            failures.append("%s: %s is not 0 in voxels coded 2 without -extend_wm" % (name, out))


def check_wm(rim, dims, ext, expected, failures, name):
    """expected(x, y, z) -> (layer, metric) for voxels coded 2."""
    nx, ny, _ = dims
    for i, r in enumerate(rim):
        if r != 2:
            continue
        x, y, z = i % nx, (i // nx) % ny, i // (nx * ny)
        layer, metric = expected(x, y, z)
        for kind in ("equidist", "equivol"):
            got = (ext["layers_" + kind][i], ext["metric_" + kind][i])
            if got != (layer, metric):
                failures.append("%s: voxel (%d, %d, %d) %s got %s, expected %s"
                                % (name, x, y, z, kind, got, (layer, metric)))
                return


def test_slab(binary, tmp, failures):
    # z 0-5 WM (2), z 6-12 GM (3), z 13-15 outside (1). GM voxel at z: d1 = z - 5,
    # d2 = 13 - z, so T = 8 everywhere; with 8 layers, WM voxel at z (x = 6 - z) gets
    # layer -x and metric -x / 8 while x <= 3, and 0 beyond.
    dims = (6, 6, 16)
    rim = volume(dims, lambda x, y, z: 2 if z <= 5 else 3 if z <= 12 else 1)
    base = run(binary, tmp, rim, dims, 8)
    ext = run(binary, tmp, rim, dims, 8, "-extend_wm", "3")
    check_unchanged(rim, base, ext, failures, "slab")

    def expected(x, y, z):
        d = 6 - z
        return (-d, -d / 8) if d <= 3 else (0, 0.0)
    check_wm(rim, dims, ext, expected, failures, "slab")


def test_narrow_gyrus(binary, tmp, failures):
    # Along x: outside 0-1, GM 2-4, WM 5-7, GM 8-10, outside 11-12. GM next to WM has
    # d1 = 1, d2 = 3, so T = 4; with 4 layers a WM voxel at distance x from its nearest
    # bank gets layer -x. x = 5 and x = 7 are 1 from one bank each; x = 6 is 2 from both
    # and goes to the first in z, y, x order (the bank at x = 4).
    dims = (13, 5, 5)
    rim = volume(dims, lambda x, y, z: 1 if x <= 1 or x >= 11 else 2 if 5 <= x <= 7 else 3)
    base = run(binary, tmp, rim, dims, 4)
    ext = run(binary, tmp, rim, dims, 4, "-extend_wm", "3")
    check_unchanged(rim, base, ext, failures, "narrow gyrus")
    check_wm(rim, dims, ext, lambda x, y, z: {5: (-1, -0.25), 6: (-2, -0.5), 7: (-1, -0.25)}[x],
             failures, "narrow gyrus")


def test_errors(binary, tmp, failures):
    dims = (6, 6, 16)
    f_rim = os.path.join(tmp, "rim.nii")
    write_nifti(f_rim, volume(dims, lambda x, y, z: 2 if z <= 5 else 3 if z <= 12 else 1), dims)
    for flags in (["-extend_wm", "1", "-incl_borders"], ["-extend_wm", "0"], ["-extend_wm", "-1"]):
        res = subprocess.run([binary, "-rim", f_rim, *flags], stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL)
        if res.returncode == 0:
            failures.append("errors: %s did not fail" % " ".join(flags))


def main():
    binary = os.path.abspath(sys.argv[1])
    failures = []
    for test in (test_slab, test_narrow_gyrus, test_errors):
        with tempfile.TemporaryDirectory() as tmp:
            test(binary, tmp, failures)
    if failures:
        print("\n".join(failures))
        sys.exit(1)
    print("All LN2_LAYERS -extend_wm tests passed")


if __name__ == "__main__":
    main()

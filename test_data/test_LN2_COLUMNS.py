#!/usr/bin/env python3
"""Correctness tests for the -extend_wm option of LN2_COLUMNS on a small synthetic rim.

Usage: python3 test_data/test_LN2_COLUMNS.py ./LN2_COLUMNS

Uses only the Python standard library: inputs are written as float32 NIfTI-1 and
outputs are read as NIfTI-1 or NIfTI-2 (int32 or float32). The test does not depend
on where LN2_COLUMNS puts its columns: it checks the extension against an independent
implementation of the rule (nearest voxel coded 3 within the distance, ties in z, y, x
order) applied to the columns of the run without -extend_wm.
"""
import math
import os
import struct
import subprocess
import sys
import tempfile

VOX = (1.0, 1.0, 1.0)


def write_nifti(path, data, dims):
    """Write a list of floats (x fastest, then y, z) as a float32 NIfTI-1."""
    hdr = bytearray(348)
    struct.pack_into("<i", hdr, 0, 348)
    struct.pack_into("<8h", hdr, 40, 3, *dims, 1, 1, 1, 1)
    struct.pack_into("<hh", hdr, 70, 16, 32)  # NIFTI_TYPE_FLOAT32, bitpix
    struct.pack_into("<8f", hdr, 76, 1, *VOX, 1, 1, 1, 1)
    struct.pack_into("<fff", hdr, 108, 352, 1, 0)  # vox_offset, scl_slope, scl_inter
    hdr[344:348] = b"n+1\0"
    with open(path, "wb") as f:
        f.write(hdr + bytes(4) + struct.pack("<%df" % len(data), *data))


def read_nifti(path):
    """Read an int16, int32 or float32 NIfTI-1 or NIfTI-2 file as a list of numbers."""
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
    fmt = {4: "h", 8: "i", 16: "f"}[dtype]
    return list(struct.unpack_from("<%d%s" % (nvox, fmt), raw, offset))


def volume(dims, value_of):
    nx, ny, nz = dims
    return [value_of(x, y, z) for z in range(nz) for y in range(ny) for x in range(nx)]


def run(binary, tmp, rim, midgm, dims, nr_columns, *flags):
    f_rim, f_mid = os.path.join(tmp, "rim.nii"), os.path.join(tmp, "midgm.nii")
    prefix = os.path.join(tmp, "out")
    write_nifti(f_rim, rim, dims)
    write_nifti(f_mid, midgm, dims)
    subprocess.run([binary, "-rim", f_rim, "-midgm", f_mid, "-nr_columns", str(nr_columns),
                    "-output", prefix, *flags], check=True, stdout=subprocess.DEVNULL)
    path = "%s_columns%d.nii" % (prefix, nr_columns)
    return read_nifti(path if os.path.exists(path) else path + ".gz")


def expected_extension(rim, columns, dims, dist):
    """Column of the nearest voxel coded 3 within dist (ties: z, y, x order) for voxels coded 2."""
    nx, ny, nz = dims
    r = [int(math.floor(dist / v)) for v in VOX]
    offsets = []
    for dz in range(-r[2], r[2] + 1):
        for dy in range(-r[1], r[1] + 1):
            for dx in range(-r[0], r[0] + 1):
                d = math.sqrt((dx * VOX[0]) ** 2 + (dy * VOX[1]) ** 2 + (dz * VOX[2]) ** 2)
                if 0 < d <= dist:
                    offsets.append((d, dx, dy, dz))
    offsets.sort(key=lambda o: o[0])  # stable: keeps z, y, x order for ties
    out = list(columns)
    for i, label in enumerate(rim):
        if label != 2:
            continue
        x, y, z = i % nx, (i // nx) % ny, i // (nx * ny)
        for _, dx, dy, dz in offsets:
            jx, jy, jz = x + dx, y + dy, z + dz
            if 0 <= jx < nx and 0 <= jy < ny and 0 <= jz < nz:
                j = jx + nx * (jy + ny * jz)
                if rim[j] == 3:
                    out[i] = columns[j]
                    break
    return out


def test_narrow_gyrus(binary, tmp, failures):
    # Along x: outside 0-1, GM 2-4, WM 5-9, GM 10-12, outside 13-14; mid-GM at x = 3 and 11.
    # With -extend_wm 2, WM at x = 5, 6 and 8, 9 is within 2 mm of one bank; x = 7 is 3 mm
    # from both and stays 0.
    dims = (15, 8, 6)
    rim = volume(dims, lambda x, y, z: 1 if x <= 1 or x >= 13 else 2 if 5 <= x <= 9 else 3)
    midgm = volume(dims, lambda x, y, z: 1 if x in (3, 11) else 0)
    base = run(binary, tmp, rim, midgm, dims, 6)
    ext = run(binary, tmp, rim, midgm, dims, 6, "-extend_wm", "2")
    want = expected_extension(rim, base, dims, 2.0)
    if any(rim[i] == 3 and ext[i] != base[i] for i in range(len(rim))):
        failures.append("narrow gyrus: columns of voxels coded 3 changed")
    if any(rim[i] == 2 and base[i] != 0 for i in range(len(rim))):
        failures.append("narrow gyrus: voxels coded 2 have columns without -extend_wm")
    bad = [i for i in range(len(rim)) if rim[i] == 2 and ext[i] != want[i]]
    if bad:
        i = bad[0]
        failures.append("narrow gyrus: %d voxels coded 2 differ from the rule, first (%d, %d, %d): got %d, expected %d"
                        % (len(bad), i % dims[0], (i // dims[0]) % dims[1], i // (dims[0] * dims[1]), ext[i], want[i]))
    if not any(rim[i] == 2 and ext[i] != 0 for i in range(len(rim))):
        failures.append("narrow gyrus: no voxel coded 2 got a column")
    if any(ext[i] != 0 for i in range(len(rim)) if i % dims[0] == 7):
        failures.append("narrow gyrus: voxels beyond -extend_wm got a column")


def test_errors(binary, tmp, failures):
    dims = (15, 8, 6)
    f_rim, f_mid = os.path.join(tmp, "rim.nii"), os.path.join(tmp, "midgm.nii")
    write_nifti(f_rim, volume(dims, lambda x, y, z: 1 if x <= 1 or x >= 13 else 2 if 5 <= x <= 9 else 3), dims)
    write_nifti(f_mid, volume(dims, lambda x, y, z: 1 if x in (3, 11) else 0), dims)
    for flags in (["-extend_wm", "1", "-incl_borders"], ["-extend_wm", "0"], ["-extend_wm", "-1"]):
        res = subprocess.run([binary, "-rim", f_rim, "-midgm", f_mid, "-nr_columns", "4", *flags],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if res.returncode == 0:
            failures.append("errors: %s did not fail" % " ".join(flags))


def main():
    binary = os.path.abspath(sys.argv[1])
    failures = []
    for test in (test_narrow_gyrus, test_errors):
        with tempfile.TemporaryDirectory() as tmp:
            test(binary, tmp, failures)
    if failures:
        print("\n".join(failures))
        sys.exit(1)
    print("All LN2_COLUMNS -extend_wm tests passed")


if __name__ == "__main__":
    main()

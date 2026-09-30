#!/usr/bin/env python3
"""
Convert an ILL neutron-imaging project folder (TIFF + scan_information.xml)
into ONE HDF5 file in the layout the software's
`generic_parallel_synchrotron_hdf5` plugin imports, optionally binning
projections, flat fields and dark fields on the way (same factor for all
three, so normalization stays valid).

Input folder (the ILL convention, see data_import/vendors/ill_neutrons.py):
    Proj/proj_*.tif          projections, one file per frame
    FlatsNeutrons.tif        flat fields (one image or a multi-page stack)
    DarksNeutrons.tif        dark fields (one image or a multi-page stack)
    scan_information.xml     scanInformation metadata (optional here)

Output HDF5 (streamed frame by frame -- the stack is never held in RAM):
    /entry/instrument/detector/data          projections  (N, H/b, W/b)
    /entry/instrument/detector/flat_fields   flat fields  (F, H/b, W/b)
    /entry/instrument/detector/dark_fields   dark fields  (D, H/b, W/b)
    /entry/instrument/sample/rotation_angle  degrees (only if angles given)
    /entry/instrument/detector/pixel_size    micrometres, already * bin
    /entry/instrument/source/scan_information_xml   raw XML, for provenance
    root attrs: beamline, converted_from, bin_y, bin_x, bin_method, ...

Angles: NOT given by the ILL project, so by default none are written and
the importer leaves them missing (fill them in later, e.g. in the
completion stage). Optionally:
  --angles-file F   real per-frame angles (text/CSV/.npy, one per projection)
  --angle-span S    ASSUMED uniform sweep of S degrees over the N projections,
                    from --angle-start (default 0); endpoint excluded unless
                    --endpoint. The file is tagged angles_source="assumed_uniform".

Binning: `--bin 4` bins 4x4; `--bin-y`/`--bin-x` override per axis.
`--method mean` (default) or `sum`. If the image size is not divisible by
the factor, a CENTRED crop is applied (keeps the centre of rotation, which
is measured from the image centre, unchanged) and a warning is printed.

Usage:
    python ill_to_hdf5.py PROJECT_DIR OUT.h5
    python ill_to_hdf5.py PROJECT_DIR OUT.h5 --bin 4
    python ill_to_hdf5.py PROJECT_DIR OUT.h5 --bin 2 --angle-span 360
    python ill_to_hdf5.py PROJECT_DIR --info          # inspect only, write nothing

Requires: h5py, numpy, tifffile (optional: tqdm for a progress bar).
"""
from __future__ import annotations

import argparse
import re
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

try:
    from tqdm import tqdm
except ImportError:  # progress bar is optional
    tqdm = None

PROJECTIONS = "/entry/instrument/detector/data"
FLAT_FIELDS = "/entry/instrument/detector/flat_fields"
DARK_FIELDS = "/entry/instrument/detector/dark_fields"
ANGLES = "/entry/instrument/sample/rotation_angle"
PIXEL_SIZE = "/entry/instrument/detector/pixel_size"
XML_COPY = "/entry/instrument/source/scan_information_xml"
BEAMLINE_TAG = "generic_parallel_synchrotron"


# --------------------------------------------------------------------------
# Binning
# --------------------------------------------------------------------------

def crop_offsets(size: int, factor: int) -> tuple[int, int]:
    """(start, stop) of the centred crop making `size` divisible by `factor`."""
    rem = size % factor
    start = rem // 2
    return start, size - (rem - start)


def bin_frames(frames: np.ndarray, bin_y: int = 1, bin_x: int = 1, method: str = "mean") -> np.ndarray:
    """
    Bin a (H, W) image or (N, H, W) stack by (bin_y, bin_x). Returns float32
    for "mean", float64-accumulated then float32 for "sum". Non-divisible
    sizes are centre-cropped first (see crop_offsets). Pure function.
    """
    if method not in ("mean", "sum"):
        raise ValueError(f"unknown binning method: {method!r}")
    single = frames.ndim == 2
    if single:
        frames = frames[None]
    n, h, w = frames.shape
    y0, y1 = crop_offsets(h, bin_y)
    x0, x1 = crop_offsets(w, bin_x)
    cropped = frames[:, y0:y1, x0:x1]
    hb, wb = (y1 - y0) // bin_y, (x1 - x0) // bin_x
    if hb == 0 or wb == 0:
        raise ValueError(f"bin factor ({bin_y}, {bin_x}) is larger than image size ({h}, {w})")
    reshaped = cropped.reshape(n, hb, bin_y, wb, bin_x)
    acc = reshaped.sum(axis=(2, 4), dtype=np.float64)
    if method == "mean":
        acc /= bin_y * bin_x
    out = acc.astype(np.float32)
    return out[0] if single else out


def binned_shape(h: int, w: int, bin_y: int, bin_x: int) -> tuple[int, int]:
    y0, y1 = crop_offsets(h, bin_y)
    x0, x1 = crop_offsets(w, bin_x)
    return (y1 - y0) // bin_y, (x1 - x0) // bin_x


def _to_dtype(arr: np.ndarray, dtype: np.dtype) -> np.ndarray:
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        arr = np.clip(np.rint(arr), info.min, info.max)
    return arr.astype(dtype)


# --------------------------------------------------------------------------
# Reading the ILL folder
# --------------------------------------------------------------------------

def natural_key(path: Path):
    """proj_2 < proj_10 (plain lexicographic sort would reverse them)."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", path.name)]


def _find_one(folder: Path, names: tuple[str, ...]) -> Path | None:
    for name in names:
        p = folder / name
        if p.is_file():
            return p
    return None


class FrameSource:
    """A list of frames, from many TIFF files and/or multi-page TIFFs; yields (H, W) arrays lazily."""

    def __init__(self, files: list[Path]):
        import tifffile

        self._tifffile = tifffile
        self.files = files
        self._counts = []
        self.shape2d = None
        self.dtype = None
        for f in files:
            with tifffile.TiffFile(f) as tf:
                # count PAGES, not series shape: a (3, H, W) stack can be misread as RGB planes
                count, frame_shape = len(tf.pages), tuple(tf.pages[0].shape)
                if len(frame_shape) != 2:
                    raise ValueError(f"{f.name}: unsupported page shape {frame_shape} (expected 2D grayscale)")
                dtype = tf.pages[0].dtype
                if self.shape2d is None:
                    self.shape2d, self.dtype = frame_shape, dtype
                elif frame_shape != self.shape2d:
                    raise ValueError(f"{f.name}: frame shape {frame_shape} != {self.shape2d}")
                self._counts.append(count)
        self.count = sum(self._counts)

    def __len__(self):
        return self.count

    def frames(self, batch: int = 16):
        """Yield (n, H, W) batches in order."""
        buf = []
        for f in self.files:
            with self._tifffile.TiffFile(f) as tf:
                for page in tf.pages:
                    buf.append(page.asarray())
                    if len(buf) >= batch:
                        yield np.stack(buf)
                        buf = []
        if buf:
            yield np.stack(buf)


def parse_scan_xml(path: Path | None) -> dict:
    """Lenient XML read: pixel size (um), declared image count, all attributes for the report."""
    info = {"pixel_size_um": None, "declared_projections": None, "attributes": [], "raw": None}
    if path is None:
        return info
    info["raw"] = path.read_text(encoding="utf-8", errors="replace")
    root = ET.fromstring(info["raw"])
    info["attributes"] = [(n.tag, dict(n.attrib)) for n in root.iter()]

    def num(node, attr):
        try:
            return float(node.get(attr)) if node is not None and node.get(attr) else None
        except ValueError:
            return None

    imager = root.find("imager")
    pixel_mm = num(imager, "pixelSize_mm")
    if pixel_mm:
        info["pixel_size_um"] = pixel_mm * 1000.0
    else:
        recon = root.find("reconstruction")
        text = recon.get("voxelSize_mm") if recon is not None else None
        if text:
            vals = [float(v) for v in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)]
            if vals and vals[0] > 0:
                info["pixel_size_um"] = vals[0] * 1000.0
    declared = num(root.find("acquisition"), "numberOfProjections")
    info["declared_projections"] = int(declared) if declared else None
    return info


def load_angles(spec, n, start, span, endpoint):
    """Angles if the user supplied them, else None (nothing is ever assumed)."""
    if spec and span is not None:
        raise ValueError("give either --angles-file or --angle-span, not both")
    if spec:
        p = Path(spec)
        angles = np.load(p) if p.suffix == ".npy" else np.loadtxt(p, delimiter="," if p.suffix == ".csv" else None)
        angles = np.asarray(angles, dtype=np.float64).ravel()
        if len(angles) != n:
            raise ValueError(f"--angles-file has {len(angles)} values but there are {n} projections")
        return angles
    if span is None:
        return None
    return start + np.linspace(0.0, span, n, endpoint=endpoint)


# --------------------------------------------------------------------------
# Conversion
# --------------------------------------------------------------------------

def _say(msg, quiet):
    if not quiet:
        print(msg)


def _stream(h5, name, source, bin_y, bin_x, method, out_dtype, compression, batch, label, quiet):
    hb, wb = binned_shape(*source.shape2d, bin_y, bin_x)
    ds = h5.create_dataset(
        name, shape=(len(source), hb, wb), dtype=out_dtype, chunks=(1, hb, wb),
        compression=compression[0], compression_opts=compression[1],
    )
    it = source.frames(batch)
    bar = tqdm(total=len(source), desc=label, unit="fr") if (tqdm and not quiet) else None
    pos = 0
    for block in it:
        if bin_y == 1 and bin_x == 1:
            out = block
        else:
            out = bin_frames(block, bin_y, bin_x, method)
        ds[pos:pos + len(block)] = _to_dtype(out, out_dtype)
        pos += len(block)
        if bar:
            bar.update(len(block))
    if bar:
        bar.close()
    return ds


def _resolve_dtype(src_dtype: np.dtype, method: str, requested: str) -> np.dtype:
    if requested != "auto":
        return np.dtype(requested)
    if np.issubdtype(src_dtype, np.floating):
        return np.dtype(src_dtype) if np.dtype(src_dtype).itemsize >= 4 else np.dtype(np.float32)
    # integers: mean stays in range; a sum of integers can overflow -> float32
    return np.dtype(src_dtype) if method == "mean" else np.dtype(np.float32)


def inspect_project(folder: Path) -> dict:
    """Locate and describe the ILL folder's contents without reading pixel data."""
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"not a folder: {folder}")
    proj_dir = folder / "Proj"
    proj_files = sorted(
        (p for p in proj_dir.iterdir() if p.is_file() and p.suffix.lower() in (".tif", ".tiff") and p.name.lower().startswith("proj_")),
        key=natural_key,
    ) if proj_dir.is_dir() else []
    if not proj_files:
        raise FileNotFoundError(f"no Proj/proj_*.tif files under {folder}")
    flat_file = _find_one(folder, ("FlatsNeutrons.tif", "FlatsNeutrons.tiff"))
    dark_file = _find_one(folder, ("DarksNeutrons.tif", "DarksNeutrons.tiff"))
    if flat_file is None or dark_file is None:
        raise FileNotFoundError("FlatsNeutrons.tif and DarksNeutrons.tif must both be in the scan root")
    xml_path = _find_one(folder, ("scan_information.xml",))
    return {
        "projections": FrameSource(proj_files),
        "flats": FrameSource([flat_file]),
        "darks": FrameSource([dark_file]),
        "xml": parse_scan_xml(xml_path),
        "xml_path": xml_path,
    }


def print_report(info: dict, out=print):
    for key in ("projections", "flats", "darks"):
        s = info[key]
        out(f"{key:<12}: {len(s):>6} frame(s), frame shape {s.shape2d}, dtype {s.dtype}")
    xml = info["xml"]
    out(f"XML          : {info['xml_path'] or 'NOT FOUND'}")
    if xml["declared_projections"] is not None:
        out(f"  declared numberOfProjections = {xml['declared_projections']}"
            f" (found {len(info['projections'])})")
    out(f"  pixel size (um) = {xml['pixel_size_um']}")
    for tag, attrs in xml["attributes"]:
        if attrs:
            out(f"  <{tag}> {attrs}")


def convert(
    folder, output, bin_y=1, bin_x=1, method="mean", dtype="auto",
    angle_span=None, angle_start=0.0, endpoint=False, angles_file=None,
    pixel_size_um=None, compression="gzip", compression_level=4, batch=16, quiet=False,
):
    import h5py

    info = inspect_project(folder)
    proj, flats, darks = info["projections"], info["flats"], info["darks"]
    if not quiet:
        print_report(info)
    if not (proj.shape2d == flats.shape2d == darks.shape2d):
        raise ValueError(
            f"image size mismatch: projections {proj.shape2d}, flats {flats.shape2d}, darks {darks.shape2d}"
        )
    declared = info["xml"]["declared_projections"]
    if declared is not None and declared != len(proj):
        print(f"WARNING: XML declares {declared} projections but {len(proj)} files found", file=sys.stderr)

    h, w = proj.shape2d
    hb, wb = binned_shape(h, w, bin_y, bin_x)
    if (h % bin_y) or (w % bin_x):
        print(f"WARNING: {h}x{w} is not divisible by ({bin_y},{bin_x}); centre-cropping to "
              f"{hb * bin_y}x{wb * bin_x} before binning", file=sys.stderr)

    base_px = pixel_size_um if pixel_size_um is not None else info["xml"]["pixel_size_um"]
    if base_px is None:
        print("WARNING: no pixel size in XML and none given (--pixel-size-um); writing 1.0 um", file=sys.stderr)
        base_px = 1.0
    # binning changes the sampling; only isotropic binning has one scalar pixel size
    if bin_y != bin_x:
        print(f"WARNING: anisotropic binning ({bin_y},{bin_x}); pixel_size written for the X axis", file=sys.stderr)
    pixel_size = base_px * bin_x

    angles = load_angles(angles_file, len(proj), angle_start, angle_span, endpoint)
    out_dtype = _resolve_dtype(np.dtype(proj.dtype), method, dtype)
    comp = (None, None) if compression == "none" else (compression, compression_level if compression == "gzip" else None)

    _say(f"Projections {len(proj)} x {h}x{w} ({proj.dtype}) -> {len(proj)} x {hb}x{wb} ({out_dtype}), "
         f"bin=({bin_y},{bin_x}) {method}, pixel size {pixel_size:g} um", quiet)
    if angles is None:
        _say("Angles: none in the project and none supplied -> not written", quiet)
    else:
        _say(f"Angles: {angles[0]:g} .. {angles[-1]:g} deg ({len(angles)} values)", quiet)

    output = Path(output)
    t0 = time.time()
    try:
        with h5py.File(output, "w") as f:
            f.attrs["beamline"] = BEAMLINE_TAG
            f.attrs["converted_from"] = str(Path(folder).resolve())
            f.attrs["source_format"] = "ill_neutrons_tiff"
            f.attrs["bin_y"], f.attrs["bin_x"] = bin_y, bin_x
            f.attrs["bin_method"] = method
            f.attrs["original_image_shape"] = (h, w)
            f.attrs["original_pixel_size_um"] = base_px
            f.attrs["angles_source"] = "none" if angles is None else ("file" if angles_file else "assumed_uniform")
            args = (bin_y, bin_x, method, out_dtype, comp, batch)
            _stream(f, PROJECTIONS, proj, *args, "projections", quiet)
            _stream(f, FLAT_FIELDS, flats, *args, "flats", quiet)
            _stream(f, DARK_FIELDS, darks, *args, "darks", quiet)
            if angles is not None:
                f.create_dataset(ANGLES, data=angles)
                f[ANGLES].attrs["units"] = "degrees"
            f.create_dataset(PIXEL_SIZE, data=float(pixel_size))
            f[PIXEL_SIZE].attrs["units"] = "um"
            if info["xml"]["raw"] is not None:
                f.create_dataset(XML_COPY, data=info["xml"]["raw"])
    except BaseException:
        output.unlink(missing_ok=True)  # never leave a half-written file that looks valid
        raise
    _say(f"Wrote {output} ({output.stat().st_size / 1e9:.3f} GB) in {time.time() - t0:.1f}s", quiet)
    return output


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("project", help="ILL project folder (contains Proj/, FlatsNeutrons.tif, DarksNeutrons.tif)")
    p.add_argument("output", nargs="?", help="output .h5 path (omit with --info)")
    p.add_argument("--info", action="store_true", help="only print what was found; write nothing")
    p.add_argument("--bin", type=int, default=1, help="bin factor for both axes (default 1 = none)")
    p.add_argument("--bin-y", type=int, help="override bin factor along image rows")
    p.add_argument("--bin-x", type=int, help="override bin factor along image columns")
    p.add_argument("--method", choices=["mean", "sum"], default="mean")
    p.add_argument("--dtype", default="auto", help="output dtype (default: keep source dtype; float32 for sum of ints)")
    p.add_argument("--angle-span", type=float, help="optional: ASSUME a uniform sweep of this many degrees (default: write no angles)")
    p.add_argument("--angle-start", type=float, default=0.0)
    p.add_argument("--endpoint", action="store_true", help="last projection is at start+span (default: excluded)")
    p.add_argument("--angles-file", help="text/CSV/.npy with one angle per projection (overrides span/start)")
    p.add_argument("--pixel-size-um", type=float, help="unbinned detector pixel size, um (default: from XML)")
    p.add_argument("--compression", choices=["gzip", "lzf", "none"], default="gzip")
    p.add_argument("--compression-level", type=int, default=4)
    p.add_argument("--batch", type=int, default=16, help="frames processed at a time")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args(argv)

    if args.info:
        print_report(inspect_project(Path(args.project)))
        return 0
    if not args.output:
        p.error("output path is required unless --info is given")
    bin_y = args.bin_y or args.bin
    bin_x = args.bin_x or args.bin
    if bin_y < 1 or bin_x < 1:
        p.error("bin factors must be >= 1")
    out = Path(args.output)
    if out.exists() and not args.overwrite:
        p.error(f"{out} exists (use --overwrite)")
    convert(
        args.project, out, bin_y=bin_y, bin_x=bin_x, method=args.method, dtype=args.dtype,
        angle_span=args.angle_span, angle_start=args.angle_start, endpoint=args.endpoint,
        angles_file=args.angles_file, pixel_size_um=args.pixel_size_um,
        compression=args.compression, compression_level=args.compression_level, batch=args.batch,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

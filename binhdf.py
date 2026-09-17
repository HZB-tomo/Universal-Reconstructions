#!/usr/bin/env python3
"""
Bin the large image stack in an HDF5 file (e.g. CT/tomography projections)
by a spatial factor (default 8x8), to shrink file size, while copying
every other dataset/group/attribute unchanged.

v3: correctly handles NeXus-style soft links. Many beamline HDF5 writers
    store the real image data at e.g. /entry/instrument/detector/data and
    then create a SOFT LINK at /entry/data/data pointing to it (so NeXus
    tools can find "the default data"). This version:
    - detects if --dataset is a soft link, and resolves it to find the
      REAL underlying dataset path
    - bins the REAL dataset (only once - never duplicated)
    - recreates soft/external links as links (not full copies) everywhere
      in the file, so aliases stay aliases instead of turning into
      independent duplicated copies
    - reports metadata bytes copied and a final per-dataset size listing
      for diagnostics

Usage:
    python bin_hdf5_v3.py input.h5 output.h5
    python bin_hdf5_v3.py input.h5 output.h5 --bin 8 --method mean
    python bin_hdf5_v3.py input.h5 output.h5 --dataset /entry/data/data --batch 20

Requires: h5py, numpy   (pip install h5py numpy)
Optional: tqdm for a progress bar (pip install tqdm)
"""
import argparse
import os
import sys
import time
import traceback

import h5py
import numpy as np

try:
    from tqdm import tqdm
    HAVE_TQDM = True
except ImportError:
    HAVE_TQDM = False


def human_size(nbytes):
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if abs(nbytes) < 1024:
            return f"{nbytes:.2f} {unit}"
        nbytes /= 1024
    return f"{nbytes:.2f} PB"


def bin_stack(frames, bin_factor, method):
    n, h, w = frames.shape
    h_crop = (h // bin_factor) * bin_factor
    w_crop = (w // bin_factor) * bin_factor
    cropped = frames[:, :h_crop, :w_crop]

    reshaped = cropped.reshape(
        n, h_crop // bin_factor, bin_factor, w_crop // bin_factor, bin_factor
    )
    acc = reshaped.astype(np.float32)

    if method == "mean":
        result = acc.mean(axis=(2, 4))
    elif method == "sum":
        result = acc.sum(axis=(2, 4))
    elif method == "max":
        result = reshaped.max(axis=(2, 4)).astype(np.float32)
    else:
        raise ValueError(f"unknown method: {method}")
    return result


def copy_attrs(src, dst):
    for key, val in src.attrs.items():
        dst.attrs[key] = val


def resolve_real_path(fin, path, max_depth=10):
    """If `path` is a soft link (possibly chained), follow it to find the
    real (hard-linked) dataset/group path. Returns the real path string.
    If `path` is already a real object, returns `path` unchanged."""
    current = path
    for _ in range(max_depth):
        link = fin.get(current, getlink=True)
        if isinstance(link, h5py.SoftLink):
            target = link.path
            if not target.startswith("/"):
                # relative soft link - resolve relative to current's parent
                parent = current.rsplit("/", 1)[0] or "/"
                target = parent.rstrip("/") + "/" + target
            current = target
        else:
            return current
    raise RuntimeError(f"Soft link chain too deep resolving {path}")


def copy_recursive(src_group, dst_group, skip_path, current_path="", stats=None):
    """Recursively copy groups/datasets from src to dst, skipping skip_path
    (the REAL dataset path, handled separately by the caller). Soft/external
    links are recreated as links, never expanded into full copies - this is
    what prevents alias paths from becoming duplicated data."""
    if stats is None:
        stats = {"bytes_copied": 0, "datasets_copied": 0, "skipped": [],
                  "links_recreated": []}

    for key in src_group.keys():
        item_path = f"{current_path}/{key}" if current_path else key
        full_path = "/" + item_path

        link = src_group.get(key, getlink=True)

        if isinstance(link, h5py.SoftLink):
            dst_group[key] = h5py.SoftLink(link.path)
            stats["links_recreated"].append((full_path, "soft", link.path))
            continue
        if isinstance(link, h5py.ExternalLink):
            dst_group[key] = h5py.ExternalLink(link.filename, link.path)
            stats["links_recreated"].append((full_path, "external",
                                              f"{link.filename}:{link.path}"))
            continue

        # real (hard-linked) object from here on
        item = src_group[key]
        if isinstance(item, h5py.Group):
            new_group = dst_group.create_group(key)
            copy_attrs(item, new_group)
            copy_recursive(item, new_group, skip_path, item_path, stats)
        elif isinstance(item, h5py.Dataset):
            if full_path == skip_path:
                stats["skipped"].append(full_path)
                continue  # this is the real data - handled separately by caller
            dst_group.copy(item, dst_group, name=key)
            nbytes = item.size * item.dtype.itemsize
            stats["bytes_copied"] += nbytes
            stats["datasets_copied"] += 1

    copy_attrs(src_group, dst_group)
    return stats


def list_output_datasets(path, top_n=15):
    rows = []

    def visitor(name, obj):
        if isinstance(obj, h5py.Dataset):
            nbytes = obj.size * obj.dtype.itemsize
            rows.append((nbytes, "/" + name, obj.shape, str(obj.dtype)))

    with h5py.File(path, "r") as f:
        f.visititems(visitor)

    rows.sort(key=lambda r: -r[0])
    print(f"\nDatasets in output file (top {top_n} by size):")
    print(f"{'size':>12}  {'shape':<22} {'dtype':<10} path")
    total = 0
    for nbytes, name, shape, dtype in rows[:top_n]:
        print(f"{human_size(nbytes):>12}  {str(shape):<22} {dtype:<10} {name}")
    for nbytes, *_ in rows:
        total += nbytes
    print(f"\nSum of all dataset (logical, uncompressed) sizes: {human_size(total)}")
    return total


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", help="path to input .h5 file")
    parser.add_argument("output", help="path to output (binned) .h5 file")
    parser.add_argument("--dataset", default="/entry/data/data",
                         help="path of the image stack dataset to bin, alias or real "
                              "(default: /entry/data/data)")
    parser.add_argument("--bin", type=int, default=8,
                         help="binning factor per spatial axis (default: 8)")
    parser.add_argument("--method", choices=["mean", "sum", "max"], default="mean",
                         help="how to combine pixels within a bin (default: mean)")
    parser.add_argument("--batch", type=int, default=20,
                         help="number of frames to process at a time (default: 20)")
    parser.add_argument("--dtype", default="uint16",
                         help="output dtype for the binned dataset (default: uint16)")
    parser.add_argument("--compression", default="gzip", choices=["gzip", "lzf", "none"],
                         help="compression for the binned dataset (default: gzip)")
    parser.add_argument("--compression-opts", type=int, default=4,
                         help="gzip compression level 0-9 (default: 4)")
    parser.add_argument("--keep-existing-output", action="store_true",
                         help="do NOT delete an existing output file before starting")
    args = parser.parse_args()

    compression = None if args.compression == "none" else args.compression
    compression_opts = args.compression_opts if compression == "gzip" else None
    out_dtype = np.dtype(args.dtype)

    if os.path.exists(args.output):
        if args.keep_existing_output:
            print(f"WARNING: {args.output} exists; --keep-existing-output set, not deleting.")
        else:
            old_size = os.path.getsize(args.output)
            print(f"Removing pre-existing output file {args.output} "
                  f"({human_size(old_size)}) before starting...")
            os.remove(args.output)

    t0 = time.time()

    try:
        with h5py.File(args.input, "r") as fin:
            if args.dataset not in fin:
                print(f"ERROR: dataset {args.dataset} not found in {args.input}")
                sys.exit(1)

            real_path = resolve_real_path(fin, args.dataset)
            if real_path != args.dataset:
                print(f"NOTE: {args.dataset} is a soft link -> real data is at "
                      f"{real_path}. Binning the real dataset; the link will be "
                      f"recreated automatically so {args.dataset} keeps working.\n")

            src_ds = fin[real_path]
            n, h, w = src_ds.shape
            src_dtype = src_ds.dtype
            orig_bytes = src_ds.size * src_dtype.itemsize

            b = args.bin
            h_new, w_new = h // b, w // b
            if h % b or w % b:
                print(f"NOTE: {h}x{w} not evenly divisible by {b}; "
                      f"cropping to {h_new*b}x{w_new*b} before binning "
                      f"(losing {h % b} row(s), {w % b} col(s)).")

            new_bytes_est = n * h_new * w_new * out_dtype.itemsize
            print(f"Real dataset  : {real_path}  shape={src_ds.shape}  dtype={src_dtype}  "
                  f"size={human_size(orig_bytes)}")
            print(f"Binning       : {b}x{b} ({args.method}) -> new shape "
                  f"({n}, {h_new}, {w_new})  dtype={out_dtype}")
            print(f"Est. new size : ~{human_size(new_bytes_est)} (before compression)")
            print()

            with h5py.File(args.output, "w") as fout:
                print("Copying all other datasets/groups/attributes/links unchanged ...")
                stats = copy_recursive(fin, fout, skip_path=real_path)
                print(f"  -> copied {stats['datasets_copied']} metadata dataset(s), "
                      f"{human_size(stats['bytes_copied'])} total")
                print(f"  -> skipped real dataset (will be rebuilt binned): {stats['skipped']}")
                print(f"  -> recreated {len(stats['links_recreated'])} link(s) as links "
                      f"(not copies): {stats['links_recreated']}")
                if stats["bytes_copied"] > 50 * 1024 * 1024:
                    print("  !! WARNING: metadata copy is >50MB - larger than expected. "
                          "There may be another large dataset not accounted for.")
                print()

                print("Creating binned image dataset and streaming frames ...")
                chunks = (1, h_new, w_new)

                # ensure parent groups of real_path exist in output
                parent = real_path.rsplit("/", 1)[0] or "/"
                if parent not in fout:
                    fout.create_group(parent)

                out_ds = fout.create_dataset(
                    real_path,
                    shape=(n, h_new, w_new),
                    dtype=out_dtype,
                    chunks=chunks,
                    compression=compression,
                    compression_opts=compression_opts,
                )

                copy_attrs(src_ds, out_ds)
                if "NDArrayDimBinning" in out_ds.attrs:
                    old_bin = np.array(out_ds.attrs["NDArrayDimBinning"])
                    out_ds.attrs["NDArrayDimBinning"] = (old_bin * b).astype(old_bin.dtype)

                frame_range = range(0, n, args.batch)
                iterator = tqdm(frame_range, unit="batch") if HAVE_TQDM else frame_range

                for start in iterator:
                    end = min(start + args.batch, n)
                    frames = src_ds[start:end]
                    binned = bin_stack(frames, b, args.method)
                    if np.issubdtype(out_dtype, np.integer):
                        info = np.iinfo(out_dtype)
                        binned = np.clip(np.round(binned), info.min, info.max)
                    out_ds[start:end] = binned.astype(out_dtype)

                    if not HAVE_TQDM:
                        print(f"  frames {start}-{end}/{n}", end="\r")

                if not HAVE_TQDM:
                    print()

                # sanity check: the originally-requested path should now resolve
                # correctly (either it IS real_path, or a link chain leads to it)
                assert args.dataset in fout, \
                    f"{args.dataset} not reachable in output - link recreation failed"
                assert fout[args.dataset].shape == (n, h_new, w_new), \
                    "alias path does not resolve to the binned dataset - link bug"

    except Exception:
        print("\n!!! An error occurred during processing - traceback below. !!!")
        traceback.print_exc()
        if os.path.exists(args.output):
            print(f"\nThe partially-written output file {args.output} "
                  f"({human_size(os.path.getsize(args.output))}) was left on disk. "
                  f"Delete it before retrying.")
        sys.exit(1)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed/60:.1f} min.")

    out_size = os.path.getsize(args.output)
    print(f"Output file size (os.path.getsize): {human_size(out_size)}")

    total_logical = list_output_datasets(args.output)
    if total_logical > 0 and out_size > total_logical * 1.5:
        print(f"\n!! NOTE: file size ({human_size(out_size)}) is notably larger than "
              f"the sum of dataset contents ({human_size(total_logical)}). Try "
              f"`h5repack -i {args.output} -o repacked.h5` to compact it.")


if __name__ == "__main__":
    main()
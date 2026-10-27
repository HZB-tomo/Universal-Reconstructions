"""
Which HDF5 chunk shape suits a reconstructed volume? (See doc/FILE_STANDARD.md, "Chunking".)
Run on the disk you care about; use a size larger than the RAM for honest numbers:

    python benchmarks/bench_volume_chunks.py --size 512
    python benchmarks/bench_volume_chunks.py --size 2048 --dir D:/scratch --dtype uint16

For each layout it writes the same volume (z, y, x), no compression, then times
  - write: filling the volume slab by slab along z (what a reconstruction does),
  - one slice along z, y and x (median over several positions, file reopened each time),
  - scrolling: reading `--scroll` neighbouring slices one after the other, per direction,
and prints milliseconds per slice. The last column is the slowest direction, the number
to keep small. The operating system caches files, so on a volume that fits in RAM the
numbers show the chunk overhead but not the disk; the cache is not flushed here.
"""
from __future__ import annotations

import argparse
import statistics
import tempfile
import time
from pathlib import Path

import h5py
import numpy as np


def layouts(size: int) -> dict:
    """name -> chunk shape (None = contiguous), clipped to the volume."""
    def clip(shape):
        return tuple(min(c, size) for c in shape)

    return {
        "contiguous": None,
        "slice (1,N,N)": clip((1, size, size)),
        "slab (16,N,N)": clip((16, size, size)),
        "cube 16": clip((16, 16, 16)),
        "cube 32": clip((32, 32, 32)),
        "cube 64": clip((64, 64, 64)),
        "cube 128": clip((128, 128, 128)),
        "brick (64,256,64)": clip((64, 256, 64)),
        "brick (64,64,256)": clip((64, 64, 256)),
    }


def write_volume(path: Path, size: int, dtype, chunks, slab: int) -> float:
    rng = np.random.default_rng(0)
    block = rng.random((slab, size, size), dtype=np.float32)
    if np.dtype(dtype) == np.uint16:
        block = (block * 60000).astype(np.uint16)
    start = time.perf_counter()
    with h5py.File(path, "w") as f:
        dataset = f.create_dataset("/entry/volume/data", shape=(size,) * 3, dtype=dtype, chunks=chunks)
        for z in range(0, size, slab):
            n = min(slab, size - z)
            dataset[z:z + n] = block[:n]
        f.flush()
    return time.perf_counter() - start


def take(data, axis: int, index: int) -> np.ndarray:
    if axis == 0:
        return data[index]
    if axis == 1:
        return data[:, index, :]
    return data[:, :, index]


def read_slice(path: Path, axis: int, index: int, cache: int) -> float:
    start = time.perf_counter()
    with h5py.File(path, "r", rdcc_nbytes=cache, rdcc_nslots=_slots(cache)) as f:
        data = f["/entry/volume/data"]
        _ = take(data, axis, index)
    return time.perf_counter() - start


def _slots(cache: int) -> int:
    """HDF5 wants a prime number of hash slots, about 100 per chunk that fits in the cache."""
    n = max(521, cache // (1 << 20) * 100)
    while any(n % d == 0 for d in range(2, int(n ** 0.5) + 1)):
        n += 1
    return n


def scroll(path: Path, axis: int, first: int, count: int, cache: int) -> float:
    start = time.perf_counter()
    with h5py.File(path, "r", rdcc_nbytes=cache, rdcc_nslots=_slots(cache)) as f:
        data = f["/entry/volume/data"]
        for i in range(first, first + count):
            _ = take(data, axis, i)
    return (time.perf_counter() - start) / count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--size", type=int, default=256, help="volume is size^3 voxels (default 256)")
    parser.add_argument("--dtype", choices=("float32", "uint16"), default="float32")
    parser.add_argument("--dir", default=None, help="folder on the disk to test (default: the system temp folder)")
    parser.add_argument("--repeat", type=int, default=5, help="positions timed per single-slice read")
    parser.add_argument("--scroll", type=int, default=20, help="neighbouring slices per scroll test")
    parser.add_argument("--settle", type=float, default=30.0, help="seconds to wait after each write, so the disk finishes flushing it before reads are timed")
    parser.add_argument("--cache-mib", default="1,512", help="chunk cache sizes to read with, comma separated (HDF5 default: 1)")
    parser.add_argument("--slab", type=int, default=16, help="slices per write; raised to the chunk depth when that is larger, as the real writer will do")
    args = parser.parse_args()

    size = args.size
    nbytes = size ** 3 * np.dtype(args.dtype).itemsize
    print(f"volume {size}^3 {args.dtype} = {nbytes / 2**20:.0f} MiB per layout; no compression")
    rng = np.random.default_rng(1)
    caches = [int(float(c) * (1 << 20)) for c in args.cache_mib.split(",")]
    results = {c: [] for c in caches}
    writes = {}
    with tempfile.TemporaryDirectory(dir=args.dir) as tmp:
        for name, chunks in layouts(size).items():
            path = Path(tmp) / "volume.h5"
            slab = max(args.slab, chunks[0]) if chunks else args.slab
            writes[name] = write_volume(path, size, args.dtype, chunks, slab)
            time.sleep(args.settle)
            for cache in caches:
                single = []
                scrolled = []
                for axis in range(3):
                    positions = rng.integers(0, size, args.repeat)
                    single.append(statistics.median(read_slice(path, axis, int(p), cache) for p in positions) * 1000)
                    first = int(rng.integers(0, max(1, size - args.scroll)))
                    scrolled.append(scroll(path, axis, first, min(args.scroll, size), cache) * 1000)
                results[cache].append((name, writes[name], single, scrolled, path.stat().st_size / 2**20))
            path.unlink()

    for cache in caches:
        rows = results[cache]
        print(f"\nchunk cache {cache / 2**20:g} MiB")
        print(f"{'layout':<20}{'write s':>9}{'z ms':>8}{'y ms':>8}{'x ms':>8}{'worst':>8}{'scroll z':>10}{'scroll y':>10}{'scroll x':>10}{'MiB':>8}")
        for name, write_s, single, scrolled, mib in rows:
            print(f"{name:<20}{write_s:>9.2f}{single[0]:>8.1f}{single[1]:>8.1f}{single[2]:>8.1f}{max(single):>8.1f}"
                  f"{scrolled[0]:>10.1f}{scrolled[1]:>10.1f}{scrolled[2]:>10.1f}{mib:>8.0f}")
        best = min(rows, key=lambda r: max(r[2]))
        print(f"best worst-direction single slice: {best[0]} ({max(best[2]):.1f} ms)")


if __name__ == "__main__":
    main()

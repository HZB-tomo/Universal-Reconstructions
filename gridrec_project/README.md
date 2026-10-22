# urct_gridrec

Gridrec (Fourier-gridding) tomographic reconstruction in NumPy/Numba for CPU and
GPU. A package of its own: it imports nothing from URCT, and can be used without
it. URCT uses it through its `gridrec` reconstruction backend
(`reconstruction/backends/gridrec.py`). The main project's FBP backend appears
only in the benchmarks and one optional test, as the thing to compare against.

```bash
pip install -e ./gridrec_project            # from the repository root; add ".[gpu]" for CuPy
```

It is a pure-Python package (Numba compiles the hot loops on first use and caches
them), so it builds into a single `py3-none-any` wheel that works everywhere:
`pip wheel ./gridrec_project --no-deps`.

```python
from urct_gridrec import gridrec
volume = gridrec(sinograms, angles_deg, center)    # (angles, slices, width) -> (slices, width, width)
```

Conventions are the main project's: angle-first sinograms of transmission values
(`apply_log=False` for attenuation), angles in degrees, image axes (y, x), same
scaling as `algotom_fbp`, one center for all slices or one per slice (tilt).

## How it works

`src/urct_gridrec/core.py` has the algorithm and its derivation. Per pair of
slices: pad and 1D-FFT every projection, apply ramp x window and the rotation-
axis phase, spread every polar sample onto a Cartesian grid with a Kaiser-Bessel
kernel, one 2D inverse FFT, divide out the kernel's transform. Two real slices
share one complex transform.

| file | role |
|---|---|
| `kernel.py`, `filters.py` | gridding kernel and its Fourier transform; ramp x window filters |
| `core.py` | the algorithm, device independent |
| `cpu.py` | Numba. Gridding is split into bands of grid rows, one owner thread per band: no data race, no atomics, bit-for-bit reproducible |
| `gpu.py` | CuPy FFTs + a numba.cuda gridding kernel (atomic adds). Runs on numba's CUDA simulator without a GPU, which is how its logic is tested |
| `reference.py` | slow float64 NumPy version, and an exact (no gridding) Fourier sum, to measure the gridding error |

The filter has the Ram-Lak DC term (2/pi^2 in bin units). The ideal ramp is zero
at DC, which loses the mean level (about -2.5% inside objects); with the term the
amplitudes match filtered backprojection.

## CPU performance work

`python benchmarks/profile_cpu.py --width 1024` prints seconds per stage (pass
`timings={}` to `gridrec()` for the same numbers from your own code). The first
profile showed gridding was only 23% of the time; the rest was single-threaded
NumPy passes over big arrays. What changed (1024 wide, 1608 angles, 16 slices;
1.51 s -> 0.46 s):

- padding, -log and the edge fill are one parallel pass that never copies the stack;
- the filter/phase spectrum uses a running product instead of a sine+cosine per sample,
  and only half of it (the spectrum of a real slice is Hermitian);
- folding the grid margins and the FFT shift are one parallel pass;
- only the width x width part of the image is transformed along the second axis;
- deapodization, masking and the real/imaginary split are one parallel pass;
- the default grid is 1.25x the width instead of 2x (gridding error ~3e-4 with the
  6-cell kernel, versus ~2e-5; the reconstruction's own error is ~100x larger).
  Less padding also makes results agree better with FBP on low-contrast data.

Tried and not kept: a kernel width fixed at compile time (+6%), other band heights
(flat between 8 and 32 rows).

## Tests and benchmarks

```bash
python -m pytest tests                      # from this folder
python benchmarks/bench_phantom.py          # Shepp-Logan, exact projections: accuracy + speed vs FBP
python benchmarks/bench_projects.py         # real files from ../gui_samples: speed + agreement with FBP
python benchmarks/bench_phantom.py --device gpu    # on a machine with CuPy and an NVIDIA GPU
```

## Status

- Integrated into URCT as the `gridrec` backend (CPU by default).
- CPU: complete, tested against the exact Fourier sum, the reference, and FBP.
- GPU: kernel and glue tested on the CUDA simulator only. Not yet run on a real
  GPU, so its speed is unmeasured and `cupy` interplay (cuFFT, memory) is untested.

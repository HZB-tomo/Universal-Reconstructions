# Phase retrieval

Propagation-based phase-contrast tomography: what it is, what the software
does about it, and how to use it. Code: `preprocessing/phase.py` (the
method), `orchestration/preprocess.py` (how it fits into a run).

## The physics, briefly

A coherent X-ray beam (a synchrotron) is changed by a sample in two ways:
it is **absorbed** (amplitude drops, Beer-Lambert) and its **phase is
shifted** (the refractive index is `1 - delta + i*beta`; `beta` is the
absorption, `delta` the phase). For hard X-rays `delta` is typically a
hundred to a thousand times larger than `beta`, so the phase carries most of
the signal in weakly absorbing samples (soft tissue, polymers, wood).

A detector only records intensity, so the phase is invisible directly. But if
the detector sits a distance `z` behind the sample, the wave has propagated
and the phase changes turn into intensity: **bright and dark fringes along
every edge**. That is useful (edges become visible even when absorption is
tiny) and harmful (the image no longer obeys Beer-Lambert, so filtered
backprojection reconstructs halos and wrong values).

**Phase retrieval** turns the fringed image back into a projected thickness
that does obey Beer-Lambert, so ordinary reconstruction works, with much
better contrast-to-noise.

### Paganin's method

If the sample is one homogeneous material, a single linear filter does it
(Paganin et al., J. Microsc. 206, 33, 2002). With `I` the flat/dark-normalized
intensity and `T` the projected thickness:

```
exp(-mu * T) = F^-1 [ F[I] / (1 + pi * lambda * z * (delta/beta) * u^2) ]
```

`lambda` is the wavelength (`12.398 / energy[keV]` Angstrom), `z` the
sample-to-detector distance, `u` the spatial frequency, `mu = 4*pi*beta/lambda`.
That is a low-pass filter applied to every projection image. The result is
a transmission again, so the `-log` inside the reconstruction backend gives
`mu * T`.

The one material number is **`delta/beta`**. It is a tuning knob in practice:
larger values (or larger `z`) smooth more. Smoothing lowers noise and also
blurs real detail, so the right value is a compromise; the blur length is
`sqrt(lambda * z * (delta/beta) / (4 pi))` (the software reports it as the
halo it needs).

**Assumptions:** monochromatic beam, one material (or a good average
`delta/beta`), weak absorption, slowly varying phase, flat/dark-corrected
data. A sample with very different materials is not handled correctly by a
single `delta/beta`.

## Using it

### In the GUI

Refinements tab, "Phase retrieval" group: tick the box, give delta/beta (the
energy, distance and pixel size appear by themselves when the file or the
completion answers know them), and the line below the fields tells you the
smoothing length and what is still missing. With "Auto-update preview" ticked in
Basic settings, changing delta/beta previews the slice again at once, and
values you already tried come back instantly. See
[`GUI_ARCHITECTURE.md`](./GUI_ARCHITECTURE.md).

### In a run (config, INI, command line)

```ini
[preprocessing]
phase_retrieval = true
phase_delta_beta = 200          # material dependent -- no default
phase_distance_mm = 120         # blank: the 'propagation_distance' metadata field
phase_energy_kev = 25           # blank: the 'energy' metadata field (many beamline files have it)
phase_pixel_um = 1.1            # blank: the file's pixel size ('voxel_size'); a silent default of 1.0 does not count
```

Distance and energy can also be given as completion answers
(`[completion]  energy = 25`, `propagation_distance = 120`). If anything is
missing or invalid the stage stops with an error naming what to set; nothing
is guessed.

In Python: `ProjectConfig(phase_retrieval=True, phase_delta_beta=200, ...)`;
`phase_method` (default `"paganin"`) and `phase_options` (method-specific,
code-only) complete the set.

### As a function

```python
from preprocessing import retrieve_phase, phase_halo_rows

T_retrieved = retrieve_phase(T, energy_kev=25, distance_mm=120, pixel_um=1.1, delta_over_beta=200)
halo = phase_halo_rows(energy_kev=25, distance_mm=120, pixel_um=1.1, delta_over_beta=200)
```

`T` is `(angles, rows, width)` flat/dark-normalized transmission; the result
has the same shape and meaning (transmission). The input is not modified.

## Where it sits in the pipeline

```
flat/dark normalization -> zinger removal -> phase retrieval -> ring removal -> beam hardening -> reconstruction (-log inside)
```

All three places that prepare rows (a whole run, a streamed chunk, a
single-slice preview) call the same function,
`orchestration.preprocess.preprocess_rows()`.

## The halo

The filter mixes neighbouring detector rows, so a block of rows cannot be
treated on its own the way a sinogram can. `preprocess_rows()` reads
`halo` extra rows on each side of the block (`halo = 8 blur lengths`, so
about `e^-8` of the filter's weight lies outside), filters, and cuts the halo
off again. At the detector's top and bottom edge the rows are repeated
instead. The result for a block is the same as the same rows of the whole
image, to better than 0.3% of the data range (tested), which is what makes
streamed reconstruction, row ranges and live previews consistent with a
full run.

Cost: a larger `z * delta/beta` means a larger halo, so more rows are read and
filtered per chunk. The default pad of the FFT equals the halo, and
projections are processed in batches of angles bounded to about 256 MB.

## Adding another method

Register a `PhaseMethod` (`register_phase_method`), like a reconstruction
backend: give it a `name`, `halo_rows(params, **options)` and
`retrieve(projections, params, **options)`. `phase_method = "your_name"` then
selects it.

## Not covered yet

Bronnikov / modified Bronnikov, multi-distance (CTF) retrieval and GPU
execution. (Beam hardening has its own page: [`BEAM_HARDENING.md`](./BEAM_HARDENING.md).)

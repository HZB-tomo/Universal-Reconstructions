# URCT HDF5 file standard (draft 0.1)

**Status: draft 0.1; decisions D1-D7 settled (see the end).** Implemented since
software version 0.2.0: the projection-file part (`core/urct_format.py`, the
`urct_hdf5` import plugin, `convert_to_hdf5()` as the writer, `check_file()`) and the
volume file (`reconstruction/volume_file.py`; `save_volume()` and `StreamingVolumeWriter`
write it, `read_volume_info()` reads it, the GUI viewer opens it). Not implemented yet:
the phantom generators and the metadata toolbox; see `doc/BACKLOG.md`. In the volume
file the record is stored as attributes of `/entry/reconstruction` (a dictionary is a
subgroup), and the streaming writer collects slices into whole 64-slice blocks so each
chunk is written once. The chunk shape of volumes was decided
by measurement (see "Chunking").

## Goal

Convert any tomography data (TIFF stacks, vendor HDF5 files) into one
HDF5 layout that carries everything needed to reconstruct it, parallel or
cone beam, and read it back with one plugin. Two kinds of file, never mixed:

| File type | Holds | Root attribute |
|---|---|---|
| **Projection file** | measured projections, flats, darks, angles, measurement metadata | `urct_file_type = "projections"` |
| **Volume file** | a reconstructed volume, how it was made, and the measurement metadata it came from | `urct_file_type = "volume"` |

Every file has the root attributes `urct_standard = "urct-hdf5"`,
`urct_version = "0.1"` and `urct_uuid` (a UUID made when the file is
written). A reader refuses a newer major version and warns on a newer minor one.

Extra, vendor-specific or user metadata is always allowed (see "Extra
metadata"); the standard only fixes what the software depends on.

## Conventions

- Layout follows NeXus naming and paths (`/entry`, `instrument`,
  `detector`, `sample`), but we do not claim NeXus compliance. Files written
  in older layouts (for example the converter's
  `/entry/instrument/sample/rotation_angle`) are always still read: every
  older layout keeps its import plugin.
- Units are fixed per field (see tables) and also written as a `units`
  attribute on every numeric dataset, so a file explains itself.
- Angles in degrees. Detector/voxel sizes in micrometres, distances in mm,
  energy in keV.
- Axis order of projections: `(angle, row, column)`; rows run along the
  rotation axis. Axis order of volumes: `(z, y, x)`, with `z` along the
  rotation axis. Both are written in an `axes` attribute.
- Strings are UTF-8. Structured options (a backend's option dict) are
  stored as a JSON string attribute, scalars as scalar datasets.
- A field that is unknown is **absent**, never filled with a made-up value.
  Fields that were filled by a default or by the user rather than read from
  the instrument carry a `source` attribute (`file`, `user`, `default`,
  `derived`, `computed`), the same vocabulary as `core/metadata.py`.

## Projection file

### Required

| Path | Shape / type | Unit | Notes |
|---|---|---|---|
| `/entry/instrument/detector/data` | `(n_angles, rows, cols)` | counts or transmission | the measured projections. Attributes: `axes="angle,row,column"`, `image_type="raw"` or `"normalized"` |
| `/entry/geometry/geometry_type` | string | `parallel` or `cone` | the app's `geometry_type` field |
| `/entry/geometry/pixel_size` or `/entry/geometry/voxel_size` | float | um | detector pixel pitch / size at the sample. In parallel beam they are equal and a file may give just one (the reader fills in the other); cone beam needs the magnification, so give both or the distances |

### Required but may be completed later (mandatory_soft)

`urct_complete` is true when: `geometry_type` is given; `angles` or `angle_span` is given;
`voxel_size` or `pixel_size` is given; and, for cone beam, both distances are given.

These are the app's `mandatory_soft` fields: a file may lack them (the
completion stage asks the user), but a file is only **complete** when they
are all present. The root attribute `urct_complete` (`true`/`false`) says
which, so tools can warn without re-checking.

| Path | Unit | Condition |
|---|---|---|
| `/entry/sample/rotation_angle` `(n_angles,)` | degrees | or `angle_span` + count |
| `/entry/geometry/angle_span` | degrees | |
| `/entry/geometry/voxel_size` | um | pixel size at the sample (parallel: equal to the detector pixel size) |
| `/entry/geometry/source_to_detector_distance` | mm | cone only |
| `/entry/geometry/source_to_object_distance` | mm | cone only |

### Recommended

| Path | Notes |
|---|---|
| `/entry/instrument/detector/flat_fields` `(n, rows, cols)` | needed unless `image_type="normalized"` |
| `/entry/instrument/detector/dark_fields` `(n, rows, cols)` | same |
| `/entry/geometry/rotation_direction` | `cw` / `ccw`, default `ccw` |
| `/entry/geometry/center_of_rotation` | hint only, pixels. Calibration (not the file) is authoritative |
| `/entry/geometry/tilt_angle` | hint only, degrees |

### Optional

| Path | Unit |
|---|---|
| `/entry/instrument/source/energy` | keV |
| `/entry/geometry/propagation_distance` | mm (phase retrieval) |
| `/entry/instrument/name`, `/entry/operator`, `/entry/sample/name` | text |
| `/entry/start_time` | ISO 8601 timestamp |

### Provenance (written by the converter)

`/entry/provenance` (a group of attributes): `source_plugin`, `source_path`,
`source_format`, `converted_by` (software and version), `converted_on`,
`bin_y`, `bin_x`, `bin_method`, `original_image_shape`.

### Chunking and compression

Projection files are read one projection at a time (flat/dark/normalization)
and one block of rows at a time (reconstruction, streaming). Chunks of
`(1, rows_per_chunk, cols)` with about 1 MB each serve both (to be confirmed
with the same measurements as the volumes). **No compression.**

## Volume file

### The volume

| Path | Shape / type | Notes |
|---|---|---|
| `/entry/volume/data` | `(z, y, x)`, float32 or uint16 | attributes: `axes="z,y,x"`, `voxel_size` (um, cubic), `units` |

**Value units.** float32 volumes are attenuation coefficients in `1/cm`
(or `1/px` if the pixel size was unknown; the attribute `units` says which).
uint16 volumes are quantized: attributes `scale` and `offset` give
`value = offset + scale * stored`, and the clipping range is recorded
(see the reconstruction record below).

### Chunking for slicing in every direction

The viewer shows slices along z, y and x, so no axis may be slow. A chunk
that is thin in one direction (the usual `(1, y, x)` slice chunk) makes
x- and y-slices read the whole volume. **No compression, and never lossy
compression** (the values are measurements). The chunk shape is decided by
experiment, not by argument:

1. Write the same volume (for example 512, 1024 and 2048 cubed, float32 and
   uint16) with different chunk shapes: slice chunks `(1, y, x)`,
   slabs `(16, y, x)`, cubes of 32, 64 and 128, and flat bricks such as
   `(64, 256, 64)`; plus one contiguous (unchunked) layout as the baseline.
2. For each, time reading a z-, y- and x-slice (cold, after the file is
   closed and reopened, and on a file larger than the RAM so the system cache
   cannot hide the disk), scrolling through 100 neighbouring slices in each
   direction, and the write time when a reconstruction fills the volume slab by
   slab along z (the streaming writer's pattern).
3. Pick the shape that keeps the worst direction acceptable (a slice in
   under about 100 ms on the target disk) without making streaming writes
   slow. Run it on the real disks in use (local SSD, network share).

`benchmarks/bench_volume_chunks.py` holds this experiment.

**Measurement** (2048 cubed float32 = 32 GiB per layout, 31.5 GB RAM, local SSD,
no compression, default 1 MiB HDF5 chunk cache; ms to read one slice, median of 3
random positions, file reopened; writes in slabs as deep as the chunk):

| Layout | z | y | x | worst | write (s) |
|---|---|---|---|---|---|
| contiguous | 10 | 195 | 19 578 | 19 578 | 47 |
| slice `(1,N,N)` | 10 | 111 | 92 844 | 92 844 | 23 |
| slab `(16,N,N)` | 9 | 94 | 94 057 | 94 057 | 48 |
| cube 16 | 261 | 254 | 1 875 | 1 875 | 46 |
| **cube 32** | 254 | 229 | 591 | **591** | 22 |
| **cube 64** | 209 | 309 | 613 | 613 | 19 |
| cube 128 | 39 | 1 088 | 5 069 | 5 069 | 27 |
| brick `(64,256,64)` | 66 | 1 884 | 4 443 | 4 443 | 27 |
| brick `(64,64,256)` | 101 | 521 | 29 478 | 29 478 | 26 |

What it shows:

- Layouts built for z-slices (contiguous, slice, slab) read a z-slice in 10 ms but
  take 20-90 s for an x-slice: unusable for a viewer.
- **Cubes of 32 and 64 are the only layouts with no bad direction** (0.2-0.6 s for a
  slice of a 32 GiB volume) and they are as fast to write as slice chunks (about 20 s)
  when the writer fills whole chunk depths. The two are equal within the run-to-run
  noise (the same layout varied by about 40% between repeated runs).
- Cube 16 is worse in x and slower to write (too many tiny chunks); 128 and the
  bricks are poor in at least one direction.
- A bigger chunk cache (512 MiB) does **not** help single slices and made
  chunk-bigger-than-cache layouts worse (HDF5 then reads whole chunks instead of
  parts of them). For cubes it changed scrolling by a few tens of percent at most.
  So: keep the default cache.
- A cube slice reads far more than it shows (a z-slice of 16 MiB touches 1 GiB of
  chunks); that is the price of treating all directions alike.

Caveats: one machine and one disk; the file is about the size of the RAM, so the
system cache helped partly. Rerun the benchmark on other disks (network share).

**Choice for 0.1: cubic chunks of 64** (1 MiB for float32, 0.5 MiB for uint16), default
chunk cache. Cube 32 is the equally good alternative; 64 was chosen because it has
8 times fewer chunks to index, so files open faster. The streaming writer must write
whole multiples of 64 slices at a time. Until it has
run, the file records the chunk shape it was written with (HDF5 does that
itself), so no reader depends on the choice. The expected winner is
roughly cubic chunks of about 1 MB, but that is a guess.

A downsampled copy for fast previews of huge volumes is possible later as
`/entry/volume/pyramid/level_1`, `level_2`, ... (each half the size).
Not part of 0.1.

### The reconstruction record

`/entry/reconstruction` records every parameter that shaped the volume, so
a run can be repeated or judged later. Scalars are datasets or attributes,
option dictionaries are JSON strings.

| Name | Meaning |
|---|---|
| `algorithm` | backend name, e.g. `algotom_fbp` |
| `filter` | filter name, e.g. `hann` |
| `num_iter` | iterations (iterative only) |
| `algorithm_options` | JSON of any other backend option |
| `device_requested`, `device_used` | `auto`/`cpu`/`gpu`, and what really ran |
| `center_of_rotation` | pixels; one number, or one per slice |
| `tilt_angle` | degrees |
| `rotation_offset` | degrees (our `rotate_deg`, the "offset angle") |
| `rows` | source rows reconstructed `[start, stop]` |
| `crop` | `[z0, z1, y0, y1, x0, x1]` after reconstruction |
| `dtype`, `clip_min`, `clip_max` | output type and clipping range |
| `speed_slices_per_s`, `duration_s`, `n_slices` | how fast it ran |
| `started`, `finished` | ISO 8601 timestamps |
| `software` | see "Software version" below |

`/entry/reconstruction/preprocessing` (same style): `normalization`,
`zinger_removal`, `ring_correction` (method + options JSON),
`phase_retrieval` (method, delta/beta, distance, energy, pixel size),
`beam_hardening` (coefficients), `binning`.

### The measurement it came from

The volume file contains **a copy of the projection file's metadata groups**
(`/entry/geometry`, `/entry/instrument` without the big `data`, `flat_fields`
and `dark_fields` datasets, `/entry/sample` without `rotation_angle` arrays
if too large, `/entry/provenance`) under the **same paths**, so one metadata
reader works for both file types. Additionally:

- `/entry/source/uuid`, `/entry/source/path`: which projection file this volume
  was reconstructed from (the uuid is the reliable link; the path may move).

## Software version

Two different versions are recorded, and they are not the same thing:

- `urct_version` (root attribute): the version of **this standard**, `major.minor`.
  Readers check it. It changes only when the layout changes.
- `/entry/software` (attributes, in every file the software writes: `converted_by` in
  projection files, `reconstructed_by` in volume files): the version of **the
  software** that wrote the file: `urct` (package version), `git_commit` (when
  known), `python`, and for volumes the versions of the backend libraries used
  (`algotom`, `numba`, `urct_gridrec`, `itk-rtk`, ...), so a result can be traced
  to the exact code.

### How the software is versioned

Semantic versioning (`MAJOR.MINOR.PATCH`; before 1.0 a new minor version may break
things). The number lives in one place, `core/version.py` (`__version__`); both
`pyproject.toml` files take it from there, `tomo --version` prints it, and every file
the software writes records it. Each release gets a section in `CHANGELOG.md` and a
git tag `vX.Y.Z` (tags are made by hand; the repository had none before 0.2.0).
The commit (`git describe --always --dirty`) is stored in `git_commit` when the code
runs from a git checkout. 0.1.0 was never changed, so 0.2.0 is the first number that
means something: the file standard is the first change that users' files depend on.

## Extra metadata

Anything beyond the standard goes under `/entry/extra/<anything>` (datasets
or attributes). Readers keep it and carry it along into volume files;
the software never interprets it. A vendor plugin that reads a foreign file
can put the foreign fields there.

## Reading and writing it

- **Reading projections:** a plugin `urct_hdf5` in `data_import/vendors/`
  recognized by the root attributes, filling `MetadataBundle` from the
  paths above. It replaces `generic_parallel_synchrotron_hdf5` for new files;
  that plugin stays for files already converted.
- **Writing projections:** the converter writes this layout (adding the
  new paths, keeping `/entry/instrument/sample/rotation_angle` readable as a
  legacy alias if **[D1]** decides to move it to `/entry/sample`).
- **Writing volumes:** `save_volume()` and `StreamingVolumeWriter` get the
  new layout, with chunks as above, the reconstruction record from the
  `ProjectConfig` and the run statistics, and the copied measurement metadata.
- **Reading volumes:** open a volume file in the viewer as a project; the
  reconstruction record shows in the metadata toolbox.
- **Validation:** a function `validate(path)` returns what is missing,
  tier by tier, using `KNOWN_FIELDS`. The metadata toolbox shows the same.

## Phantoms

The test phantom generators (`tests/fixtures/realistic_phantom`,
`geometric_phantom`, `generate_*.py`) currently write the old
`generic_parallel_synchrotron` layout. They are to be changed to write the new
standard, projection files first (so every test also exercises the new reader and
the `urct_complete` flag), later also the "reference volume" as a volume file. Done
after the writer exists; tracked in `doc/BACKLOG.md`.

## Decisions

| | Decision |
|---|---|
| D1 | NeXus paths for new files (angles in `/entry/sample/rotation_angle`); all older layouts still read through their plugins. |
| D2 | Both `pixel_size` (detector) and `voxel_size` (sample) exist; in parallel beam one may be omitted. |
| D3 | Projection files are not compressed. |
| D4 | Volumes are not compressed, never lossy; chunk shape decided by experiment. |
| D5 | Flats and darks are separate datasets, not one dataset with an `image_key`. |
| D6 | A file may be written with missing soft fields; `urct_complete=false` says so. |
| D7 | Version 0.1 leaves out the preview pyramid, multi-scan/dual-energy files and curved detectors. |

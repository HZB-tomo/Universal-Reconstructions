# Changelog

Versions follow semantic versioning (`MAJOR.MINOR.PATCH`); before 1.0 a new
minor version may change behavior or file layouts. The number lives in
`core/version.py`; each release is a git tag `vX.Y.Z`.

## 0.2.0

First version with a real version number: 0.1.0 was never changed or tagged,
so everything built before this point is counted as 0.1.x.

### Added
- URCT HDF5 file standard, draft 0.1 (`doc/FILE_STANDARD.md`): `urct_hdf5`
  import plugin; `convert_to_hdf5()` writes it (NeXus-style paths, geometry
  group, provenance and software records, completeness flag). Older layouts are
  still read by their own plugins.
- Volume files of the standard: `save_volume()` and `StreamingVolumeWriter` write
  `/entry/volume/data` in 64-voxel cubic chunks (uncompressed) with the reconstruction
  record (algorithm, filter, iterations, centre, tilt, offset angle, device, speed,
  preprocessing, clipping, crop), the measurement metadata, a link to the source file and
  the software versions; `reconstruction.volume_file.read_volume_info()` reads them.
- `core/version.py` (`__version__`), `tomo --version`; the software version is
  recorded in the files it writes.
- Cone-beam FDK reconstruction (`rtk_fdk`, CPU, via RTK); Paganin phase
  retrieval; beam-hardening correction; gridrec backend (`urct_gridrec` wheel,
  CPU and GPU); race-free CPU backprojection and one-slice-per-launch GPU FBP.
- Reconstruction filter choice (`filter` in the INI file, a Filter box in the
  GUI); algorithms with missing packages are listed greyed out with an install hint.
- GUI: reconstruct-all dialog with streaming to HDF5, progress and speed,
  slicing viewer (x/y/z, sinograms), COR search, metadata-aware import menu.
- Image viewer rebuilt as the reusable `imageviewer` package (doc/IMAGEVIEWER.md): a new histogram panel docked
  under the image (whole image or visible area, log counts, Auto with clip, Min-Max, Scale to view, Lock, black/white
  point boxes, colormaps); a dockable toolbar with a line tool showing length and angle; K opens a live line profile.
  The old built-in histogram on the right is gone.
- Metadata toolbox (Tools > Metadata toolbox): every metadata field of the selected project,
  grouped by tier, with its value, status (read from the file, set by you, default, derived,
  calculated, missing) and what to do about a missing one.
- `benchmarks/bench_volume_chunks.py` (chunk-shape experiment).

### Changed
- Exported HDF5 volumes moved from `/volume` to `/entry/volume/data` (`export_dataset_path`);
  older files are still opened by the viewer and `read_volume_info()`.
- `convert_to_hdf5()` no longer compresses by default (`compression="none"`)
  and writes the new layout instead of the interim one.

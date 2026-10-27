# Backlog

## Generic HDF5 standard

One HDF5 layout for projection files and for exported volumes. Draft in
[FILE_STANDARD.md](./FILE_STANDARD.md); decisions settled, not implemented.

Done in 0.2.0: the `urct_hdf5` reader plugin, the converter writing it, software
versioning (`core/version.py`, `CHANGELOG.md`, `tomo --version`; git tags are
still to be made by hand).

Also done in 0.2.0: the volume file (writer, reader, GUI viewer, pipeline record).

Work items left, in order:

- Open a volume file in the GUI as an item of its own, with its reconstruction record shown.
- Change the phantom generators in `tests/fixtures` to write the new standard.
- Make the git tag `v0.2.0`.

## TIFF import

- Projection files are ordered by file name as text, so names without
  zero padding (`proj_2`, `proj_10`) load out of order.
- A multi-page TIFF used as flat or dark field is read as a 4D array.

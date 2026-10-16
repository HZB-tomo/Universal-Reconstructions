# Backlog

## Generic HDF5 standard

One HDF5 layout for projection files and for exported volumes. Not
designed yet.

Current state:

- Converted projections (`data_import/converter.py`) use the layout read
  by the `generic_parallel_synchrotron_hdf5` plugin.
- Exported volumes are one dataset (`/volume`) with a
  `reconstruction_settings` attribute.

To define:

- Required and optional datasets and attributes.
- Units, angle convention, geometry types including cone-beam.
- Metadata fields and where they are stored.
- Format version.
- One layout covering both projection files and volume files.

## TIFF import

- Projection files are ordered by file name as text, so names without
  zero padding (`proj_2`, `proj_10`) load out of order.
- A multi-page TIFF used as flat or dark field is read as a 4D array.

# Converter

Converts a project read by the import plugins (a TIFF folder) into one
HDF5 file, optionally binning projections, flat fields and dark fields.
Available in the GUI, on the command line and in Python.

## GUI

Tools > Convert TIFF project to HDF5...

1. Add one or more project folders.
2. Choose an output folder (default: next to each source folder), a
   plugin (default: auto-detect), the bin factor, the binning method and
   the output dtype.
3. Click Convert. Stop cancels the running conversion and removes its
   partial file.

Output files are named `<folder>.h5`, or `<folder>_bin<N>.h5` when binned.
A project that fails is reported in the log and the rest continue.

## Command line

```bash
tomo convert SOURCE OUT.h5
tomo convert SOURCE OUT.h5 --bin 4
tomo convert SOURCE OUT.h5 --plugin ill_neutrons --bin-y 2 --bin-x 4 --method sum
```

| Option | Meaning |
|---|---|
| `--format tiff\|hdf5` | Source format. Default `tiff`. |
| `--plugin NAME` | Registered plugin. Default: auto-detect. |
| `--bin N` | Bin factor for both axes. Default 1. |
| `--bin-y N`, `--bin-x N` | Bin factor per axis. |
| `--method mean\|sum` | Default `mean`. |
| `--dtype` | Output dtype. Default: source dtype (`float32` for `sum` of integers). |
| `--overwrite` | Replace an existing output file. |

## Python

```python
from data_import.converter import convert_to_hdf5

convert_to_hdf5("scan_folder", "scan.h5", bin_y=4, bin_x=4)
```

Also accepts `plugin=`, `method=`, `dtype=`, `compression=`,
`compression_level=`, `batch=`, `overwrite=`, `progress=fn(done, total, label)`
and `cancel=fn() -> bool`. Raises `ConversionError` for an unsupported
source and `ConversionCancelled` when `cancel()` returns True. Any
failure removes the partial output file.

Binning functions are in `preprocessing.binning`: `bin_frames()`,
`binned_shape()`.

## What is written

```
/entry/instrument/detector/data           projections
/entry/instrument/detector/flat_fields    flat fields (if the source has them)
/entry/instrument/detector/dark_fields    dark fields (if the source has them)
/entry/instrument/sample/rotation_angle   degrees (if the source has them)
/entry/instrument/detector/pixel_size     micrometres, multiplied by the bin factor (if known)
```

Root attributes: `beamline`, `source_plugin`, `source_path`,
`source_format`, `bin_y`, `bin_x`, `bin_method`, `original_image_shape`,
`source_metadata` (JSON).

Only parallel-beam projects can be converted. Images whose size is not
divisible by the bin factor are cropped symmetrically.

The layout is the one read by the `generic_parallel_synchrotron_hdf5`
plugin. A generic HDF5 standard is on the [backlog](./BACKLOG.md).

## ILL converter script

`ill_to_hdf5.py` converts an ILL neutron-imaging project folder into one
HDF5 file that the software imports, optionally binning the data. It
also writes angles from `--angles-file` or `--angle-span`.

### Input

```
project/
  Proj/proj_*.tif          projections, one file per frame
  FlatsNeutrons.tif        flat fields (single image or multi-page)
  DarksNeutrons.tif        dark fields (single image or multi-page)
  scan_information.xml     optional
```

### Usage

```bash
python ill_to_hdf5.py PROJECT_DIR OUT.h5
python ill_to_hdf5.py PROJECT_DIR OUT.h5 --bin 8
python ill_to_hdf5.py PROJECT_DIR --info
```

| Option | Meaning |
|---|---|
| `--info` | Print what was found (frame counts, shapes, XML attributes). Writes nothing. |
| `--bin N` | Bin projections, flats and darks by N in both directions. Default 1. |
| `--bin-y N`, `--bin-x N` | Bin factor per axis. |
| `--method mean\|sum` | Binning method. Default `mean`. |
| `--dtype` | Output dtype. Default: source dtype (`float32` for `sum` of integers). |
| `--pixel-size-um` | Unbinned pixel size in micrometres. Default: from the XML. |
| `--angles-file F` | Text, CSV or `.npy` file with one angle in degrees per projection. |
| `--angle-span S` | Write a uniform angle series over S degrees. |
| `--angle-start A` | First angle for `--angle-span`. Default 0. |
| `--endpoint` | Last projection is at `start + span`. |
| `--compression gzip\|lzf\|none` | Default `gzip`, level 4 (`--compression-level`). |
| `--batch N` | Frames processed at a time. Default 16. |
| `--overwrite` | Replace an existing output file. |

Without `--angles-file` or `--angle-span` no angles are written. Set the
angle span in the GUI or in the `[completion]` section of a config
instead.

Images whose size is not divisible by the bin factor are cropped
symmetrically. The written pixel size is multiplied by the bin factor.

### Output

```
/entry/instrument/detector/data           projections
/entry/instrument/detector/flat_fields    flat fields
/entry/instrument/detector/dark_fields    dark fields
/entry/instrument/sample/rotation_angle   angles in degrees (only if given)
/entry/instrument/detector/pixel_size     micrometres
/entry/instrument/source/scan_information_xml
```

Root attributes: `beamline`, `converted_from`, `bin_y`, `bin_x`,
`bin_method`, `original_image_shape`, `original_pixel_size_um`,
`angles_source`.

The file is imported by the `generic_parallel_synchrotron_hdf5` plugin.

### Python

```python
import ill_to_hdf5

ill_to_hdf5.convert("project", "out.h5", bin_y=4, bin_x=4, angle_span=360.0)
binned = ill_to_hdf5.bin_frames(stack, bin_y=2, bin_x=2, method="mean")
```

### Reconstructing a converted project

Projections that are already flat/dark corrected need the
`projections_normalized` option, and a project without angles needs an
angle span. See [`CLI.md`](./CLI.md) and [`ORCHESTRATION.md`](./ORCHESTRATION.md).

# URCT

A modular, plugin-based tomographic reconstruction pipeline: takes raw
projection data from a scanning instrument and turns it into a
reconstructed 3D volume, then exports it to disk.

*(The underlying Python package is still named `tomo_project`, and its
CLI command is still `tomo` -- see "Install" below. URCT is this
project's name going forward; renaming the package itself is a
separate task.)*

## What it is, and why it's useful

Every scanning instrument writes its raw data in its own, incompatible
file layout, and every research group tends to end up with its own
one-off reconstruction script glued to whichever instrument it started
with. URCT separates that problem into independent stages --

```
import  ->  completion  ->  preprocessing  ->  calibration  ->  reconstruction  ->  export
```

-- so that supporting a new instrument, swapping in a different
reconstruction algorithm, or calibrating a new detector are all
localized changes, not rewrites. The result is meant to be the one
tool you reach for regardless of which instrument produced the data or
which algorithm you want to reconstruct it with.

## Unique features

- **Any instrument, via a plugin.** Reading a new vendor's file format
  is a matter of writing one plugin, not modifying the pipeline. A
  generic HDF5/TIFF fallback already covers common layouts out of the
  box.
- **Any reconstruction algorithm, via a plugin.** The same is true on
  the reconstruction side: FBP, gridrec, and ASTRA's BP/SIRT/SART/ART/CGLS
  all ship as independent, swappable backends behind one shared
  contract -- adding another one is a single new file.
- **No large-RAM workstation needed to *look at* your data.** A
  project can be opened and browsed slice by slice straight from disk;
  the raw projection stack is never loaded into memory all at once,
  regardless of how large the file is.
- **No large-RAM workstation needed to *reconstruct*, either.**
  Streaming reconstruction processes and writes one chunk of slices at
  a time, so a volume far larger than available RAM can still be
  reconstructed on ordinary hardware -- see
  [`doc/RECONSTRUCTION.md`](./doc/RECONSTRUCTION.md)'s "Streaming"
  section.

These are architectural commitments, not just today's feature list --
the aim is that support for further instruments and algorithms keeps
growing without ever needing to touch the core pipeline. Today that
core supports parallel-beam geometries; cone-beam reconstruction is
architecturally anticipated (see `Geometry.geometry_type`) but not yet
implemented.

## Three ways to use it

**Desktop GUI** -- import, preview, calibrate, and reconstruct
interactively, including batch export of several projects at once:

```bash
tomo-gui
```

See [`doc/GUI_ARCHITECTURE.md`](./doc/GUI_ARCHITECTURE.md).

**Command line** -- run one project, or a whole batch of them,
unattended, from a plain-text config file:

```bash
tomo init-config scan.ini   # a commented, fillable config template
tomo run scan.ini           # run one project
tomo batch batch.ini        # run every project in a batch config, independently
```

See [`doc/CLI.md`](./doc/CLI.md).

**Scripting / Jupyter** -- drive any stage directly from Python for
full control, or follow along in the worked notebooks in
[`doc/demo/`](./doc/demo/):

| Notebook | Covers |
|---|---|
| [`0_data_import.ipynb`](./doc/demo/0_data_import.ipynb) | Importing a raw file into a common representation |
| [`01_test_synchrotron_plugin.ipynb`](./doc/demo/01_test_synchrotron_plugin.ipynb) | Writing/testing a vendor import plugin |
| [`02_completion.ipynb`](./doc/demo/02_completion.ipynb) | Filling in metadata a vendor's file didn't provide |
| [`03_realistic_phantom.ipynb`](./doc/demo/03_realistic_phantom.ipynb) | The synthetic phantom/detector simulator used to test the pipeline |
| [`04_reconstruction.ipynb`](./doc/demo/04_reconstruction.ipynb) | Calibration and reconstruction |
| [`05_orchestration.ipynb`](./doc/demo/05_orchestration.ipynb) | Running one project end to end from a config |
| [`06_batch_orchestration.ipynb`](./doc/demo/06_batch_orchestration.ipynb) | Running many projects unattended |

## Install

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1        # Windows PowerShell; see below for macOS/Linux
pip install -e .[dev,gui]
```

`[dev]` installs `pytest`; `[gui]` installs the desktop GUI's
dependencies (PySide6, pyqtgraph, psutil). Some functionality
(zinger/ring-artifact removal, most reconstruction backends, the
`algotom_cor` calibration backend) needs the optional `algotom`
package:

```bash
pip install -e .[preprocessing]
```

ASTRA Toolbox (`pip install -e .[reconstruction_extra]`) is a further,
separate optional dependency, needed only by the five ASTRA-based
reconstruction backends.

Full setup walkthrough, including macOS/Linux activation, Jupyter
kernel setup, and Linux system packages for the GUI:
[`doc/GETTING_STARTED.md`](./doc/GETTING_STARTED.md).

## Documentation

- [`doc/GETTING_STARTED.md`](./doc/GETTING_STARTED.md) — environment setup
- [`doc/USING_THE_PIPELINE.md`](./doc/USING_THE_PIPELINE.md) — end-user walkthrough of all five stages
- [`doc/GUI_ARCHITECTURE.md`](./doc/GUI_ARCHITECTURE.md) — the desktop GUI
- [`doc/CLI.md`](./doc/CLI.md) — the `tomo` command line: every subcommand, exit codes, example output
- [`doc/ORCHESTRATION.md`](./doc/ORCHESTRATION.md) — running one or many projects unattended, from a config file
- [`doc/ARCHITECTURE.md`](./doc/ARCHITECTURE.md) — how the codebase is structured
- [`doc/WRITING_A_PLUGIN.md`](./doc/WRITING_A_PLUGIN.md) — adding support for a new instrument
- [`doc/PROJECT_METADATA.md`](./doc/PROJECT_METADATA.md) — every metadata field's tier, strategy, and unit
- [`doc/CALIBRATION.md`](./doc/CALIBRATION.md) — center-of-rotation and tilt finding
- [`doc/RECONSTRUCTION.md`](./doc/RECONSTRUCTION.md) — reconstruction backends, units, streaming, adding a backend
- [`doc/REALISTIC_PHANTOM.md`](./doc/REALISTIC_PHANTOM.md) — the phantom/detector simulator used to test the rest of the pipeline

## Running the tests

```bash
pytest tests/
```

Uses small synthetic datasets; runs in under two minutes.

## License

MIT — see [`LICENSE`](./LICENSE). You're free to use, modify, and
redistribute this software, including commercially, with no royalty
and no need to ask; just keep the copyright notice.

## Author

Shahabeddin Dayani, [Helmholtz-Zentrum Berlin für Materialien und Energie (HZB)](https://www.helmholtz-berlin.de/).

## Acknowledgments

Built on top of, and grateful to, the open-source projects this
pipeline's optional reconstruction/preprocessing backends wrap:
[algotom](https://github.com/algotom/algotom) (Diamond Light Source),
the [ASTRA Toolbox](https://www.astra-toolbox.com/), and
[TomoPy](https://tomopy.readthedocs.io/); and to
[PySide6](https://doc.qt.io/qtforpython/)/[pyqtgraph](https://www.pyqtgraph.org/)
for the desktop GUI.

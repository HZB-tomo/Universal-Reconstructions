# tomo_project

A tomographic reconstruction pipeline: takes raw projection data from a
scanning instrument and turns it into a reconstructed 3D volume.

## Aim

Instruments write their raw data in different, incompatible file
layouts. This project separates the problem into independent stages so
that supporting a new instrument, swapping in a different
reconstruction algorithm, or calibrating a new detector are all
localized changes rather than rewrites:

```
import  ->  completion  ->  preprocessing  ->  calibration  ->  reconstruction  ->  export
```

- **import** reads a vendor's file (HDF5 or TIFF) into a common
  in-memory representation, via a plugin per vendor.
- **completion** fills in metadata a vendor's file didn't provide
  (deriving it from other fields, or a documented default).
- **preprocessing** turns raw projections into reconstruction-ready
  sinograms (flat/dark correction, zinger and ring-artifact removal).
- **calibration** finds the center of rotation and any rotation-axis
  tilt.
- **reconstruction** turns calibrated sinograms into a reconstructed
  image or volume, and writes it to disk.

Calibration and reconstruction are built as swappable backends (a
registry, like the vendor plugins), so adding support for a different
reconstruction package is a matter of writing one new backend file,
not restructuring the pipeline.

`orchestration/` drives one project, or many independently, through
every stage above from a single config -- either built directly in
code, or a hand-editable INI file, so a batch of projects can be set
up and run unattended. `pip install -e .` also registers a `tomo`
command-line tool over the same thing:

```bash
tomo init-config scan.ini   # a commented, fillable config template
tomo run scan.ini           # run one project
tomo batch batch.ini        # run every project in a batch config, independently
```

See [`doc/ORCHESTRATION.md`](./doc/ORCHESTRATION.md) and
[`doc/CLI.md`](./doc/CLI.md).

## Install

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1        # Windows PowerShell; see below for macOS/Linux
pip install -e .[dev]
```

`[dev]` installs `pytest`. Some functionality (zinger/ring-artifact
removal, the `algotom_fbp` reconstruction backend, the `algotom_cor`
calibration backend) needs the optional `algotom` package:

```bash
pip install -e .[preprocessing]
```

Full setup walkthrough, including macOS/Linux activation and Jupyter
kernel setup: [`doc/GETTING_STARTED.md`](./doc/GETTING_STARTED.md).

## Usage

```python
from data_import.importer import DataImporter
from core.data_interface import DataInterface
import completion

importer = DataImporter()
project = importer.import_project("my_scan.h5", format="hdf5", storage_mode="ram")
completion.run(project)

print(project)
print(DataInterface(project).get_projections().shape)
```

That's the first stage of the pipeline; see
[`doc/USING_THE_PIPELINE.md`](./doc/USING_THE_PIPELINE.md) for all five
stages end to end, or the worked notebooks in
[`doc/demo/`](./doc/demo/).

## Documentation

- [`doc/USING_THE_PIPELINE.md`](./doc/USING_THE_PIPELINE.md) — end-user walkthrough of all five stages
- [`doc/ORCHESTRATION.md`](./doc/ORCHESTRATION.md) — running one or many projects unattended, from a config file
- [`doc/CLI.md`](./doc/CLI.md) — the `tomo` command line: every subcommand, exit codes, example output
- [`doc/ARCHITECTURE.md`](./doc/ARCHITECTURE.md) — how the codebase is structured
- [`doc/WRITING_A_PLUGIN.md`](./doc/WRITING_A_PLUGIN.md) — adding support for a new instrument
- [`doc/PROJECT_METADATA.md`](./doc/PROJECT_METADATA.md) — every metadata field's tier, strategy, and unit
- [`doc/CALIBRATION.md`](./doc/CALIBRATION.md) — center-of-rotation and tilt finding
- [`doc/RECONSTRUCTION.md`](./doc/RECONSTRUCTION.md) — the reconstruction backend contract, axis conventions, adding a backend
- [`doc/REALISTIC_PHANTOM.md`](./doc/REALISTIC_PHANTOM.md) — the phantom/detector simulator used to test the rest of the pipeline
- [`doc/GETTING_STARTED.md`](./doc/GETTING_STARTED.md) — environment setup

## Running the tests

```bash
pytest tests/
```

Uses small synthetic datasets; runs in a few seconds.

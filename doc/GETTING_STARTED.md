# Getting started

## 1. Set up a virtual environment

From the project root (the folder containing `pyproject.toml`):

```bash
python3 -m venv .venv          # Windows: python -m venv .venv
```

Activate it (repeat this every time you open a new terminal):

```bash
source .venv/bin/activate                # macOS/Linux
.venv\Scripts\Activate.ps1               # Windows PowerShell
.venv\Scripts\activate.bat               # Windows cmd.exe
```

## 2. Install the project

```bash
pip install -e .[dev]
```

The `-e` (editable) install means `import core` and `import data_import`
work from any notebook or script running in this virtual environment.
`[dev]` also installs `pytest`.

If you're working from a Jupyter notebook, make sure its kernel is
actually using this virtual environment (check the kernel name in the
top right of the notebook). If it isn't listed yet:

```bash
pip install ipykernel
python -m ipykernel install --user --name=tomo_project --display-name="Python (tomo_project)"
```

## 3. Run the test suite

```bash
pytest tests/ -v
```

The tests use small synthetic datasets, so this runs in a few seconds.

## 4. Generate the full-size synthetic dataset (optional)

To generate the full-spec dataset (256x256x512 volume, 800 angles;
takes a few minutes):

```bash
python -m tests.fixtures.generate_synthetic
```

This writes both formats to `tests/fixtures/generated/full/`:
`synthetic.h5` and a `synthetic_data/` folder.

## 5. Load a project and look at the data

```python
from data_import.importer import DataImporter
from core.data_interface import DataInterface

importer = DataImporter()
project = importer.import_project(
    "tests/fixtures/generated/full/synthetic.h5",
    format="hdf5",
    storage_mode="ram",   # or "disk" for hdf5; see ARCHITECTURE.md
)

print(project)
print("needs user input:", project.needs_user_input(), project.missing_soft_fields())

di = DataInterface(project)
print(di.get_projections().shape)   # (800, 256, 512)
print(di.get_metadata("angles"))
```

If you're calling this from a notebook that doesn't live at the project
root, use an absolute path (see the note on relative paths below).

### Viewing a projection with matplotlib

```python
import matplotlib.pyplot as plt

proj_0 = di.get_projections(index=0)   # just one angle, not the whole stack
plt.imshow(proj_0, cmap="gray")
plt.colorbar()
plt.show()
```

## A note on paths from notebooks

Relative paths (like `"tests/fixtures/generated/full/synthetic.h5"`
above) are resolved relative to the notebook kernel's current working
directory, which may not be the project root. If you hit
`FileNotFoundError`, either use an absolute path, or add this to your
first cell to locate the project root:

```python
import sys
from pathlib import Path

here = Path.cwd()
project_root = next(p for p in [here, *here.parents] if (p / "data_import").is_dir())
sys.path.insert(0, str(project_root))
```

## See also

- [`ARCHITECTURE.md`](./ARCHITECTURE.md)
- [`USING_THE_PIPELINE.md`](./USING_THE_PIPELINE.md)
- [`WRITING_A_PLUGIN.md`](./WRITING_A_PLUGIN.md)

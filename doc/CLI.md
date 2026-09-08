# Command line (`tomo`)

`tomo` runs the pipeline from a terminal, without writing any Python.
It is a thin wrapper over `orchestration/`'s `PipelineRun`/`run_batch`
(`orchestration/cli.py`) -- every subcommand just calls one function
from there and prints the result. See
[`ORCHESTRATION.md`](./ORCHESTRATION.md) for what those functions
actually do; this page is about using `tomo` itself.

## Install

```bash
pip install -e .
```

registers the `tomo` command (via `pyproject.toml`'s
`[project.scripts]`). Check it worked:

```bash
tomo --help
```

## Typical workflow

```bash
tomo init-config scan.ini      # write a template
# ... edit scan.ini: fill in path, and anything else you want to change ...
tomo validate scan.ini         # check it before committing to a real run
tomo run scan.ini              # run it
```

For several projects at once:

```bash
tomo init-config batch.ini --batch   # write a batch-shaped template
# ... edit batch.ini: one [project_id:stage] block per project ...
tomo validate batch.ini              # check every project first
tomo batch batch.ini                 # run all of them, independently
```

## `tomo init-config`

```bash
tomo init-config <path> [--batch]
```

Writes a commented, fillable example config to `<path>`. Without
`--batch`, a single-project template (bare `[stage]` sections). With
`--batch`, a batch-shaped template (`[DEFAULT:stage]` plus one example
`[project_a:stage]` block to copy for each real project). Never reads
or runs anything; always exits `0`.

## `tomo validate`

```bash
tomo validate <config.ini>
```

Dry-runs the config through import, completion, preprocessing, and
calibration -- not reconstruction or export, the expensive stages --
and reports what it found. Works on either a single-project or a batch
file; a batch file prints one line per project.

```
$ tomo validate scan.ini
scan: OK

$ tomo validate broken.ini
does_not_exist: ["unexpected error: [Errno 2] ... No such file or directory ..."]
```

Exits `0` if nothing was found, `1` if any project had a problem.
Running this before `tomo run`/`tomo batch` is how you find out about a
missing field or a bad path before committing to the real (slower) run
-- see `ORCHESTRATION.md`'s "Batch runs" section for why this check
exists and what it does and doesn't guarantee.

## `tomo run`

```bash
tomo run <config.ini> [--interactive]
```

Runs one project through the full pipeline (`PipelineRun(config).run_all()`)
and prints its log:

```
$ tomo run scan.ini
imported via generic_parallel_synchrotron_hdf5
completion resolved
preprocessed 16 rows
calibrated: center=32.5, tilt_deg=0.0
reconstructed volume (16, 64, 64)
postprocessed to float32, shape (16, 64, 64)
exported to scan_reconstructed.h5
done: scan_reconstructed.h5
```

`--interactive` prompts on stdin for any field completion can't resolve
automatically, instead of failing immediately. Without it (the
default), a missing field fails right away -- the same behavior
`tomo batch` always uses, since there's nobody to answer a prompt in an
unattended run.

Refuses a batch-shaped config, naming `tomo batch` as the right command
instead of trying to run it as one project.

Exits `0` on success, `1` on failure (a missing field, a bad path, a
reconstruction error, ...), printing `FAILED: <reason>` to stderr
either way.

## `tomo batch`

```bash
tomo batch <batch_config.ini>
```

Runs every project in the file (`run_batch()`), independently -- one
project failing never stops the others:

```
$ tomo batch projects.ini
scan_200: OK
scan_300: OK
scan_400: FAILED: [complete] missing required fields: ['geometry_type']

validation problems (advisory -- every project above was still attempted):
  scan_400: ["[complete] missing required fields: ['geometry_type']"]
```

Refuses a single-project config, naming `tomo run` as the right command
instead.

Exits `0` only if every project succeeded, `1` if any failed. To retry
just what failed after fixing the config, see `ORCHESTRATION.md`'s
`PipelineResult.config` pattern (that part needs Python, not `tomo`
directly, since it involves editing a config in memory before
re-running).

## Exit codes

| Command | `0` | `1` |
|---|---|---|
| `init-config` | always | (never fails) |
| `validate` | nothing found | any project had a problem |
| `run` | the project succeeded | it failed, or the config was batch-shaped |
| `batch` | every project succeeded | any project failed, or the config was single-project |

Meant for scripting -- checkable with `$?` (or `%errorlevel%` on
Windows) without parsing `tomo`'s printed output.

## Why `run`/`batch` don't crash with a traceback

`PipelineRun.run_all()` and `run_batch()` are deliberately narrow about
what they catch: `run_all()` only catches its own `PipelineStageError`,
and `run_batch()`'s broader catch exists specifically so one project's
unanticipated failure can't take down the rest of a batch. Used as a
library (a notebook, a future GUI), a genuine bug is meant to surface
normally, not be silently swallowed.

A CLI user doesn't want a raw Python traceback for something as
mundane as a typo'd path, though -- so `tomo run` and `tomo batch` each
add their own broad exception catch around the call, purely as a
presentation choice at the CLI layer. `PipelineRun`/`run_batch`
themselves are unchanged; see `orchestration/cli.py`'s `_cmd_run()`.

## See also

- [`ORCHESTRATION.md`](./ORCHESTRATION.md): what each subcommand
  actually calls, the config file format, `ProjectConfig`'s full field
  list.
- [`demo/example_single_project.ini`](./demo/example_single_project.ini),
  [`demo/example_batch_projects.ini`](./demo/example_batch_projects.ini):
  real, working example configs.
- [`demo/05_orchestration.ipynb`](./demo/05_orchestration.ipynb),
  [`demo/06_batch_orchestration.ipynb`](./demo/06_batch_orchestration.ipynb):
  the same functionality driven from Python instead of the command line.

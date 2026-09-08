"""
The `tomo` command line: a thin wrapper over PipelineRun/run_batch. All
the real logic lives in pipeline.py/batch.py; this module is argument
parsing, calling one function, and printing the result.

    tomo run <config.ini>          one project, run_all()
    tomo batch <config.ini>        every project in a batch config, independently
    tomo validate <config.ini>     dry-run (single or batch), no reconstruction/export
    tomo init-config <path>        write a commented example config
"""
from __future__ import annotations

import argparse
import sys

from orchestration.batch import run_batch, validate_config
from orchestration.config_ini import load_config, write_config_template
from orchestration.pipeline import PipelineRun


def _cmd_run(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    if isinstance(config, dict):
        print(f"error: {args.config} is a batch config; use 'tomo batch' instead", file=sys.stderr)
        return 1

    run = PipelineRun(config, interactive=args.interactive)
    try:
        result = run.run_all()
    except Exception as e:
        # run_all() itself only catches PipelineStageError -- deliberately,
        # so a real bug still surfaces when PipelineRun is used as a
        # library (a notebook, a GUI). A CLI user just wants a clean
        # one-line error instead of a traceback for something as
        # mundane as a bad path, so this catch is the CLI's own choice,
        # the same way run_batch() makes it for the same reason.
        print(f"FAILED: {e}", file=sys.stderr)
        return 1

    for line in result.log:
        print(line)

    if not result.success:
        print(f"FAILED: {result.error}", file=sys.stderr)
        return 1
    print(f"done: {result.volume_path}")
    return 0


def _cmd_batch(args: argparse.Namespace) -> int:
    configs = load_config(args.config)
    if not isinstance(configs, dict):
        print(f"error: {args.config} is a single-project config; use 'tomo run' instead", file=sys.stderr)
        return 1

    report = run_batch(list(configs.values()))
    for result in report.results:
        status = "OK" if result.success else f"FAILED: {result.error}"
        print(f"{result.project_id}: {status}")

    if report.validation_problems:
        print("\nvalidation problems (advisory -- every project above was still attempted):")
        for project_id, problems in report.validation_problems.items():
            print(f"  {project_id}: {problems}")

    return 1 if report.failed else 0


def _cmd_validate(args: argparse.Namespace) -> int:
    loaded = load_config(args.config)
    configs = loaded if isinstance(loaded, dict) else {loaded.resolved_project_id(): loaded}

    any_problems = False
    for project_id, config in configs.items():
        problems = validate_config(config)
        if problems:
            any_problems = True
            print(f"{project_id}: {problems}")
        else:
            print(f"{project_id}: OK")

    return 1 if any_problems else 0


def _cmd_init_config(args: argparse.Namespace) -> int:
    write_config_template(args.path, batch=args.batch)
    print(f"wrote {args.path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tomo", description="Run the tomo_project reconstruction pipeline.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_run = subparsers.add_parser("run", help="Run one project through the full pipeline.")
    p_run.add_argument("config", help="Path to a single-project INI config.")
    p_run.add_argument(
        "--interactive", action="store_true",
        help="Prompt on stdin for any field completion can't resolve automatically.",
    )
    p_run.set_defaults(func=_cmd_run)

    p_batch = subparsers.add_parser("batch", help="Run every project in a batch config, independently.")
    p_batch.add_argument("config", help="Path to a batch INI config.")
    p_batch.set_defaults(func=_cmd_batch)

    p_validate = subparsers.add_parser(
        "validate", help="Dry-run a config (single or batch) without reconstructing or exporting.",
    )
    p_validate.add_argument("config", help="Path to an INI config.")
    p_validate.set_defaults(func=_cmd_validate)

    p_init = subparsers.add_parser("init-config", help="Write a commented example config.")
    p_init.add_argument("path", help="Where to write the template.")
    p_init.add_argument(
        "--batch", action="store_true", help="Write a batch-shaped template instead of a single-project one.",
    )
    p_init.set_defaults(func=_cmd_init_config)

    return parser


def main(argv: list = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

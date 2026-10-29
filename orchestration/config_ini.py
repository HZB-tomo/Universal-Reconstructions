"""
Reads/writes ProjectConfig as INI text -- the format a person without
a programming background can hand-edit: plain key = value lines, "#"
comments, no brackets/quotes/commas to get wrong.

Two grammars, auto-detected by whether any section name contains ":":

Single-project: sections are bare stage names.

    [import]
    path = data/scanA.h5

    [calibration]
    center = 56.0

Batch: sections are "<project_id>:<stage>", plus an optional
"DEFAULT:<stage>" per stage supplying values every project in the file
inherits unless its own section overrides that key. (This is NOT
configparser's built-in [DEFAULT] -- that merges by key name into
every section regardless of stage, and "path" means something
different in [import] than in [export]; scoping defaults per-stage
avoids that collision.)

    [DEFAULT:reconstruction]
    backend = algotom_fbp

    [scanA:import]
    path = data/scanA.h5

    [scanB:import]
    path = data/scanB.h5

The [completion] section is not field-mapped like the others: every
key in it becomes one entry of ProjectConfig.completion_answers
directly, since it corresponds 1:1 to core/known_fields.py's "ask"
fields (geometry_type, sample_name, ...) rather than a fixed set of
pipeline settings.
"""
from __future__ import annotations

import configparser
from pathlib import Path
from typing import Callable, Optional, Union

from orchestration.config import ProjectConfig


def _parse_bool(v: str) -> bool:
    lowered = v.strip().lower()
    if lowered in ("true", "yes", "1", "on"):
        return True
    if lowered in ("false", "no", "0", "off"):
        return False
    raise ValueError(f"expected true/false, got {v!r}")


def _literal(*choices: str) -> Callable[[str], str]:
    def parser(v: str) -> str:
        if v not in choices:
            raise ValueError(f"expected one of {choices}, got {v!r}")
        return v

    return parser


def _parse_crop(v: str) -> tuple:
    parts = [p.strip() for p in v.split(",")]
    if len(parts) != 6:
        raise ValueError(f"expected 6 comma-separated integers (z0,z1,y0,y1,x0,x1), got {v!r}")
    z0, z1, y0, y1, x0, x1 = (int(p) for p in parts)
    return ((z0, z1), (y0, y1), (x0, x1))


def _parse_floats(v: str) -> tuple:
    try:
        values = tuple(float(p) for p in v.split(",") if p.strip())
    except ValueError:
        raise ValueError(f"expected comma-separated numbers, got {v!r}") from None
    if not values:
        raise ValueError(f"expected at least one number, got {v!r}")
    return values


def _parse_rows(v: str) -> slice:
    parts = [p.strip() for p in v.split(",")]
    if len(parts) != 2:
        raise ValueError(f"expected 'start,stop', got {v!r}")
    start, stop = (int(p) for p in parts)
    return slice(start, stop)


# section name -> {ini key: (ProjectConfig field name, parser)}
FIELD_SPECS: dict[str, dict[str, tuple]] = {
    "import": {
        "project_id": ("project_id", str),
        "path": ("path", str),
        "format": ("format", _literal("hdf5", "tiff")),
        "storage_mode": ("storage_mode", _literal("ram", "disk")),
        "plugin_name": ("plugin_name", str),
    },
    "preprocessing": {
        "rows": ("rows", _parse_rows),
        "reference_reducer": ("reference_reducer", _literal("mean", "median")),
        "dark_offset": ("dark_offset", float),
        "projections_normalized": ("projections_normalized", _parse_bool),
        "zinger_removal": ("zinger_removal", _parse_bool),
        "zinger_threshold": ("zinger_threshold", float),
        "zinger_size": ("zinger_size", int),
        "ring_correction": ("ring_correction", _parse_bool),
        "ring_method": ("ring_method", str),
        "phase_retrieval": ("phase_retrieval", _parse_bool),
        "phase_method": ("phase_method", str),
        "phase_delta_beta": ("phase_delta_beta", float),
        "phase_distance_mm": ("phase_distance_mm", float),
        "phase_energy_kev": ("phase_energy_kev", float),
        "phase_pixel_um": ("phase_pixel_um", float),
        "beam_hardening": ("beam_hardening", _parse_bool),
        "bh_coefficients": ("bh_coefficients", _parse_floats),
    },
    "calibration": {
        "backend": ("calibration_backend", str),
        "center": ("center", float),
        "tilt_deg": ("tilt_deg", float),
        "tilt_num_rows": ("tilt_num_rows", int),
        "tilt_margin": ("tilt_margin", int),
        "tilt_outlier_tolerance": ("tilt_outlier_tolerance", float),
    },
    "reconstruction": {
        "backend": ("reconstruction_backend", str),
        "device": ("device", str),
        "filter": ("reconstruction_filter", str),
        "rotate_deg": ("rotate_deg", float),
    },
    "postprocess": {
        "crop": ("crop", _parse_crop),
        "dtype": ("dtype", _literal("float32", "uint16")),
        "dtype_clip_min": ("dtype_clip_min", float),
        "dtype_clip_max": ("dtype_clip_max", float),
    },
    "export": {
        "format": ("export_format", _literal("tiff", "hdf5")),
        "path": ("export_path", str),
        "dataset_path": ("export_dataset_path", str),
        "stream": ("stream", _parse_bool),
        "stream_chunk_rows": ("stream_chunk_rows", int),
    },
}


def _format_value(value, parse) -> Optional[str]:
    """The inverse of `parse`, for save_config() below -- only the three
    non-passthrough parsers need special handling; str/float/int/
    _literal(...) round-trip through plain str(value)."""
    if parse is _parse_bool:
        return "true" if value else "false"
    if parse is _parse_crop:
        (z0, z1), (y0, y1), (x0, x1) = value
        return f"{z0},{z1},{y0},{y1},{x0},{x1}"
    if parse is _parse_floats:
        return ", ".join(repr(float(v)) for v in value)
    if parse is _parse_rows:
        if not isinstance(value, slice):
            return None  # can't represent a non-slice rows value in this grammar
        return f"{value.start},{value.stop}"
    return str(value)


def save_config(config: ProjectConfig, path: Union[str, Path]) -> None:
    """
    Writes `config` to `path` as single-project INI text -- the inverse
    of load_config() for a single-project file. Round-trips every field
    FIELD_SPECS covers, plus completion_answers (its own [completion]
    section); a field left at `None` is omitted, same as "blank" means
    to load_config(). reconstruction_options/ring_options are never
    written -- they're not in the INI grammar at all (see this module's
    docstring), so a config using them loses those specific values on
    export.
    """
    lines = []
    for section_name, field_map in FIELD_SPECS.items():
        section_lines = []
        for ini_key, (field_name, parse) in field_map.items():
            value = getattr(config, field_name)
            if value is None:
                continue
            formatted = _format_value(value, parse)
            if formatted is None:
                continue
            section_lines.append(f"{ini_key} = {formatted}")
        if section_lines:
            lines.append(f"[{section_name}]")
            lines.extend(section_lines)
            lines.append("")

    if config.completion_answers:
        lines.append("[completion]")
        for key, value in config.completion_answers.items():
            lines.append(f"{key} = {value}")
        lines.append("")

    Path(path).write_text("\n".join(lines).rstrip() + "\n")


def _build_config(sections: dict[str, dict[str, str]]) -> tuple[Optional[ProjectConfig], list[str]]:
    """sections: {stage_name: {ini_key: raw_string_value}}, already
    merged (batch mode's DEFAULT:stage + project:stage) or straight
    from a single-project file's bare-named sections."""
    kwargs: dict = {}
    errors: list[str] = []

    for section_name, field_map in FIELD_SPECS.items():
        raw = sections.get(section_name, {})
        for ini_key, (field_name, parse) in field_map.items():
            value = raw.get(ini_key, "").strip()
            if not value:
                continue  # blank means "use ProjectConfig's own default"
            try:
                kwargs[field_name] = parse(value)
            except ValueError as e:
                errors.append(f"[{section_name}] '{ini_key}': {e}")

    completion_answers = {k: v for k, v in sections.get("completion", {}).items() if v.strip()}
    if completion_answers:
        kwargs["completion_answers"] = completion_answers

    if errors:
        return None, errors
    return ProjectConfig(**kwargs), []


def load_config(path: Union[str, Path]) -> Union[ProjectConfig, dict[str, ProjectConfig]]:
    """
    Parses `path` as either a single-project or a batch INI config
    (auto-detected). Returns one ProjectConfig for a single-project
    file, or {project_id: ProjectConfig} for a batch file.

    Raises ValueError, naming every problem found, if any section's
    values don't parse or validate.
    """
    # inline_comment_prefixes lets "key = value   # note" work, not just
    # a "#" comment on its own line -- the natural way someone hand-
    # editing this file would annotate a value.
    parser = configparser.ConfigParser(inline_comment_prefixes=("#",))
    parser.read(path)

    is_batch = any(":" in section for section in parser.sections())

    if not is_batch:
        sections = {s: dict(parser.items(s)) for s in parser.sections()}
        config, errors = _build_config(sections)
        if errors:
            raise ValueError(f"Problems in {path}: {errors}")
        return config

    defaults: dict[str, dict[str, str]] = {}
    projects: dict[str, dict[str, dict[str, str]]] = {}
    for section in parser.sections():
        scope, sep, stage = section.partition(":")
        if not sep:
            raise ValueError(
                f"Batch config {path}: section '[{section}]' must be '[project_id:stage]' or '[DEFAULT:stage]'"
            )
        kv = dict(parser.items(section))
        if scope == "DEFAULT":
            defaults.setdefault(stage, {}).update(kv)
        else:
            projects.setdefault(scope, {}).setdefault(stage, {}).update(kv)

    configs: dict[str, ProjectConfig] = {}
    all_errors: list[str] = []
    for project_id, stage_sections in projects.items():
        merged = {
            stage: {**defaults.get(stage, {}), **stage_sections.get(stage, {})}
            for stage in set(defaults) | set(stage_sections)
        }
        config, errors = _build_config(merged)
        if errors:
            all_errors.append(f"project '{project_id}': {errors}")
            continue
        if not config.project_id:
            config.project_id = project_id
        configs[project_id] = config

    if all_errors:
        raise ValueError(f"Problems in {path}: {all_errors}")
    return configs


def resolve_config(data_path: Union[str, Path], explicit_path: Optional[Union[str, Path]] = None) -> Optional[ProjectConfig]:
    """
    explicit_path, if given, is loaded and returned (must be a
    single-project config). Otherwise, looks for a sidecar
    "<data_path>.tomo_config.ini" next to the data and loads that if
    present. Returns None if neither is found -- the caller decides
    whether to prompt interactively or treat that as a validation
    failure.
    """
    if explicit_path is not None:
        result = load_config(explicit_path)
        if isinstance(result, dict):
            raise ValueError(f"{explicit_path} is a batch config; resolve_config() expects a single-project config")
        return result

    sidecar = Path(str(data_path) + ".tomo_config.ini")
    if sidecar.exists():
        return resolve_config(data_path, explicit_path=sidecar)
    return None


_TEMPLATE = """\
# tomo_project config template. Blank values fall back to the automatic
# behavior (auto-detect, auto-calibrate, library defaults) -- fill in
# only what you need to change. See doc/ORCHESTRATION.md.

[import]
path =
format = hdf5
storage_mode = ram
# plugin_name =        # only needed if more than one plugin could match this file

[completion]
# one line per field the file doesn't already provide, e.g.:
# geometry_type = parallel
# sample_name = my_sample

[preprocessing]
# rows = 0,100         # start,stop; blank = every row
# projections_normalized = true   # projections already flat/dark corrected: skip that step
# zinger_removal = true
# ring_correction = true
# phase_retrieval = true  # propagation-based phase contrast (Paganin); needs delta/beta, and energy,
# phase_delta_beta =      #   distance and pixel size from the file or from the lines below
# phase_distance_mm =     # sample-to-detector distance
# phase_energy_kev =
# phase_pixel_um =        # effective pixel size at the sample
# beam_hardening = true   # polynomial linearization of the attenuation (removes cupping)
# bh_coefficients = 1, 0.3, -0.04   # c1, c2, c3, ... of  p_corrected = c1 p + c2 p^2 + c3 p^3

[calibration]
# center =             # blank = auto find_center()
# tilt_deg =           # blank = auto find_tilt()

[reconstruction]
# backend =           # blank = gridrec (parallel beam) or astra_fdk on the GPU (cone beam); see doc/RECONSTRUCTION.md
# filter = hann        # ramp (sharpest, noisiest), hann, hamming, ...; blank = the algorithm's default
# rotate_deg =         # blank = no rotation; degrees about the scan axis

[postprocess]
dtype = float32
# crop = 4,8,0,96,0,96  # z0,z1,y0,y1,x0,x1

[export]
format = hdf5
path = {project_id}_volume.h5
# stream = true          # chunk by chunk instead of the whole volume at once;
#                           needs center/tilt (and, for uint16, dtype_clip_min/max) already resolved, not blank
# stream_chunk_rows =     # blank = size automatically from available RAM/cores
"""

_BATCH_TEMPLATE = """\
# tomo_project batch config template: several projects in one file. See
# doc/ORCHESTRATION.md's "Batch grammar" section.
#
# Sections are [project_id:stage]. [DEFAULT:stage] supplies a value
# every project inherits unless its own section overrides that key --
# add more [DEFAULT:stage] entries for anything all your projects share
# (reconstruction backend, output dtype, ...), and keep each project's
# own sections to just what's different about it.
#
# Copy the [project_a:...] block below (renaming "project_a") for each
# additional project.

[DEFAULT:reconstruction]
# backend =           # blank = gridrec (parallel beam) or astra_fdk on the GPU (cone beam)

[DEFAULT:postprocess]
dtype = float32

[project_a:import]
path =
format = hdf5

[project_a:calibration]
# center =             # blank = auto find_center()
# tilt_deg =           # blank = auto find_tilt()

[project_a:export]
format = hdf5
path = {project_id}_volume.h5
"""


def write_config_template(path: Union[str, Path], batch: bool = False) -> None:
    """
    Writes a commented, fully-populated example config to `path`, ready
    to copy and fill in. `batch=True` writes a batch-shaped skeleton
    (one example project plus [DEFAULT:stage] sections) instead of a
    single-project one.
    """
    Path(path).write_text(_BATCH_TEMPLATE if batch else _TEMPLATE)

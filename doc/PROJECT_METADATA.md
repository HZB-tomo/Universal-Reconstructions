# Project metadata fields

This is the human-readable reference for every metadata field a
`ProjectObject` carries from import through to reconstruction handoff.

The source of truth is [core/known_fields.py](../core/known_fields.py),
consulted at runtime by `data_import`'s importer and `completion/`. If
this document and the code disagree, the code is correct.

This documents the completion-stage fields, plus `center_of_rotation`
and `tilt_angle`, which are filled in by `calibration/`. Preprocessing-
and reconstruction-stage fields (flat/dark correction status, ring
removal, log transform, etc.) are not listed here.

## Vocabulary

Two independent axes describe every field:

- **Tier**: the consequence of the field staying missing. Defined in
  `core/metadata.py`:
  - `mandatory_hard`: missing it fails import outright.
  - `mandatory_soft`: import still succeeds, but the field shows up in
    `needs_user_input()` until it's resolved.
  - `optional`: no consequence if missing.
  - `computed`: never blocks anything. Always carries a best-available
    value that a later stage can refine and overwrite.
- **Resolution strategy**: how a missing value gets filled in. Defined
  in `core/known_fields.py` (`Strategy`), consumed by
  `completion/engine.py`'s automatic pass:
  - `Derive`: computed automatically from other fields already present
    (e.g. angles from angle span and projection count). Built-in
    derivers live in [`completion/derivers.py`](../completion/derivers.py),
    registered via `completion/engine.py`'s `register_deriver()`.
  - `Default`: a fixed fallback value, applied automatically, always
    overridable by the user.
  - `Ask`: no safe default; something outside `completion/` supplies it
    via `completion.provide()`.
  - `Calculate`: an algorithm proposes a value from the image data, and
    the user confirms or adjusts it. `completion/` never attempts these.

A field can list more than one strategy, tried in order; `angles` is
derive-then-ask.

## Fields

| Field | Category | Tier | Strategy | Default | Unit | Notes |
|---|---|---|---|---|---|---|
| `angles` | Geometry / scan | `mandatory_soft` | Derive (from `angle_span` + projection count + `rotation_direction`) → else Ask | — | degrees | Projection count is read off the array shape, never stored separately. |
| `geometry_type` | Geometry / scan | `mandatory_soft` | Ask | — | categorical: `parallel` / `cone` | No safe default — a wrong guess silently corrupts every downstream calculation. Gates whether the cone-beam-only rows below apply. |
| `angle_span` | Geometry / scan | `mandatory_soft` | Derive (max − min of `angles`, if present) → else Ask | — | degrees | |
| `rotation_direction` | Geometry / scan | `optional` | Default | `ccw` | categorical: `cw` / `ccw` | Wrong default only flips orientation, never invalidates the reconstruction. |
| `voxel_size` | Pixel / detector | `mandatory_soft` | Default | `1` | µm | |
| `detector_pixel_size` | Pixel / detector | `mandatory_soft` | Derive (from `voxel_size`) → else Default | `1` | µm | Distinct from `voxel_size` only in cone-beam (magnification). `voxel_size` always resolves first, so this derives from it in practice. |
| `source_to_detector_distance` | Cone-beam geometry | `mandatory_soft`, conditional on `geometry_type == "cone"` | Ask | — | mm | Unused for parallel-beam projects. |
| `source_to_object_distance` | Cone-beam geometry | `mandatory_soft`, conditional on `geometry_type == "cone"` | Ask | — | mm | Unused for parallel-beam projects. |
| `center_of_rotation` | Alignment | `computed` | Calculate (tool proposes) + user confirms | `0` | pixels (deviation from image center, sub-pixel float) | Never blocks import or reconstruction; reconstruction proceeds on the current value whether or not it's been confirmed. |
| `tilt_angle` | Alignment | `computed` | Calculate (tool proposes) + user confirms | `0` | degrees | Same non-blocking behavior as `center_of_rotation`; an uncorrected tilt degrades quality but doesn't invalidate the run. |
| `operator` | Provenance | `optional` | Ask | — | text | |
| `instrument_name` | Provenance | `optional` | — | — | text | |
| `acquisition_datetime` | Provenance | `optional` | — | — | timestamp | |
| `sample_name` | Provenance | `optional` | Ask | — | text | Useful for naming output volumes later. |
| `energy` | Provenance | `optional` | — (no prompt) | — | keV | Purely a pass-through if the vendor happens to provide it — never asked for, never blocks or affects calculations. |

# Project metadata fields

This is the human-readable reference for every metadata field a
`ProjectObject` carries from import through to reconstruction handoff.

**The actual source of truth is code, not this file**: [core/known_fields.py](../core/known_fields.py)
is what `data_import`'s importer and `completion/` both consult at
runtime. This document exists so "why is this one required" has a
readable place to look instead of reading the registry's dataclass
fields directly — but if the two ever disagree, the code is correct and
this file is stale and needs updating, not the other way around.

This documents the **completion stage** fields (this is the stage
currently implemented — see `completion/`). Calibration adds
`center_of_rotation` and `tilt_angle`, both already listed below since
their tier/strategy is settled even though the calibration tools
themselves aren't built yet. Preprocessing- and reconstruction-stage
fields (flat/dark correction status, ring removal, log transform, etc.)
are intentionally not listed here yet — they'll be added when that
stage is designed.

## Vocabulary

Two independent axes describe every field:

- **Tier** — the consequence of the field staying missing. The four
  tiers live in `core/metadata.py`:
  - `mandatory_hard` — missing it fails import outright.
  - `mandatory_soft` — import still succeeds, but the field shows up in
    `needs_user_input()` until it's resolved.
  - `optional` — no consequence if missing, ever.
  - `computed` — never blocks anything, at import or later. Always
    carries a best-available value (a seed guess if nothing better has
    been supplied yet) that a later stage can refine and overwrite.
- **Resolution strategy** — *how* a missing value actually gets filled
  in. Lives in `core/known_fields.py` (`Strategy`), consumed by
  `completion/engine.py`'s automatic pass:
  - `Derive` — computed automatically from other fields already present
    (e.g. angles from angle span + projection count). The built-in
    derivers live in [`completion/derivers.py`](../completion/derivers.py),
    registered via `completion/engine.py`'s `register_deriver()`.
  - `Default` — a fixed fallback value applied automatically if nothing
    else is available, but always overridable by the user.
  - `Ask` — no safe default exists; something outside `completion/`
    (a notebook cell today, a GUI form later) has to supply it via
    `completion.provide()`.
  - `Calculate` — an algorithm proposes a value from the actual image
    data, and the user confirms or adjusts it (calibration stage only —
    `completion/` never attempts these).

A field can list more than one strategy, tried in order — e.g. `angles`
is derive-then-ask.

## Fields

| Field | Category | Tier | Strategy | Default | Unit | Notes |
|---|---|---|---|---|---|---|
| `angles` | Geometry / scan | `mandatory_soft` | Derive (from `angle_span` + projection count + `rotation_direction`) → else Ask | — | degrees | Projection count is read off the array shape, never stored separately. Was `mandatory_hard` in the vendor plugins; now `mandatory_soft` in [synthetic.py](../data_import/vendors/synthetic.py) and [hypothetical_synchrotron.py](../data_import/vendors/hypothetical_synchrotron.py) since completion can recover it. |
| `geometry_type` | Geometry / scan | `mandatory_soft` | Ask | — | categorical: `parallel` / `cone` | No safe default — a wrong guess silently corrupts every downstream calculation. Gates whether the cone-beam-only rows below apply. |
| `angle_span` | Geometry / scan | `mandatory_soft` | Derive (max − min of `angles`, if present) → else Ask | — | degrees | Covers the "180 / 360 / limited angle" distinction as a plain number rather than a second categorical field that would need to stay in sync with it. |
| `rotation_direction` | Geometry / scan | `optional` | Default | `ccw` | categorical: `cw` / `ccw` | Wrong default only flips orientation, never invalidates the reconstruction. |
| `voxel_size` | Pixel / detector | `mandatory_soft` | Default | `1` | µm | |
| `detector_pixel_size` | Pixel / detector | `mandatory_soft` | Derive (from `voxel_size`) → else Default | `1` | µm | Only meaningfully distinct from `voxel_size` in cone-beam (magnification); `voxel_size` always resolves first (defaults to `1` itself), so in practice this derives every time rather than reaching its own static default. |
| `source_to_detector_distance` | Cone-beam geometry | `mandatory_soft`, conditional on `geometry_type == "cone"` | Ask | — | mm | Unused for parallel-beam projects. |
| `source_to_object_distance` | Cone-beam geometry | `mandatory_soft`, conditional on `geometry_type == "cone"` | Ask | — | mm | Unused for parallel-beam projects. |
| `center_of_rotation` | Alignment | `computed` | Calculate (tool proposes) + user confirms | `0` | pixels (deviation from image center, sub-pixel float) | Never blocks import or reconstruction; reconstruction proceeds on the current value whether or not it's been confirmed. |
| `tilt_angle` | Alignment | `computed` | Calculate (tool proposes) + user confirms | `0` | degrees | Same non-blocking behavior as `center_of_rotation`; an uncorrected tilt degrades quality but doesn't invalidate the run. |
| `operator` | Provenance | `optional` | Ask | — | text | |
| `instrument_name` | Provenance | `optional` | — | — | text | |
| `acquisition_datetime` | Provenance | `optional` | — | — | timestamp | |
| `sample_name` | Provenance | `optional` | Ask | — | text | Useful for naming output volumes later. |
| `energy` | Provenance | `optional` | — (no prompt) | — | keV | Purely a pass-through if the vendor happens to provide it — never asked for, never blocks or affects calculations. |

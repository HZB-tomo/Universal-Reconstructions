"""
The pipeline stage between import and calibration: given a
ProjectObject fresh out of DataImporter, drives every mandatory_soft
field toward resolved, using the strategy core/known_fields.py
specifies for it.

Never blocks waiting for a human. run() does everything it can
automatically; provide() records an answer once something outside this
package (a notebook cell, a GUI) has collected it. See
core/known_fields.status() for a full readout of what remains.

completion never touches computed-tier fields (center_of_rotation,
tilt_angle); those are calibration's job.

Importing this package registers the built-in derivers (angles,
angle_span, detector_pixel_size; see completion/derivers.py).
"""
from completion.engine import provide, register_deriver, run
from completion import derivers  # noqa: F401

__all__ = ["run", "provide", "register_deriver"]

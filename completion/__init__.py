"""
completion/ is the pipeline stage between import and calibration: given
a ProjectObject fresh out of DataImporter, drive every mandatory_soft
field toward resolved, using whichever strategy core/known_fields.py
says applies to it (derive automatically, apply a default, or wait for
something outside this package to supply an answer).

This package deliberately never blocks waiting for a human -- there's
no GUI yet (today: notebook/scripting; later: a GUI), and baking a
blocking prompt into this code would make it useless to whichever one
isn't currently driving it. Instead: run() does everything it can
automatically, and provide() is how an answer gets recorded once
something else (a notebook cell today) has collected it from a user.
See core/known_fields.status() for a full readout of what's left after
either of those runs.

completion never touches "computed"-tier fields (center_of_rotation,
tilt_angle) -- those are calibration's job, not this stage's.

Importing this package registers the built-in derivers (angles,
angle_span, detector_pixel_size -- see completion/derivers.py) as a
side effect, the same way importing data_import registers its built-in
vendor plugins.
"""
from completion.engine import provide, register_deriver, run
from completion import derivers  # noqa: F401 -- registers the built-in derivers

__all__ = ["run", "provide", "register_deriver"]

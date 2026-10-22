"""
Slice axes for the viewer's z / y / x buttons. Qt-free.

A 3D stack (n0, n1, n2) is shown one 2D section at a time:
    z -> axis 0: stack[i]         (projection image / reconstructed xy slice)
    y -> axis 1: stack[:, i, :]   (sinogram for a projection stack / xz slice of a volume)
    x -> axis 2: stack[:, :, i]   (angles x rows / yz slice of a volume)
"""
from __future__ import annotations

AXIS_NAMES = ("z", "y", "x")


def section_index(axis: int, i: int):
    """The numpy-style index selecting section `i` along `axis`. Works on
    ndarrays, h5py datasets and DataInterface.get_projections()."""
    return (slice(None),) * axis + (i,) if axis else i


def axis_labels(axis: int) -> tuple[str, str]:
    """(row axis name, column axis name) of the 2D section shown for `axis`,
    for the cursor readout."""
    return {0: ("y", "x"), 1: ("z", "x"), 2: ("z", "y")}[axis]


def axis_tooltip(axis: int, kind: str) -> str:
    if kind == "projections":
        return {
            0: "z: one projection image per angle (default)",
            1: "y: sinogram of one detector row (angle x column)",
            2: "x: one detector column across all angles (angle x row)",
        }[axis]
    return {
        0: "z: one xy slice at a time (default)",
        1: "y: xz section through the volume",
        2: "x: yz section through the volume",
    }[axis]

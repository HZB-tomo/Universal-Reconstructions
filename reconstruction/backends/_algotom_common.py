"""
Shared helper for every algotom-based reconstruction backend: the
lazy, error-message-annotated import of algotom.rec.reconstruction.
Not a backend itself -- no @register_backend here.
"""
from __future__ import annotations


def algotom_reconstruction(requiring_backend: str):
    try:
        import algotom.rec.reconstruction as rec
    except ImportError as e:
        raise ImportError(
            f"The '{requiring_backend}' reconstruction backend requires the 'algotom' package, which isn't installed. "
            "Install it with: pip install algotom"
        ) from e
    return rec

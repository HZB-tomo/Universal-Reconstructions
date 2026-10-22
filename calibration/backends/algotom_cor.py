"""
Center-of-rotation backend wrapping algotom.prep.calculation.find_center_vo.

Requires the optional `algotom` package, imported lazily. Raises
ImportError with install instructions if it is not installed.

Method reference:
    Vo, N.T., Drakopoulos, M., Atwood, R.C., Reinhard, C., "Reliable
    method for calculating the center of rotation in parallel-beam
    tomography," Optics Express 22(16), 19078-19086 (2014).
    https://doi.org/10.1364/OE.22.019078
"""
from __future__ import annotations

import os
from typing import Optional

import numpy as np

from calibration.backend import CalibrationBackend, register_backend


def _algotom_calculation():
    try:
        import algotom.prep.calculation as calc
    except ImportError as e:
        raise ImportError(
            "The 'algotom' calibration backend requires the 'algotom' package, which isn't installed. "
            "Install it with: pip install algotom"
        ) from e
    return calc


@register_backend
class AlgotomCorBackend(CalibrationBackend):
    name = "algotom"

    def find_center(
        self,
        sino_180: np.ndarray,
        start: Optional[float] = None,
        stop: Optional[float] = None,
        **options,
    ) -> float:
        calc = _algotom_calculation()
        return float(calc.find_center_vo(np.asarray(sino_180), start=start, stop=stop, **options))

    def search_centers(
        self,
        sino_180: np.ndarray,
        start: float,
        stop: float,
        n_candidates: int = 5,
        radius: int = 4,
        step: float = 0.25,
        ratio: float = 0.5,
        **options,
    ) -> list[tuple[float, float]]:
        """
        Vo's metric (find_center_vo) evaluated across the whole window
        instead of only keeping its minimum: every local minimum of the
        coarse curve is a candidate, refined to `step` pixels and scored,
        so a wrong global minimum doesn't hide the right answer.
        Returns up to `n_candidates` (center, metric) pairs, best (lowest
        metric) first. A `ratio`/`options` meaning matches find_center_vo.
        """
        calc = _algotom_calculation()
        from joblib import Parallel, delayed
        from scipy import ndimage as ndi

        sino = np.asarray(sino_180, dtype=np.float32)
        nrow, ncol = sino.shape
        start_c = int(np.clip(np.floor(start), 0, ncol - 1))
        stop_c = int(np.clip(np.ceil(stop), 0, ncol - 1))
        if stop_c <= start_c:
            raise ValueError(f"empty search window [{start}, {stop}] for a sinogram {ncol} columns wide")

        # Same downsampling find_center_vo uses for large sinograms.
        dsp_row = int(np.floor(nrow / 900.0)) if nrow > 1000 else 1
        dsp_col = int(np.floor(ncol / 640.0)) if ncol > 1000 else 1
        coarse = ndi.gaussian_filter(sino, (3, 1), mode="reflect")
        if dsp_row > 1 or dsp_col > 1:
            coarse = calc.downsample_cor(coarse, dsp_row, dsp_col)
        c_nrow, c_ncol = coarse.shape
        flip, comp = np.fliplr(coarse), np.flipud(coarse)
        mask = calc.make_inverse_double_wedge_mask(2 * c_nrow, c_ncol, 0.5 * ratio * c_ncol, None, None)
        positions = np.arange(int(np.floor(start_c / dsp_col)), int(np.ceil(stop_c / dsp_col)) + 1)
        positions = positions[(positions >= 0) & (positions < c_ncol)]
        metrics = np.array(
            Parallel(n_jobs=max(1, (os.cpu_count() or 2) - 1), prefer="threads")(
                delayed(calc.calculate_center_metric)(p, coarse, flip, comp, mask) for p in positions
            ),
            dtype=float,
        )

        # Local minima of the coarse curve (a window edge counts: the true
        # minimum may lie beyond it), kept at least `min_sep` apart, best first.
        left = np.r_[np.inf, metrics[:-1]]
        right = np.r_[metrics[1:], np.inf]
        minima = np.flatnonzero((metrics <= left) & (metrics <= right))
        min_sep = max(3, int(round(0.02 * c_ncol)))
        chosen: list[int] = []
        for idx in minima[np.argsort(metrics[minima])]:
            if all(abs(idx - j) >= min_sep for j in chosen):
                chosen.append(int(idx))
            if len(chosen) >= n_candidates:
                break

        # Refine each on the full-resolution sinogram, and score every
        # refined position with the same metric so they are comparable.
        fine = ndi.gaussian_filter(sino, (2, 2), mode="reflect")
        fine_flip, fine_comp = np.fliplr(fine), np.flipud(fine)
        fine_mask = calc.make_inverse_double_wedge_mask(2 * nrow, ncol, 0.5 * ratio * ncol, None, None)
        refined: list[tuple[float, float]] = []
        for idx in chosen:
            rough = float(positions[idx] * dsp_col + 0.5 * dsp_col)
            center = float(calc.fine_search_cor(
                sino, rough, max(radius, dsp_col), step, ratio, denoise=True, hor_drop=None, ver_drop=None,
            ))
            if any(abs(center - c) < 1.0 for c, _ in refined):
                continue
            refined.append((center, float(calc.calculate_center_metric(center, fine, fine_flip, fine_comp, fine_mask))))
        refined.sort(key=lambda cm: cm[1])
        # Remember whether the winner is a coarse minimum sitting on the window edge.
        self.last_search_at_edge = bool(chosen) and chosen[0] in (0, len(positions) - 1)
        return refined

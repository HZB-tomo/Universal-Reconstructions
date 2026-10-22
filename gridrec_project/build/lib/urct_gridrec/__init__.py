"""
urct_gridrec: gridrec tomographic reconstruction for CPU (Numba) and GPU
(CuPy + numba.cuda). A standalone prototype; nothing here imports URCT.

    from urct_gridrec import gridrec
    volume = gridrec(sinograms, angles_deg, center)     # (angles, slices, width) -> (slices, width, width)
"""
from urct_gridrec.api import gpu_available, gridrec
from urct_gridrec.core import Plan, make_plan

__all__ = ["gridrec", "gpu_available", "make_plan", "Plan"]

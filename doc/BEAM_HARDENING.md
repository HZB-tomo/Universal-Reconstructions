# Beam hardening correction

What beam hardening is, what the software does about it, and how to use it.
Code: `preprocessing/beam_hardening.py` (the correction and the fits),
`orchestration/preprocess.py` (where it runs).

## The physics, briefly

Filtered backprojection assumes **Beer-Lambert's law for one X-ray energy**: the
measured attenuation `p = -ln(I / I0)` is proportional to the thickness `L` of
material the beam went through (`p = mu * L`).

A real beam is a **spread of energies** (a laboratory X-ray tube, a white or
pink synchrotron beam; neutron beams from a reactor behave the same way).
Low-energy photons are absorbed first, so the part of the beam that gets
through more material has a higher average energy: it is **hardened**, and
absorbed less per centimetre. The measured attenuation

```
p(L) = -ln( integral S(E) exp(-mu(E) * L) dE / integral S(E) dE )
```

bends away from the straight line `mu * L` (it is concave). Reconstructed as if
it were linear, a uniform object shows **cupping** (the middle is too dark, the
edge too bright) and dense parts cast dark **streaks** onto their neighbours.
Quantitative values (attenuation coefficients) are wrong.

## What the correction does

It replaces the measured attenuation `p` by a polynomial of it,

```
p_corrected = c1*p + c2*p^2 + c3*p^3 + ...
```

chosen so that `p_corrected` is proportional to thickness again. For one
dominant material this removes the cupping almost completely. (A sample with
two very different materials, such as bone in tissue, cannot be linearized by one
polynomial; that needs an iterative correction, which is not implemented.)

`(c1, c2, c3) = (1, 0, 0)` changes nothing. Typical hardening gives a positive
`c2` (thick parts are raised) and a small negative `c3`.

Like every preprocessing step it works on transmission and returns
transmission (the reconstruction backends take the `-log` themselves):
`T_out = exp(-g(-ln T))`. Values <= 0 count as 1, like the backends do;
T > 1 (noise) is continued with slope `c1`; above the largest attenuation
where the polynomial is still increasing (`valid_attenuation_limit()`) it
continues as a straight line, so the map never turns around. A polynomial that
is not increasing at least up to attenuation 3 (T = 5%) is refused with an
error, because two different thicknesses would map to the same value.

## Where the coefficients come from

1. **A calibration.** Measure the attenuation through known thicknesses (a step
   wedge, or a cylinder whose chord lengths are known) and fit
   `fit_polynomial(measured, mu_reference * thickness, order=3)`.
2. **A model of the beam.** If you know the spectrum and the material's `mu(E)`:

   ```python
   from preprocessing import coefficients_from_spectrum

   c = coefficients_from_spectrum(energies_kev, spectrum, mu_per_cm, max_thickness_cm=1.2, order=3)
   ```

   The target is a straight line with the beam's initial slope (thin-object
   limit), or `mu` at a `reference_energy_kev` you choose.
3. **By eye** (the GUI): reconstruct a preview slice of a uniform object and
   raise `c2` until the middle is as bright as the edge.

In a test with a 20-80 keV spectrum through an aluminium-like cylinder, the
uncorrected reconstruction had its middle 11% too dark (centre/edge = 0.89);
the fitted cubic brought it to 1.00.

## Using it

### In the GUI

Refinements tab, "Beam hardening" group: tick the box and adjust `c1`, `c2`,
`c3`. The line below shows what the current values do to attenuations of 1, 2
and 3, or, in red, why they cannot be used. With "Auto-update preview" on in
Basic settings the preview follows every change, and values already tried come
back instantly.

### In a run

```ini
[preprocessing]
beam_hardening = true
bh_coefficients = 1, 0.3, -0.02     # c1, c2, c3, ...
```

Python: `ProjectConfig(beam_hardening=True, bh_coefficients=(1.0, 0.3, -0.02))`;
`bh_options={"max_attenuation": 8.0}` (code-only) sets how far the polynomial is
used. A missing or unusable set of coefficients stops the stage with an error
(before anything is written in a streamed run).

### As a function

```python
from preprocessing import correct_beam_hardening

T_corrected = correct_beam_hardening(T, coefficients=(1.0, 0.3, -0.02))
```

## Where it sits in the pipeline

```
flat/dark normalization -> zinger removal -> phase retrieval -> ring removal -> beam hardening -> reconstruction
```

Beam hardening is last on purpose: it is non-linear, and ring (stripe) removal
assumes the stripes have not yet been stretched by it. It is a per-value
operation, so it needs no halo and streamed, eager and preview results agree.

## Not covered yet

Multi-material (iterative) correction, estimating the coefficients automatically
from a preview reconstruction, and loading spectra/attenuation tables in the GUI.

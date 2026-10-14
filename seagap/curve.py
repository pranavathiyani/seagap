"""Accumulation-curve fitting with heteroscedastic standardisation.

Shared by steps 09 and 10 so both use the same, corrected version.

The scatter of accumulation residuals shrinks as sample size grows, so residuals
must be divided by the sd expected at their own n rather than one global number.
An earlier version modelled that sd as a LINEAR function of log10(n):

    sd(n) = (c0 + c1 * log10 n) * sqrt(pi/2)

with c1 negative. That crosses zero and turns negative at large n. Clamping the
result at 1e-6 then produced a z of -114642 for the one order with 6,194
compounds: a modelling failure rendered as a number rather than an error.

The sd is now modelled on the log scale, which cannot go negative:

    sd(n) = exp(a + b * log10 n) * sqrt(pi/2)

and a floor is applied as a fraction of the global residual sd, with the floor
reported so that hitting it is visible rather than silent.
"""

import numpy as np

SD_FLOOR_FRAC = 0.25      # never let the local sd fall below this x the global sd


def fit_accumulation(n_units, n_classes, sd_floor_frac=SD_FLOOR_FRAC):
    """Fit n_classes = a * n_units^b and standardise residuals locally.

    Returns a dict with the fit, the residuals, the local sd, z, and diagnostics
    including how many points hit the sd floor.
    """
    n_units = np.asarray(n_units, dtype=float)
    n_classes = np.asarray(n_classes, dtype=float)
    if np.any(n_units <= 0) or np.any(n_classes <= 0):
        raise ValueError("accumulation fit needs positive counts")

    x = np.log10(n_units)
    y = np.log10(n_classes)
    b1, b0 = np.polyfit(x, y, 1)
    resid = y - (b0 + b1 * x)
    r2 = 1 - (resid ** 2).sum() / ((y - y.mean()) ** 2).sum()
    global_sd = float(resid.std(ddof=1))

    # Log-scale model of the absolute residual: strictly positive at any n.
    # Exponentiating a fit of log|resid| recovers the GEOMETRIC mean, which sits
    # below the arithmetic mean by Jensen's inequality, so the sd comes out too
    # small and z too large. The sqrt(pi/2) half-normal factor does not fix that.
    # The shape is right, so the level is calibrated afterwards by rescaling until
    # z has unit sd overall, leaving the n-dependence intact.
    eps = max(1e-4, 0.01 * global_sd)
    a1, a0 = np.polyfit(x, np.log(np.abs(resid) + eps), 1)
    sd_shape = np.exp(a0 + a1 * x)
    calib = float((resid / sd_shape).std(ddof=1))
    sd_model = sd_shape * calib

    floor = sd_floor_frac * global_sd
    n_floored = int((sd_model < floor).sum())
    sd_local = np.maximum(sd_model, floor)

    return {
        "a": float(10 ** b0), "b": float(b1), "r2": float(r2),
        "resid": resid, "sd_local": sd_local, "z": resid / sd_local,
        "global_sd": global_sd,
        "sd_log_intercept": float(a0), "sd_log_slope": float(a1),
        "sd_calibration": calib,
        "sd_floor": float(floor), "n_floored": n_floored,
        "pred": 10 ** (b0 + b1 * x),
    }


def sd_at(fit, n):
    """Local sd the fit implies at sample size n, after calibration and flooring."""
    v = np.exp(fit["sd_log_intercept"]
               + fit["sd_log_slope"] * np.log10(float(n))) * fit["sd_calibration"]
    return max(v, fit["sd_floor"])

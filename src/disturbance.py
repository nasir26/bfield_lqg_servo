"""Ambient magnetic disturbance at the trap.

[SIMULATED] This is a synthetic disturbance model, not a measured spectrum.

Content, referred to the field at the trap and quoted in nT:

  * mains fundamental at 50 Hz with harmonics at 100, 150 and 250 Hz,
  * each harmonic with independently drifting amplitude and phase,
  * a 1/f thermal/DC drift term,
  * a white broadband floor.

The drift is the part that matters for the controller design. Real mains hum is
not phase-locked to anything the servo controls: the amplitude and phase of each
harmonic wander. That wander is precisely what the estimator has to track, and
its rate is what sets the process-noise intensity `Q` on the internal-model
oscillator states.

Those two numbers must agree. `harmonic_process_noise_intensity()` derives the
`Q` entries from the same random-walk rates used to synthesise the signal here,
so the estimator is tuned to the disturbance it will actually see rather than to
a hand-picked constant. `tests/test_consistency.py` asserts the agreement.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

NT = 1e-9  # tesla per nanotesla


@dataclass(frozen=True)
class Harmonic:
    freq_hz: float
    amplitude_nT: float
    # Random-walk rates. Units are "per sqrt(second)": the standard deviation
    # accumulated by a one-second walk.
    amp_rw_nT_per_rts: float
    phase_rw_rad_per_rts: float


@dataclass(frozen=True)
class DisturbanceParams:
    harmonics: Tuple[Harmonic, ...] = (
        Harmonic(50.0, 100.0, 0.5, 0.05),
        Harmonic(100.0, 30.0, 0.2, 0.05),
        Harmonic(150.0, 20.0, 0.15, 0.05),
        Harmonic(250.0, 8.0, 0.08, 0.05),
    )
    drift_nT_per_rts: float = 2.0      # 1/f-ish thermal/DC wander
    broadband_nT_rtHz: float = 0.05    # white floor
    seed: int = 0

    @property
    def freqs_hz(self) -> List[float]:
        return [h.freq_hz for h in self.harmonics]


def nominal() -> DisturbanceParams:
    return DisturbanceParams()


def generate(
    p: DisturbanceParams,
    fs_hz: float,
    n: int,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Disturbance field at the trap, in tesla, length `n` at rate `fs_hz`.

    Each harmonic is synthesised as a sinusoid whose amplitude and phase each
    follow an independent random walk, so the signal is narrowband but not
    coherent with any fixed reference.
    """
    if rng is None:
        rng = np.random.default_rng(p.seed)
    dt = 1.0 / fs_hz
    t = np.arange(n) * dt
    out = np.zeros(n, dtype=float)

    sqrt_dt = np.sqrt(dt)
    for h in p.harmonics:
        amp = h.amplitude_nT + np.cumsum(rng.normal(0.0, h.amp_rw_nT_per_rts * sqrt_dt, n))
        # Amplitude is physically non-negative; reflect rather than clip so the
        # walk does not pile up against a barrier.
        amp = np.abs(amp)
        phase = np.cumsum(rng.normal(0.0, h.phase_rw_rad_per_rts * sqrt_dt, n))
        out += amp * np.sin(2.0 * np.pi * h.freq_hz * t + phase)

    drift = np.cumsum(rng.normal(0.0, p.drift_nT_per_rts * sqrt_dt, n))
    out += drift

    # White floor: an amplitude spectral density in nT/sqrt(Hz) corresponds to a
    # per-sample standard deviation of density * sqrt(fs/2) * sqrt(2) = density
    # * sqrt(fs) for a real signal sampled at fs.
    out += rng.normal(0.0, p.broadband_nT_rtHz * np.sqrt(fs_hz), n)

    return out * NT


def harmonic_process_noise_intensity(p: DisturbanceParams) -> np.ndarray:
    """Continuous-time process-noise intensity for each oscillator state pair.

    Returns a 1-D array of length 2*len(harmonics), giving the diagonal `Q`
    entries for the (sine, cosine) state pair of each harmonic, in T^2/s.

    Derivation. A harmonic modelled as `a(t) sin(w t + phi(t))` is carried by two
    quadrature states whose magnitude is the amplitude `a`. An amplitude random
    walk of rate `sigma_a` per sqrt(s) injects variance `sigma_a^2` per second
    directly. A phase random walk of rate `sigma_p` per sqrt(s) rotates the
    state vector; at amplitude `a` that displaces the tip at rate `a * sigma_p`,
    contributing `(a * sigma_p)^2` per second. The two are independent, so the
    intensities add.

    This is why the oscillator states are controllable from `w`. Built with zero
    process noise they would sit exactly on the imaginary axis (unit circle,
    discretised) and be uncontrollable, and the Riccati solve would fail or
    return a meaningless gain. The fix is physical: mains amplitude and phase
    genuinely do drift, and this function is that drift written down. Damping the
    poles to radius 1-eps would also "work", and would bury a tuning knob in the
    model where nobody finds it later.
    """
    q = []
    for h in p.harmonics:
        amp_var = (h.amp_rw_nT_per_rts * NT) ** 2
        phase_var = (h.amplitude_nT * NT * h.phase_rw_rad_per_rts) ** 2
        intensity = amp_var + phase_var
        q.extend([intensity, intensity])
    return np.asarray(q, dtype=float)


def drift_process_noise_intensity(p: DisturbanceParams) -> float:
    """Continuous-time process-noise intensity for the drift integrator, T^2/s."""
    return float((p.drift_nT_per_rts * NT) ** 2)


def broadband_variance(p: DisturbanceParams, fs_hz: float) -> float:
    """Per-sample variance of the white floor at rate `fs_hz`, in T^2."""
    return float((p.broadband_nT_rtHz * NT * np.sqrt(fs_hz)) ** 2)

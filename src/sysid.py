"""System identification from excitation records.

[SIMULATED] The "measurements" are synthesised from `src.plant`. The point of
this module is that it is the *same code path* that will run on real measured
data later: nothing downstream knows whether the frequency response came from a
model or a lock-in amplifier.

Two disciplines are enforced here because both are easy to skip and both make
the difference between a design that survives hardware and one that does not:

1. **Coherence gating.** Frequency bins where the input and output are not
   coherent carry no usable information about the plant; they carry noise. Bins
   below `COHERENCE_FLOOR` are dropped from the fit and the plot says how many.
2. **Held-out validation.** The model is fitted on a chirp and scored on a
   multisine it has never seen. The reported fit percentage is the validation
   one. A model scored on its own training record will flatter itself and then
   produce an LQG design that fails on hardware.

Only `B_sensor` is identified, because only `B_sensor` is measurable. The
trap-field path is known to the *simulation* but is not available to the
identification, exactly as on hardware.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import control as ct
import numpy as np
from scipy import signal

COHERENCE_FLOOR = 0.9


@dataclass
class ExcitationRecord:
    """One input/output record at a fixed sample rate."""

    u: np.ndarray
    y: np.ndarray
    fs_hz: float
    kind: str

    @property
    def n(self) -> int:
        return int(self.u.size)


@dataclass
class FrequencyResponse:
    f_hz: np.ndarray
    G: np.ndarray            # complex response, sensor path
    coherence: np.ndarray
    kept: np.ndarray         # bool mask of bins passing the coherence gate

    @property
    def n_kept(self) -> int:
        return int(self.kept.sum())


def chirp_excitation(
    fs_hz: float,
    duration_s: float,
    f0_hz: float = 0.5,
    f1_hz: float = 5e3,
    amplitude_v: float = 1.0,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Logarithmic chirp with low-frequency amplitude scheduling.

    The coil is a lowpass: a constant-amplitude drive that is comfortable at
    1 kHz will push the current far outside its linear range at 1 Hz. The
    schedule tapers the drive at low frequency so the record stays linear, which
    is the difference between identifying the plant and identifying the
    amplifier's clipping behaviour.
    """
    t = np.arange(int(fs_hz * duration_s)) / fs_hz
    x = signal.chirp(t, f0=f0_hz, f1=f1_hz, t1=t[-1], method="logarithmic")
    # instantaneous frequency of a log chirp
    f_inst = f0_hz * (f1_hz / f0_hz) ** (t / t[-1])
    schedule = np.clip(f_inst / max(f0_hz * 20.0, 1e-9), 0.05, 1.0)
    return amplitude_v * schedule * x


def multisine_excitation(
    fs_hz: float,
    duration_s: float,
    f_lo_hz: float = 1.0,
    f_hi_hz: float = 3e3,
    n_tones: int = 60,
    amplitude_v: float = 1.0,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Random-phase multisine, used only for validation.

    Deliberately a different excitation class from the chirp: a model that has
    merely memorised the chirp's time-frequency trajectory will not reproduce a
    signal whose tones are all present at once.
    """
    if rng is None:
        rng = np.random.default_rng(1)
    t = np.arange(int(fs_hz * duration_s)) / fs_hz
    freqs = np.geomspace(f_lo_hz, f_hi_hz, n_tones)
    phases = rng.uniform(0, 2 * np.pi, n_tones)
    x = np.zeros_like(t)
    for f, ph in zip(freqs, phases):
        x += np.sin(2 * np.pi * f * t + ph) / np.sqrt(n_tones)
    # normalise to the requested peak drive
    x = x / np.max(np.abs(x)) * amplitude_v
    return x


def simulate_record(
    sys_ct: ct.StateSpace,
    u: np.ndarray,
    fs_hz: float,
    kind: str,
    sensor_noise_T: float,
    rng: np.random.Generator,
    disturbance: Optional[np.ndarray] = None,
) -> ExcitationRecord:
    """Run an excitation through the plant and return the measured record.

    `sys_ct` has inputs [u, d] and outputs [B_trap, B_meas]; only B_meas is
    returned, because only B_meas exists on hardware.
    """
    n = u.size
    t = np.arange(n) / fs_hz
    d = np.zeros(n) if disturbance is None else disturbance[:n]
    U = np.vstack([u, d])
    _, yout = ct.forced_response(sys_ct, T=t, U=U)
    y_meas = np.asarray(yout)[1]
    y_meas = y_meas + rng.normal(0.0, sensor_noise_T, n)
    return ExcitationRecord(u=u, y=y_meas, fs_hz=fs_hz, kind=kind)


def empirical_transfer(
    rec: ExcitationRecord, nperseg: int = 1 << 13
) -> FrequencyResponse:
    """Welch-averaged ETFE with coherence, coherence-gated."""
    f, Pyu = signal.csd(rec.y, rec.u, fs=rec.fs_hz, nperseg=nperseg)
    _, Puu = signal.welch(rec.u, fs=rec.fs_hz, nperseg=nperseg)
    _, coh = signal.coherence(rec.u, rec.y, fs=rec.fs_hz, nperseg=nperseg)
    with np.errstate(divide="ignore", invalid="ignore"):
        G = np.where(Puu > 0, Pyu / Puu, 0.0)
    kept = (coh >= COHERENCE_FLOOR) & (f > 0)
    return FrequencyResponse(f_hz=f, G=G, coherence=coh, kept=kept)


def stepped_sine_response(
    sys_ct: ct.StateSpace,
    freqs_hz: np.ndarray,
    fs_hz: float,
    sensor_noise_T: float,
    rng: np.random.Generator,
    amplitude_v: float = 1.0,
    cycles: int = 40,
    settle_cycles: int = 10,
) -> FrequencyResponse:
    """Lock-in (stepped-sine) frequency response of the sensor path.

    Why this rather than a chirp ETFE. This plant spans about ten decades of
    gain between DC and 10 kHz. An FFT-based estimate from a broadband
    excitation is limited by spectral leakage: the very large low-frequency
    content of `y` leaks into high-frequency bins where the true response is
    orders of magnitude smaller, and the estimate there is leakage rather than
    signal. Measured against the known plant, a noiseless chirp ETFE still had a
    median relative error near 0.8 for exactly this reason.

    Stepped sine has no such limit. One frequency is driven at a time and the
    response is recovered by correlating against quadrature references over a
    whole number of cycles, which rejects everything at other frequencies.
    Dynamic range is then set by measurement noise and averaging time, not by
    window sidelobes. It is also what one actually does on hardware for a plant
    with this much rolloff.

    Coherence is not defined for a single-tone measurement in the Welch sense,
    so a per-frequency signal-to-noise proxy is used in its place: the ratio of
    the recovered in-band amplitude to the residual after the fitted tone is
    removed. It is written into `coherence` so the same gating logic applies
    downstream.
    """
    G = np.zeros(freqs_hz.size, dtype=complex)
    snr = np.zeros(freqs_hz.size, dtype=float)

    # Settling must be governed by the *plant's* slowest mode, not by the drive
    # period. The coil L/R pole near 32 Hz has a time constant of about 5 ms, so
    # at 5 kHz a wait of ten drive cycles is 2 ms and the measurement window is
    # still dominated by an unsettled transient that is orders of magnitude
    # larger than the true high-frequency response. Ignoring this was worth a
    # median relative error near 0.8.
    poles = np.linalg.eigvals(np.asarray(sys_ct.A))
    slow = poles[poles.real < 0].real
    tau_slow = 1.0 / np.min(np.abs(slow)) if slow.size else 0.0
    n_settle_plant = int(np.ceil(8.0 * tau_slow * fs_hz))

    for i, f in enumerate(freqs_hz):
        n_settle = max(int(np.ceil(settle_cycles * fs_hz / f)), n_settle_plant)
        # Use a whole number of drive cycles so the quadrature references are
        # exactly orthogonal to everything at other frequencies.
        samples_per_cycle = fs_hz / f
        n_meas = int(round(cycles * samples_per_cycle))
        n = n_settle + n_meas
        t = np.arange(n) / fs_hz
        u = amplitude_v * np.sin(2 * np.pi * f * t)
        _, yout = ct.forced_response(sys_ct, T=t, U=np.vstack([u, np.zeros(n)]))
        y = np.asarray(yout)[1] + rng.normal(0.0, sensor_noise_T, n)

        # Correlate over the measurement window only, after the transient.
        tm = t[n_settle:]
        ym = y[n_settle:]
        ref_s = np.sin(2 * np.pi * f * tm)
        ref_c = np.cos(2 * np.pi * f * tm)
        a = 2.0 * np.mean(ym * ref_s)
        b = 2.0 * np.mean(ym * ref_c)
        resp = (a + 1j * b) / amplitude_v
        G[i] = resp

        residual = ym - (a * ref_s + b * ref_c)
        amp = np.hypot(a, b)
        rms_res = np.sqrt(np.mean(residual**2))
        snr[i] = amp / (amp + np.sqrt(2.0) * rms_res + 1e-300)

    kept = snr >= COHERENCE_FLOOR
    return FrequencyResponse(f_hz=np.asarray(freqs_hz, dtype=float), G=G,
                             coherence=snr, kept=kept)


def _initial_poles(f_hz: np.ndarray, order: int) -> np.ndarray:
    """Complex-conjugate starting poles spread over the measured band.

    Gustavsen & Semlyen's recommendation: weakly damped pairs distributed
    logarithmically across the fitted range, so every decade has something to
    latch onto.
    """
    n_pairs = order // 2
    out: List[complex] = []
    if n_pairs > 0:
        betas = 2.0 * np.pi * np.geomspace(f_hz.min(), f_hz.max(), n_pairs)
        for b in betas:
            alpha = -b / 100.0
            out.extend([alpha + 1j * b, alpha - 1j * b])
    if order % 2 == 1:
        out.append(-2.0 * np.pi * float(np.exp(np.mean(np.log(f_hz)))))
    return np.asarray(out[:order], dtype=complex)


def _pair_flags(poles: np.ndarray) -> np.ndarray:
    """Classify each pole: 0 real, 1 first of a conjugate pair, 2 second of one."""
    flags = np.zeros(poles.size, dtype=int)
    k = 0
    while k < poles.size:
        if abs(poles[k].imag) < 1e-12:
            flags[k] = 0
            k += 1
        else:
            flags[k] = 1
            if k + 1 < poles.size:
                flags[k + 1] = 2
            k += 2
    return flags


def _pole_basis(s: np.ndarray, poles: np.ndarray):
    """Real-valued partial-fraction basis.

    A complex pole pair contributes two *real* basis functions,

        1/(s-a) + 1/(s-abar)      and      j/(s-a) - j/(s-abar),

    whose coefficients are Re(c) and Im(c). Expanding a conjugate pair as two
    independent complex columns and then solving a real least squares silently
    forces the residues to be real, which cannot represent a resonant pair at
    all: the fitter could not reproduce even an exact two-pole system until this
    was fixed.
    """
    pair = _pair_flags(poles)
    cols = []
    for k in range(poles.size):
        if pair[k] == 0:
            cols.append(1.0 / (s - poles[k]))
        elif pair[k] == 1:
            a = poles[k]
            cols.append(1.0 / (s - a) + 1.0 / (s - np.conj(a)))
        else:
            a = poles[k - 1]
            cols.append(1j / (s - a) - 1j / (s - np.conj(a)))
    return np.column_stack(cols), pair


def _coef_to_residues(coef: np.ndarray, pair: np.ndarray) -> np.ndarray:
    """Map real basis coefficients back to complex residues."""
    res = np.zeros(coef.size, dtype=complex)
    for k in range(coef.size):
        if pair[k] == 0:
            res[k] = coef[k]
        elif pair[k] == 1:
            res[k] = coef[k] + 1j * (coef[k + 1] if k + 1 < coef.size else 0.0)
        else:
            res[k] = np.conj(res[k - 1])
    return res


def _relocate(poles: np.ndarray, sigma_coef: np.ndarray, pair: np.ndarray):
    """New poles = zeros of sigma, as eigenvalues of a real companion form.

    For a conjugate pair the 2x2 real block is [[Re a, Im a], [-Im a, Re a]]
    with input vector [2, 0], which is the real similarity transform of
    diag(a, abar) with input [1, 1].
    """
    n = poles.size
    A = np.zeros((n, n), dtype=float)
    b = np.zeros(n, dtype=float)
    for k in range(n):
        if pair[k] == 0:
            A[k, k] = poles[k].real
            b[k] = 1.0
        elif pair[k] == 1:
            re, im = poles[k].real, poles[k].imag
            A[k, k] = re
            A[k, k + 1] = im
            A[k + 1, k] = -im
            A[k + 1, k + 1] = re
            b[k] = 2.0
            b[k + 1] = 0.0
    H = A - np.outer(b, sigma_coef)
    new = np.linalg.eigvals(H)
    if not np.all(np.isfinite(new)):
        return None
    new = np.where(new.real > 0, -new.real + 1j * new.imag, new)
    # Sort so conjugate partners are adjacent, which the basis builder assumes.
    order_idx = np.lexsort((-np.abs(new.imag), new.real))
    return new[order_idx]


def vector_fit(
    fr: FrequencyResponse,
    order: int,
    n_iter: int = 12,
) -> ct.StateSpace:
    """Rational fit of a measured frequency response by vector fitting.

    Reference: B. Gustavsen and A. Semlyen, "Rational approximation of frequency
    domain responses by vector fitting", IEEE Trans. Power Delivery 14(3),
    1052-1061 (1999).

    Why not a monomial least squares. Writing the fit as a ratio of polynomials
    in `s` and solving for their coefficients requires a Vandermonde-like basis
    in `s^k`. Over the four decades measured here that basis is hopeless: the
    condition number was 4e18 at order 3 and 9e31 at order 6, so the solve
    returned noise and the fitted models were unusable above order 3 even after
    frequency normalisation.

    Vector fitting instead expands in partial fractions over a set of poles, and
    then *relocates* those poles by solving a linear problem whose zeros are the
    new poles. Each iteration is well conditioned because the basis functions
    1/(s - a_n) are bounded and localised, and the pole set converges in a
    handful of passes.

    Poles are kept in complex-conjugate pairs and any pole that lands in the
    right half plane is reflected, since the identified plant is known stable
    and a right-half pole is a fitting artefact.
    """
    f = fr.f_hz[fr.kept]
    G = fr.G[fr.kept]
    s = 1j * 2.0 * np.pi * f
    # Fit the response scaled to unit magnitude; the scale is restored at the
    # end. Without this the residues span the same ten decades the plant does.
    scale = float(np.max(np.abs(G)))
    Gs = G / scale

    poles = _initial_poles(f, order)
    # Relative weighting would put ~4000x more weight on the smallest-magnitude
    # points across this plant's 72 dB range and starve the band the loop
    # actually uses, so weight mildly rather than by 1/|G|.
    weights = 1.0 / np.sqrt(np.maximum(np.abs(Gs), np.max(np.abs(Gs)) * 1e-6))

    for _ in range(n_iter):
        basis, pair = _pole_basis(s, poles)
        A = np.hstack([basis, np.ones((s.size, 1), dtype=complex), -Gs[:, None] * basis])
        Aw = A * weights[:, None]
        bw = Gs * weights
        x, *_ = np.linalg.lstsq(np.vstack([Aw.real, Aw.imag]),
                                np.concatenate([bw.real, bw.imag]), rcond=None)
        sigma_coef = x[poles.size + 1:]
        new_poles = _relocate(poles, sigma_coef, pair)
        if new_poles is None:
            break
        poles = new_poles

    basis, pair = _pole_basis(s, poles)
    A = np.hstack([basis, np.ones((s.size, 1), dtype=complex)])
    Aw = A * weights[:, None]
    bw = Gs * weights
    x, *_ = np.linalg.lstsq(np.vstack([Aw.real, Aw.imag]),
                            np.concatenate([bw.real, bw.imag]), rcond=None)
    residues = _coef_to_residues(x[:poles.size], pair)
    d_term = x[poles.size]

    # Pole-residue -> transfer function, then state space. Building the
    # polynomial from conjugate-paired roots keeps the coefficients real.
    n = poles.size
    den = np.real(np.poly(poles))
    num = np.zeros(n, dtype=complex)
    for k in range(n):
        others = np.delete(poles, k)
        num = num + residues[k] * np.poly(others)
    num = np.real(num)
    num_full = np.concatenate([[0.0], num]) + d_term * den
    sysr = ct.tf2ss(ct.tf(num_full * scale, den))
    return _stabilise(sysr)


def fit_state_space(fr: FrequencyResponse, order: int) -> ct.StateSpace:
    """Monomial-basis rational fit. SUPERSEDED by `vector_fit`; kept for contrast.

    Retained because the comparison is instructive rather than because it is
    used: even with frequency normalisation its basis is conditioned at 1e11 to
    1e18 over this measurement band, and the fits it returns are unusable above
    order three. `order_sweep` calls `vector_fit`.

    Uses a stabilised vector-fitting-style least squares on the gated bins:
    solve for numerator/denominator coefficients of a rational function in
    `s = j2πf`, then realise. Poles found in the right half plane are reflected,
    since the identified physical plant is known to be stable and a right-half
    pole here is a fitting artefact rather than a discovery.
    """
    f = fr.f_hz[fr.kept]
    G = fr.G[fr.kept]

    # Frequency normalisation. The monomial basis s^k over four decades is
    # hopelessly conditioned -- measured cond(A) was 4e18 at order 3 and 9e31 at
    # order 6, so the least squares was solving noise. Fitting in a normalised
    # variable sh = s/w0, with w0 the geometric-mean frequency, keeps every
    # column within a few orders of magnitude of the others; the resulting
    # coefficients are unscaled afterwards.
    w0 = float(np.exp(np.mean(np.log(2.0 * np.pi * f))))
    sh = 1j * (2.0 * np.pi * f) / w0

    # Rows: G(sh) * (sh^n + a_{n-1} sh^{n-1} + ... + a_0) = b_m sh^m + ... + b_0
    n = order
    m = order  # allow a proper-or-biproper fit; leading b may come out ~0
    A_cols = []
    for k in range(n):                      # -G * sh^k  -> denominator coeffs a_k
        A_cols.append(-G * sh**k)
    for k in range(m + 1):                  # sh^k       -> numerator coeffs b_k
        A_cols.append(sh**k)
    A = np.column_stack(A_cols)
    rhs = G * sh**n

    # Weight by 1/|G| so the fit is relative rather than dominated by the
    # low-frequency gain, which is orders of magnitude larger than the tail.
    w = 1.0 / np.maximum(np.abs(G), np.abs(G).max() * 1e-6)
    Aw = A * w[:, None]
    rhsw = rhs * w

    # Solve the complex least squares as a real one so conjugate symmetry holds.
    Ar = np.vstack([Aw.real, Aw.imag])
    br = np.concatenate([rhsw.real, rhsw.imag])
    coef, *_ = np.linalg.lstsq(Ar, br, rcond=None)

    a_hat = np.concatenate([coef[:n], [1.0]])[::-1]   # den in sh, highest first
    b_hat = coef[n:][::-1]                            # num in sh, highest first

    # Undo the normalisation: a coefficient multiplying sh^k = (s/w0)^k carries
    # a factor w0^-k, so coefficient j (counting from the highest power) is
    # divided by w0 raised to that power.
    na = a_hat.size - 1
    nb = b_hat.size - 1
    a = np.array([a_hat[j] / w0 ** (na - j) for j in range(a_hat.size)], dtype=float)
    b = np.array([b_hat[j] / w0 ** (nb - j) for j in range(b_hat.size)], dtype=float)

    sys = ct.tf2ss(ct.tf(b, a))
    return _stabilise(sys)


def _stabilise(sys: ct.StateSpace) -> ct.StateSpace:
    """Reflect any right-half-plane poles into the left half plane."""
    poles = np.linalg.eigvals(np.asarray(sys.A))
    if np.all(poles.real < 0):
        return sys
    tfm = ct.ss2tf(sys)
    num = np.asarray(tfm.num[0][0], dtype=float)
    den = np.asarray(tfm.den[0][0], dtype=float)
    r = np.roots(den)
    r = np.where(r.real > 0, -r.real + 1j * r.imag, r)
    den_new = np.real(np.poly(r)) * den[0]
    return ct.tf2ss(ct.tf(num, den_new))


def fit_percentage(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Standard NRMSE fit percentage: 100 * (1 - ||y-yhat|| / ||y-mean(y)||)."""
    err = np.linalg.norm(y_true - y_pred)
    ref = np.linalg.norm(y_true - np.mean(y_true))
    if ref == 0:
        return float("nan")
    return float(100.0 * (1.0 - err / ref))


def validate(sys: ct.StateSpace, rec: ExcitationRecord) -> Tuple[float, np.ndarray]:
    """Score a model on a record it was not fitted to."""
    t = np.arange(rec.n) / rec.fs_hz
    _, yhat = ct.forced_response(sys, T=t, U=rec.u)
    yhat = np.asarray(yhat).reshape(-1)
    # Skip the first 5% to let the simulation transient settle.
    k = max(1, rec.n // 20)
    return fit_percentage(rec.y[k:], yhat[k:]), yhat


def order_sweep(
    fr: FrequencyResponse,
    validation: ExcitationRecord,
    orders: List[int],
) -> Dict[int, Dict[str, float]]:
    """Fit each order and score it on held-out data.

    Returns validation fit per order, so the order is chosen from a curve rather
    than by hand.
    """
    out: Dict[int, Dict[str, float]] = {}
    for o in orders:
        try:
            sys = vector_fit(fr, o)
            vfit, _ = validate(sys, validation)
            out[o] = {"validation_fit_pct": vfit, "n_states": int(sys.nstates)}
        except Exception as exc:  # a failed fit is a result, not a crash
            out[o] = {"validation_fit_pct": float("nan"), "error": str(exc)}
    return out

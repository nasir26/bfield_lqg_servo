#!/usr/bin/env python
"""Checkpoint 1: plant and disturbance model, open-loop response.

[SIMULATED] Produces figures/fig01_openloop.pdf and results/checkpoint1.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import control as ct

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src import disturbance as dist
from src import plant as pl
from src.style import setup_style, save_fig, SINGLE, DOUBLE


def main() -> None:
    setup_style()
    p = pl.nominal()
    dp = dist.nominal()
    sysm = pl.build(p)

    w = 2.0 * np.pi * np.logspace(-1, 5, 2000)

    # u -> B_trap and u -> B_meas
    P_ut = pl.drive_to_trap(p)
    H_ts = pl.sensor_transfer(p)
    M = pl.measurement_chain(p)

    mag_ut, ph_ut, _ = ct.frequency_response(P_ut, w)
    mag_um, ph_um, _ = ct.frequency_response(P_ut * H_ts * M, w)
    mag_ts, ph_ts, _ = ct.frequency_response(H_ts * M, w)
    mag_geom, _, _ = ct.frequency_response(H_ts, w)   # geometry alone, no instrument

    f = w / (2.0 * np.pi)

    fig, axes = plt.subplots(3, 1, figsize=(SINGLE, 6.8), sharex=False)
    fig.subplots_adjust(hspace=0.45)

    ax = axes[0]
    ax.loglog(f, np.abs(mag_ut), label=r"$u \to B_{\mathrm{trap}}$")
    ax.loglog(f, np.abs(mag_um), linestyle="--", label=r"$u \to B_{\mathrm{meas}}$")
    ax.axvline(p.coil_pole_hz, color="0.6", lw=0.7, ls=":")
    ax.annotate(f"coil L/R\n{p.coil_pole_hz:.0f} Hz", xy=(p.coil_pole_hz, 1e-6),
                fontsize=6, color="0.4", ha="left")
    ax.set_ylabel("magnitude (T/V)")
    ax.legend(frameon=False, fontsize=6.5)
    ax.grid(True, which="both", alpha=0.25)

    ax = axes[1]
    ax.semilogx(f, np.abs(mag_geom), color="C2",
                label=r"geometry only  $H_{ts}$")
    ax.semilogx(f, np.abs(mag_ts), color="C4", ls="--",
                label=r"incl. instrument  $H_{ts}M$")
    ax.axhline(1.0, color="0.6", lw=0.7, ls=":")
    ax.axvline(p.sensor_transfer_corner_hz, color="0.6", lw=0.7, ls=":")
    ax.set_ylabel(r"$B_{\mathrm{sensor}}/B_{\mathrm{trap}}$")
    ax.set_xlim(1e-1, 1e5)
    ax.legend(frameon=False, fontsize=6.5, loc="lower left")
    ax.annotate("above the corner the sensor and the ions\nno longer see the same field",
                xy=(0.15, 0.42), xycoords="axes fraction", fontsize=6.5, color="0.35")
    ax.grid(True, which="both", alpha=0.25)

    # disturbance spectrum
    fs = 50e3
    n = int(fs * 2.0)
    d = dist.generate(dp, fs, n)
    from scipy.signal import welch
    fw, pxx = welch(d, fs=fs, nperseg=1 << 15)
    ax = axes[2]
    ax.loglog(fw[1:], np.sqrt(pxx[1:]) / 1e-9, color="C3", lw=0.8)
    for h in dp.harmonics:
        ax.axvline(h.freq_hz, color="0.6", lw=0.6, ls=":")
    ax.set_xlim(1, 1e3)
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel(r"disturbance (nT/$\sqrt{\mathrm{Hz}}$)")
    ax.grid(True, which="both", alpha=0.25)

    save_fig(fig, "fig01_openloop")

    dc = np.asarray(ct.dcgain(sysm), dtype=float)
    out = {
        "marker": "[SIMULATED]",
        "plant": {
            "n_states": int(sysm.nstates),
            "coil_pole_hz": p.coil_pole_hz,
            "dc_gain_u_to_trap_T_per_V": float(dc[0, 0]),
            "dc_gain_d_to_trap": float(dc[0, 1]),
            "dc_gain_u_to_meas_T_per_V": float(dc[1, 0]),
            "dc_gain_d_to_meas": float(dc[1, 1]),
            "dc_mismatch_d": float(dc[1, 1] - dc[0, 1]),
        },
        "disturbance": {
            "rms_nT": float(np.std(d) / 1e-9),
            "harmonics_hz": dp.freqs_hz,
            "harmonic_Q_intensity_T2_per_s": dist.harmonic_process_noise_intensity(dp).tolist(),
            "drift_Q_intensity_T2_per_s": dist.drift_process_noise_intensity(dp),
        },
        "note": (
            "dc_mismatch_d is the disturbance gain difference between sensor and "
            "trap at DC. It is a hard floor on DC suppression at the ions: the loop "
            "drives the sensor to zero, and the trap keeps whatever the calibration "
            "and gradient got wrong."
        ),
    }
    (REPO / "results").mkdir(exist_ok=True)
    (REPO / "results" / "checkpoint1.json").write_text(json.dumps(out, indent=2) + "\n")

    print("[SIMULATED] checkpoint 1")
    print(f"  plant states           : {sysm.nstates}")
    print(f"  coil L/R pole          : {p.coil_pole_hz:.2f} Hz")
    print(f"  DC  u->B_trap          : {dc[0,0]:.3e} T/V")
    print(f"  DC  d->B_trap          : {dc[0,1]:.4f}")
    print(f"  DC  d->B_meas          : {dc[1,1]:.4f}")
    print(f"  DC disturbance mismatch: {dc[1,1]-dc[0,1]:.4f}  <- floor on trap suppression")
    print(f"  disturbance RMS        : {np.std(d)/1e-9:.1f} nT")
    print("  wrote figures/fig01_openloop.pdf, results/checkpoint1.json")


if __name__ == "__main__":
    main()

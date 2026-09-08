"""Physical plant model for the ion-trap B-field servo.

[SIMULATED] Every quantity in this module is a model. Nothing here has been
measured on hardware.

The single most important structural decision is that this plant has **two
outputs**:

    B_trap    the field at the ion position. This is the REGULATED variable
              and it is NEVER measured.
    B_sensor  the field at the fluxgate, centimetres away and outside the
              vacuum envelope. This is the ONLY measurement.

They differ for two physical reasons, both modelled explicitly:

1. *Eddy currents.* A stainless vacuum chamber attenuates and phase-shifts the
   field at the ions but not at a sensor mounted outside it. That is the
   `sensor_transfer` block: unity at DC, departing from unity above a corner in
   the low kHz.
2. *Gradient.* A distant disturbance source does not produce the same amplitude
   at two points centimetres apart. `coupling_error` is the fraction of the
   disturbance that appears at the sensor but not at the trap.

On top of both sits a DC calibration error `dc_gain_mismatch`. A 1% error
between sensor and trap puts a hard floor on DC suppression that no amount of
loop gain removes, because the loop drives the *sensor* to zero and the trap
inherits whatever the calibration got wrong.

Modelling this with one output would make the servo look excellent while the
ions see something worse. Suppression is therefore always reported at both
points, and only the trap number is physics.

Transport delays are rational-approximated by Padé for the continuous design;
the discrete design in `discretize.py` replaces them with explicit delay states,
which is the representation the controller is actually built against.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional

import control as ct
import numpy as np


@dataclass(frozen=True)
class PlantParams:
    """Every physical constant, named once.

    Defaults are plausible values for a Helmholtz pair driving a trap volume
    with an external fluxgate. They are *not* measured; see docs/validation.md.
    """

    # --- current driver ---------------------------------------------------
    driver_bw_hz: float = 20e3        # single-pole lowpass corner
    driver_gain_a_per_v: float = 0.1  # A/V

    # --- coil -------------------------------------------------------------
    coil_L_H: float = 10e-3           # inductance
    coil_R_ohm: float = 2.0           # series resistance -> L/R corner ~31.8 Hz
    driver_out_impedance_ohm: float = 0.0   # moves the coil pole realistically
    shield_pole_hz: Optional[float] = None  # optional mutual-inductance-to-shield pole

    # --- coil to field ----------------------------------------------------
    coil_T_per_A: float = 1e-3        # 1 mT/A
    eddy_delay_s: float = 3e-6        # transport delay, chamber eddy currents

    # --- fluxgate ---------------------------------------------------------
    sensor_bw_hz: float = 1e3         # second-order response
    sensor_Q: float = 0.7             # modest Q
    sensor_noise_pT_rtHz: float = 10.0

    # --- anti-alias filter ------------------------------------------------
    aa_bw_hz: float = 10e3            # explicit: its phase eats loop margin
    aa_order: int = 2

    # --- loop latency -----------------------------------------------------
    latency_s: float = 2e-6           # ADC + fabric + DAC settling, lumped

    # --- sensor-to-trap mismatch (the parameters we know least well) ------
    sensor_transfer_corner_hz: float = 2e3  # eddy-current corner, trap -> sensor
    dc_gain_mismatch: float = 1.0           # 1.0 = perfect calibration
    coupling_error: float = 0.02            # disturbance fraction seen only at sensor

    # --- numerics ---------------------------------------------------------
    pade_order: int = 3

    @property
    def coil_pole_hz(self) -> float:
        R = self.coil_R_ohm + self.driver_out_impedance_ohm
        return R / (2.0 * np.pi * self.coil_L_H)

    @property
    def sensor_noise_T_rtHz(self) -> float:
        return self.sensor_noise_pT_rtHz * 1e-12


def _lowpass(f_hz: float) -> ct.TransferFunction:
    w = 2.0 * np.pi * f_hz
    return ct.tf([w], [1.0, w])


def _second_order(f_hz: float, Q: float) -> ct.TransferFunction:
    w = 2.0 * np.pi * f_hz
    return ct.tf([w * w], [1.0, w / Q, w * w])


def _pade_delay(tau_s: float, order: int) -> ct.TransferFunction:
    if tau_s <= 0:
        return ct.tf([1.0], [1.0])
    num, den = ct.pade(tau_s, order)
    return ct.tf(num, den)


def drive_to_trap(p: PlantParams) -> ct.TransferFunction:
    """u (volts at the DAC) -> B_trap (tesla), open loop.

    Cascade: driver lowpass, coil L/R, optional shield pole, coil-to-field
    constant, eddy-current transport delay, and the lumped loop latency.
    """
    g = ct.tf([p.driver_gain_a_per_v], [1.0]) * _lowpass(p.driver_bw_hz)
    g = g * _lowpass(p.coil_pole_hz)
    if p.shield_pole_hz is not None:
        g = g * _lowpass(p.shield_pole_hz)
    g = g * ct.tf([p.coil_T_per_A], [1.0])
    g = g * _pade_delay(p.eddy_delay_s, p.pade_order)
    g = g * _pade_delay(p.latency_s, p.pade_order)
    return g


def sensor_transfer(p: PlantParams) -> ct.TransferFunction:
    """B_trap -> B_sensor.

    Unity at DC (up to the calibration error), rolling off above the
    eddy-current corner. This is the block that makes the sensor a different
    observable from the regulated variable, and its corner frequency is one of
    the least well known parameters in the whole model.
    """
    return ct.tf([p.dc_gain_mismatch], [1.0]) * _lowpass(p.sensor_transfer_corner_hz)


def measurement_chain(p: PlantParams) -> ct.TransferFunction:
    """B_sensor -> ADC counts (in tesla-equivalent units).

    Fluxgate dynamics followed by the anti-alias filter. Kept separate from
    `sensor_transfer` because this part is instrument response, not geometry.
    """
    g = _second_order(p.sensor_bw_hz, p.sensor_Q)
    for _ in range(max(1, p.aa_order)):
        g = g * _lowpass(p.aa_bw_hz)
    return g


def build(p: PlantParams) -> ct.StateSpace:
    """Two-input, two-output continuous plant.

    inputs  : [u, d]      u = DAC drive (V); d = field disturbance at the trap (T)
    outputs : [B_trap, B_meas]

    `B_meas` is what the ADC sees: the trap field passed through the
    sensor-to-trap transfer and the instrument chain, plus the fraction of the
    disturbance that reaches the sensor without reaching the trap.

    The disturbance enters at the *output* (the field), not at the plant input.
    Injecting it at the input is the most common modelling error in this class
    of servo: it would be shaped by the coil and driver dynamics on its way in,
    which is not what ambient mains hum does.
    """
    P_ut = drive_to_trap(p)                    # u -> B_trap
    H_ts = sensor_transfer(p)                  # B_trap -> B_sensor
    M = measurement_chain(p)                   # B_sensor -> B_meas

    # u -> B_trap ; u -> B_meas
    G11 = P_ut
    G21 = P_ut * H_ts * M
    # d -> B_trap (direct, unity) ; d -> B_meas
    G12 = ct.tf([1.0], [1.0])
    # The disturbance reaches the sensor two ways: through the same geometry as
    # the trap field, and via the gradient term that bypasses the trap entirely.
    G22 = (H_ts * M) + ct.tf([p.coupling_error], [1.0]) * M

    # Assemble by stacking rather than `append`, which would block-diagonalise
    # and give each output its own input.
    row1 = ct.tf([[_as_num(G11), _as_num(G12)]], [[_as_den(G11), _as_den(G12)]])
    row2 = ct.tf([[_as_num(G21), _as_num(G22)]], [[_as_den(G21), _as_den(G22)]])
    sys = ct.tf2ss(_vstack_tf(row1, row2))
    return sys


def _as_num(g: ct.TransferFunction):
    return np.asarray(g.num[0][0], dtype=float).tolist()


def _as_den(g: ct.TransferFunction):
    return np.asarray(g.den[0][0], dtype=float).tolist()


def _vstack_tf(top: ct.TransferFunction, bot: ct.TransferFunction) -> ct.TransferFunction:
    """Stack two 1xN transfer-function rows into a 2xN system."""
    num = [top.num[0], bot.num[0]]
    den = [top.den[0], bot.den[0]]
    return ct.tf(num, den)


def nominal() -> PlantParams:
    """The design plant."""
    return PlantParams()


def perturbed(rng: np.random.Generator, base: Optional[PlantParams] = None) -> PlantParams:
    """Draw a plant from the tolerance family used in the Monte Carlo.

    The sensor-to-trap parameters get the widest spread deliberately: they are
    geometry and calibration, which are the quantities least well known in a
    real installation, and section 6 of the brief expects their contribution to
    be broken out separately.
    """
    b = base or nominal()
    return replace(
        b,
        coil_L_H=b.coil_L_H * float(rng.uniform(0.9, 1.1)),           # +/-10%
        sensor_bw_hz=b.sensor_bw_hz * float(rng.uniform(0.8, 1.2)),   # +/-20%
        latency_s=float(rng.uniform(1e-6, 5e-6)),                     # 1-5 us
        dc_gain_mismatch=float(rng.uniform(0.98, 1.02)),              # +/-2%
        sensor_transfer_corner_hz=b.sensor_transfer_corner_hz * float(rng.uniform(0.5, 1.5)),
        coupling_error=float(rng.uniform(0.0, 0.04)),
    )

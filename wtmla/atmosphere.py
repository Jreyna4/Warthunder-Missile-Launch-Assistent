"""International Standard Atmosphere (ISA), vectorized with numpy."""

from __future__ import annotations

import numpy as np

G0 = 9.80665          # m/s^2
R_AIR = 287.05287     # J/(kg K)
GAMMA = 1.4
RHO0 = 1.225          # kg/m^3 at sea level


def isa(alt_m):
    """Return (density kg/m^3, speed of sound m/s) for altitude(s) in metres."""
    h = np.clip(np.asarray(alt_m, dtype=float), -500.0, 32000.0)
    t = np.where(h < 11000.0, 288.15 - 0.0065 * h, 216.65)
    p = np.where(
        h < 11000.0,
        101325.0 * (t / 288.15) ** 5.25588,
        22632.06 * np.exp(-G0 * (h - 11000.0) / (R_AIR * 216.65)),
    )
    rho = p / (R_AIR * t)
    a = np.sqrt(GAMMA * R_AIR * t)
    return rho, a


def mach_to_tas(mach, alt_m):
    return np.asarray(mach, dtype=float) * isa(alt_m)[1]


def tas_to_mach(tas, alt_m):
    return np.asarray(tas, dtype=float) / isa(alt_m)[1]


def ias_to_tas(ias, alt_m):
    """Indicated -> true airspeed (EAS approximation, good enough for the game HUD)."""
    rho, _ = isa(alt_m)
    return np.asarray(ias, dtype=float) * np.sqrt(RHO0 / rho)


def tas_to_ias(tas, alt_m):
    rho, _ = isa(alt_m)
    return np.asarray(tas, dtype=float) * np.sqrt(rho / RHO0)

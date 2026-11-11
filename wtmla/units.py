"""Convert what you read on the in-game HUD into true airspeed in m/s."""

from __future__ import annotations

from .atmosphere import ias_to_tas, mach_to_tas

SPEED_UNITS = ("km/h TAS", "km/h IAS", "Mach", "kn TAS", "kn IAS", "mph TAS", "m/s TAS")


def to_tas_ms(value: float, unit: str, alt_m: float) -> float:
    u = unit.strip()
    if u == "Mach":
        return float(mach_to_tas(value, alt_m))
    factor = {"km/h": 1 / 3.6, "kn": 0.514444, "mph": 0.44704, "m/s": 1.0}[u.split()[0]]
    v = value * factor
    if u.endswith("IAS"):
        v = float(ias_to_tas(v, alt_m))
    return float(v)


ALT_UNITS = ("m", "ft")


def to_metres(value: float, unit: str) -> float:
    return value * 0.3048 if unit == "ft" else value

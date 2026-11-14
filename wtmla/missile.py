"""Missile data model.

A MissileSpec holds the in-game values for one missile, taken from the
"Guided weaponry data (in-game values)" spreadsheet. Every field is optional
because different tabs of the sheet (IR, SARH, ARH ...) have different rows.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Optional

# Guidance categories used by the advisor.
IR = "IR"
SARH = "SARH"
ARH = "ARH"
MANUAL = "MANUAL"
UNKNOWN = "UNKNOWN"


@dataclass
class MissileSpec:
    name: str
    category: str = UNKNOWN
    source_tab: Optional[str] = None

    # Physical properties
    mass: Optional[float] = None                 # kg
    fuel1: Optional[float] = None                # kg
    fuel2: Optional[float] = None                # kg
    calibre_mm: Optional[float] = None           # mm
    length: Optional[float] = None               # m

    # Engine properties
    thrust1: Optional[float] = None              # N
    burn1: Optional[float] = None                # s
    isp1: Optional[float] = None                 # s
    dv1: Optional[float] = None                  # m/s
    start_delay1: Optional[float] = None         # s
    thrust2: Optional[float] = None
    burn2: Optional[float] = None
    isp2: Optional[float] = None
    dv2: Optional[float] = None
    total_dv: Optional[float] = None
    s2_time: Optional[float] = None              # 2nd stage: time from launch above which it starts
    s2_eta: Optional[float] = None               # 2nd stage: ETA below which it starts
    s2_speed: Optional[float] = None             # 2nd stage: speed below which it starts
    s2_closure: Optional[float] = None           # 2nd stage: closure rate below which it starts

    # Fuze and warhead
    explosive_mass: Optional[float] = None       # kg TNT eq.
    warhead: Optional[str] = None
    penetration: Optional[float] = None
    proximity_fuze: Optional[bool] = None
    prox_range: Optional[float] = None           # m
    prox_delay: Optional[float] = None           # s (treated as arming time after launch)

    # Guidance
    guidance_type: Optional[str] = None
    band: Optional[str] = None
    signal: Optional[str] = None
    guidance_start_delay: Optional[float] = None  # s
    guidance_duration: Optional[float] = None     # s
    gimbal_limit: Optional[float] = None          # deg
    track_rate: Optional[float] = None            # deg/s
    uncaged_before_launch: Optional[bool] = None
    lock_range_all_aspect: Optional[float] = None   # km
    lock_range_rear_aspect: Optional[float] = None  # km
    seeker_on_range: Optional[float] = None         # km
    max_lock_range: Optional[float] = None          # km
    can_lock_after_launch: Optional[bool] = None
    inertial_drift: Optional[float] = None          # m/s
    datalink: Optional[bool] = None
    pn_mult: Optional[float] = None
    pn_schedule: list = field(default_factory=list)     # [(eta_s, percent)]

    # Flight characteristics
    drag_mult: Optional[float] = None
    ballistic_coeff: Optional[float] = None
    max_fin_aoa: Optional[float] = None
    wing_area_mult: Optional[float] = None
    start_speed: Optional[float] = None
    max_speed: Optional[float] = None               # m/s
    min_range: Optional[float] = None               # m
    flight_range_limit: Optional[float] = None      # km
    max_g: Optional[float] = None                   # G
    pull_schedule: list = field(default_factory=list)   # [(time_s, percent)]
    thrust_vectoring: Optional[bool] = None
    loft: Optional[bool] = None
    loft_angle: Optional[float] = None              # deg
    loft_target_elevation: Optional[float] = None   # deg
    loft_max_target_angular_change: Optional[float] = None  # deg/s
    notes: Optional[str] = None

    # Rows we don't model yet are kept here so nothing from the sheet is lost.
    extra: dict = field(default_factory=dict)

    # ------------------------------------------------------------------ helpers
    @property
    def reference_area(self) -> float:
        """Body cross-section in m^2 (from calibre)."""
        d = (self.calibre_mm or 200.0) / 1000.0
        return math.pi * (d / 2.0) ** 2

    @property
    def burnout_mass(self) -> float:
        return (self.mass or 0.0) - (self.fuel1 or 0.0) - (self.fuel2 or 0.0)

    @property
    def has_second_stage(self) -> bool:
        return bool(self.thrust2 and self.burn2)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "MissileSpec":
        known = {f.name for f in fields(cls)}
        kwargs = {k: v for k, v in data.items() if k in known}
        extra = dict(kwargs.pop("extra", {}) or {})
        for k, v in data.items():
            if k not in known:
                extra[k] = v
        spec = cls(**kwargs, extra=extra)
        spec.pn_schedule = [tuple(x) for x in spec.pn_schedule]
        spec.pull_schedule = [tuple(x) for x in spec.pull_schedule]
        return spec

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "MissileSpec":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def summary(self) -> str:
        def f(v, unit="", fmt="{:g}"):
            return "-" if v is None else fmt.format(v) + unit

        lines = [
            f"{self.name}  [{self.category}]  guidance: {self.guidance_type or '-'}",
            f"Mass {f(self.mass, ' kg')}, calibre {f(self.calibre_mm, ' mm')}, "
            f"max speed {f(self.max_speed, ' m/s')}, max G {f(self.max_g)}",
            f"Motor 1: {f(self.thrust1, ' N')} for {f(self.burn1, ' s')}"
            + (f"; motor 2: {f(self.thrust2, ' N')} for {f(self.burn2, ' s')}" if self.has_second_stage else ""),
            f"Lock range all-aspect {f(self.lock_range_all_aspect, ' km')}, "
            f"rear-aspect {f(self.lock_range_rear_aspect, ' km')}",
            f"Guidance time {f(self.guidance_duration, ' s')}, range limit {f(self.flight_range_limit, ' km')}, "
            f"min range {f(self.min_range, ' m')}",
            f"Prox fuze: {'yes, ' + f(self.prox_range, ' m') if self.proximity_fuze else 'no'}; "
            f"loft: {'yes, ' + f(self.loft_angle, ' deg') if self.loft else 'no'}",
        ]
        return "\n".join(lines)


def classify_guidance(guidance_type: Optional[str], tab_name: Optional[str] = None) -> str:
    """Work out IR / SARH / ARH from the guidance text or the sheet tab name."""
    for text in (guidance_type, tab_name):
        if not text:
            continue
        t = text.upper()
        # Check SARH before ARH because "SARH" contains "ARH".
        if re.search(r"\bSARH\b", t):
            return SARH
        if re.search(r"\bARH\b", t):
            return ARH
        if re.search(r"\bIR\b", t) or "INFRARED" in t:
            return IR
        if "MANUAL" in t or "MCLOS" in t or "SACLOS" in t or "BEAM" in t:
            return MANUAL
    return UNKNOWN


def name_key(name: str) -> str:
    """'AIM-7F Sparrow' -> 'AIM7FSPARROW' (case and punctuation ignored)."""
    return "".join(ch for ch in name.upper() if ch.isalnum())


def _prefix_keys(name: str) -> set:
    words = name.split()
    return {name_key(" ".join(words[:i])) for i in range(1, len(words) + 1)}


def names_match(a: str, b: str) -> bool:
    """Same missile? 'AIM-7F' = 'AIM-7F Sparrow', but 'R-27R' != 'R-27R1' and 'Super 530D' != 'Super 530F'."""
    ka, kb = name_key(a), name_key(b)
    if not ka or not kb:
        return False
    return ka == kb or ka in _prefix_keys(b) or kb in _prefix_keys(a)


def slug(name: str) -> str:
    """File-system-safe version of a missile or aircraft name."""
    s = re.sub(r"[^A-Za-z0-9._+-]+", "_", name.strip())
    return s.strip("_") or "unnamed"

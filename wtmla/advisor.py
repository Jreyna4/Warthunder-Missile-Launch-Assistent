"""Turns raw ranges into launch advice.

    advise(...) -> Advice  with
        r_max_km        longest range that still hits a target that keeps flying straight
        r_ne_km         no-escape range: hits even if the target breaks away at launch
        recommended_km  launch at or inside this range (no-escape, limited by your sensors)
        aggressive_km   halfway between R_ne and R_max; fine if the target is unaware
        sensor_limit_km lock/track limit from the missile seeker or your radar
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from .aircraft import AircraftSpec
from .envelope import envelope
from .missile import ARH, IR, MANUAL, SARH, MissileSpec
from .nn import Regressor
from .simulator import PhysicsConfig, simulate
from .train import SAMPLE_BOUNDS, fingerprint, make_features, model_path

REAR_ASPECT_DEG = 135.0   # aspect at or above this counts as "rear" for IR seekers


@dataclass
class Advice:
    missile: str
    aircraft: Optional[str]
    r_max_km: float
    r_ne_km: float
    recommended_km: Optional[float]
    aggressive_km: Optional[float]
    min_range_km: float
    sensor_limit_km: Optional[float]
    source: str
    tof_s: Optional[float] = None
    active_after_s: Optional[float] = None
    notes: list[str] = field(default_factory=list)

    def text(self) -> str:
        def km(v):
            return "-" if v is None else f"{v:.1f} km"

        lines = [f"{self.missile}" + (f" from {self.aircraft}" if self.aircraft else ""),
                 f"  RECOMMENDED launch range : {km(self.recommended_km)}",
                 f"  Aggressive (unaware tgt) : {km(self.aggressive_km)}",
                 f"  No-escape range  (R_ne)  : {km(self.r_ne_km)}",
                 f"  Max range        (R_max) : {km(self.r_max_km)}",
                 f"  Minimum range            : {self.min_range_km * 1000:.0f} m",
                 f"  Sensor / lock limit      : {km(self.sensor_limit_km)}"]
        if self.tof_s is not None:
            lines.append(f"  Time of flight at recommended range: {self.tof_s:.0f} s")
        if self.active_after_s is not None:
            lines.append(f"  Seeker goes active about {self.active_after_s:.0f} s after launch")
        lines.append(f"  (computed by: {self.source})")
        lines += [f"  * {n}" for n in self.notes]
        return "\n".join(lines)


class ModelCache:
    def __init__(self, models_dir: Path):
        self.models_dir = Path(models_dir)
        self._cache: dict[str, tuple[float, Regressor]] = {}

    def get(self, missile_name: str) -> Optional[Regressor]:
        p = model_path(self.models_dir, missile_name)
        if not p.exists():
            return None
        mtime = p.stat().st_mtime
        hit = self._cache.get(missile_name)
        if hit and hit[0] == mtime:
            return hit[1]
        reg = Regressor.load(p)
        self._cache[missile_name] = (mtime, reg)
        return reg


def _in_bounds(ss, sa, ts, ta) -> bool:
    b = SAMPLE_BOUNDS
    return (b["shooter_speed_ms"][0] <= ss <= b["shooter_speed_ms"][1]
            and b["shooter_alt_m"][0] <= sa <= b["shooter_alt_m"][1]
            and b["target_speed_ms"][0] <= ts <= b["target_speed_ms"][1]
            and b["target_alt_m"][0] <= ta <= b["target_alt_m"][1])


def sensor_limit(missile: MissileSpec, aircraft: Optional[AircraftSpec], aspect_deg: float):
    """Return (limit_km or None, notes, can_fire)."""
    notes: list[str] = []
    can_fire = True
    limit = None
    radar = aircraft.radar if aircraft else None
    radar_km = radar.track_range_km if radar else None

    if missile.category == IR:
        rear = aspect_deg >= REAR_ASPECT_DEG
        all_asp = missile.lock_range_all_aspect
        rear_km = missile.lock_range_rear_aspect or all_asp
        if rear:
            limit = rear_km
        elif all_asp:
            limit = all_asp
        else:
            can_fire = False
            notes.append("Rear-aspect IR seeker: it can only lock from behind the target "
                         f"(aspect ≥ {REAR_ASPECT_DEG:.0f}°).")
        notes.append("IR missile: the seeker must lock before launch. Lock range drops against "
                     "targets with engines at idle or dropping flares.")
    elif missile.category == SARH:
        limit = radar_km
        if missile.max_lock_range:
            limit = min(limit, missile.max_lock_range) if limit else missile.max_lock_range
        notes.append("SARH missile: keep your radar locked on the target until impact.")
        if aircraft and not radar_km:
            notes.append("Your aircraft card has no radar track range; add it to limit the advice.")
        if aircraft and not radar.can_guide_sarh:
            notes.append("Your aircraft card says the radar cannot guide SARH missiles.")
    elif missile.category == ARH:
        gt = (missile.guidance_type or "").upper()
        mid_course = bool(missile.datalink) or "IOG" in gt or "INS" in gt
        if mid_course:
            limit = radar_km
            notes.append("ARH with inertial/datalink guidance: fire from a radar lock; it flies "
                         "to the predicted point and turns its own seeker on near the target.")
            if aircraft and not radar_km:
                notes.append("Your aircraft card has no radar track range; add it to limit the advice.")
        else:
            limit = missile.lock_range_all_aspect
            notes.append("ARH without mid-course guidance: the missile's own seeker must lock before launch.")
        if aircraft and not radar.can_guide_arh and mid_course:
            notes.append("Your aircraft card says the radar cannot launch ARH missiles.")
    elif missile.category == MANUAL:
        can_fire = False
        notes.append("Manually guided missile: launch advice is not supported.")
    return limit, notes, can_fire


def advise(missile: MissileSpec, shooter_speed_ms: float, shooter_alt_m: float,
           target_speed_ms: Optional[float] = None, target_alt_m: Optional[float] = None,
           aspect_deg: float = 0.0, aircraft: Optional[AircraftSpec] = None,
           cfg: Optional[PhysicsConfig] = None, models: Optional[ModelCache] = None,
           force_simulator: bool = False, escape_g: Optional[float] = None) -> Advice:
    cfg = cfg or PhysicsConfig()
    escape_g = cfg.escape_turn_g if escape_g is None else escape_g
    ts = shooter_speed_ms if target_speed_ms is None else target_speed_ms
    ta = shooter_alt_m if target_alt_m is None else target_alt_m
    aspect_deg = float(np.clip(abs(aspect_deg), 0.0, 180.0))
    notes: list[str] = []

    source = None
    reg = None if (force_simulator or models is None) else models.get(missile.name)
    if reg is not None:
        if reg.meta.get("fingerprint") != fingerprint(missile, cfg):
            notes.append("Trained model is out of date for this missile data - using the simulator. "
                         "Retrain to get instant answers again.")
            reg = None
        elif abs(reg.meta.get("escape_g", escape_g) - escape_g) > 1e-6:
            reg = None
        elif not _in_bounds(shooter_speed_ms, shooter_alt_m, ts, ta):
            notes.append("Inputs are outside the trained range - using the simulator.")
            reg = None
    if reg is not None:
        pred = reg.predict(make_features(shooter_speed_ms, shooter_alt_m, ts, ta, aspect_deg))[0]
        r_max, r_ne = max(pred[0], 0.0), max(pred[1], 0.0)
        r_ne = min(r_ne, r_max)
        source = "neural network (trained on the simulator)"
    else:
        env = envelope(missile, shooter_speed_ms, shooter_alt_m, ts, ta, aspect_deg, escape_g=escape_g, cfg=cfg)
        r_max, r_ne = env["r_max"][0] / 1000.0, env["r_ne"][0] / 1000.0
        if env["r_max_at_edge"][0]:
            notes.append("Max range reached the end of the search window; the real max may be longer.")
        source = "physics simulator"

    limit, s_notes, can_fire = sensor_limit(missile, aircraft, aspect_deg)
    notes += s_notes
    if aircraft and not aircraft.carries(missile.name) and aircraft.weapons:
        notes.append(f"{missile.name} is not in the weapon list of {aircraft.name}'s card.")

    def cap(v):
        if v is None or v <= 0:
            return None
        return min(v, limit) if limit else v

    recommended = cap(r_ne) if can_fire else None
    aggressive = cap(r_ne + 0.5 * (r_max - r_ne)) if (can_fire and r_max > 0) else None
    if can_fire and r_ne <= 0 and r_max > 0:
        notes.append("No no-escape zone: a target that turns away at launch can outrun this missile. "
                     "Only fire if it is committed towards you, or get closer / faster / higher.")
    if can_fire and r_max <= 0:
        notes.append("The missile cannot reach the target in these conditions.")
    if limit and r_ne > limit:
        notes.append(f"Your sensors ({limit:.1f} km) limit the shot, not the missile's energy.")

    advice = Advice(missile=missile.name, aircraft=aircraft.name if aircraft else None,
                    r_max_km=r_max, r_ne_km=r_ne, recommended_km=recommended, aggressive_km=aggressive,
                    min_range_km=(missile.min_range or 0.0) / 1000.0, sensor_limit_km=limit,
                    source=source, notes=notes)

    # Time of flight (and ARH "pitbull" time) at the recommended range.
    shot = recommended or aggressive
    if shot:
        trk = None
        if missile.category == ARH and missile.lock_range_all_aspect:
            trk = missile.lock_range_all_aspect * 1000.0
        res = simulate(missile, shooter_speed_ms, shooter_alt_m, ts, ta, aspect_deg, shot * 1000.0,
                       track_range_m=trk, cfg=cfg)
        if res["hit"][0]:
            advice.tof_s = float(res["tof"][0])
            if trk is not None and np.isfinite(res["track_time"][0]):
                advice.active_after_s = float(res["track_time"][0])
    return advice

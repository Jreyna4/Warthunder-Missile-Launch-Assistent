"""Batched 3D point-mass missile flight simulator.

Many engagements are flown *at the same time* with numpy arrays, which makes it
fast enough to search for launch ranges and to generate training data for the
neural network.

Coordinates: x = downrange (shooter -> target at launch), y = crossrange,
z = altitude. The shooter is at x = 0 and points its nose at the target.

The model is an estimate built from the spreadsheet values:
  * one or two rocket motors (thrust, burn time, fuel mass, start delay, 2nd-stage triggers)
  * Mach-dependent drag scaled by the sheet's drag multiplier + induced drag when pulling G
  * lateral acceleration limited by max G, by lift (dynamic pressure x wing area multiplier)
    and by the "pull limit" schedule at launch
  * proportional navigation using the sheet's PN multiplier and its ETA schedule
  * simple loft (loft angle / target elevation / max target angular change)
  * proximity fuze radius + arming time, guidance duration and flight range limit
It does NOT model seeker countermeasures, notching, chaff/flares, terrain or radar
break-locks. Tune PhysicsConfig against your own in-game tests.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from .atmosphere import G0, isa
from .missile import MissileSpec

REASONS = {0: "flying", 1: "hit", 2: "hit the ground", 3: "guidance timed out",
           4: "flight range limit", 5: "could not close on target"}


@dataclass
class PhysicsConfig:
    """Global model constants. These are not in the sheet; tune them if results look off."""
    dt: float = 0.05                    # integration step, s
    cl_max: float = 20.0                # max lift coefficient (body-area based) before wing multiplier
    induced_drag_k: float = 0.004       # induced drag factor:  Cdi = k * CL^2
    impact_radius_m: float = 4.0        # counts as a hit without a proximity fuze
    loft_gain: float = 2.0              # 1/s, how hard the missile steers to its loft climb angle
    loft_max_climb_deg: float = 45.0
    loft_min_eta_s: float = 8.0         # no loft when time-to-impact is shorter than this
    escape_turn_g: float = 5.0          # default target break turn for "no-escape" range
    mach_table: tuple = (0.0, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0, 4.0, 5.0)
    cd_table: tuple = (0.30, 0.32, 0.55, 0.60, 0.52, 0.43, 0.33, 0.28, 0.25)

    @classmethod
    def load(cls, path: Optional[Path]) -> "PhysicsConfig":
        if path is None or not Path(path).exists():
            return cls()
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        for k in ("mach_table", "cd_table"):
            if k in known:
                known[k] = tuple(known[k])
        return cls(**known)

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")


def _dot(a, b):
    return np.einsum("ij,ij->i", a, b)


def _norm(a):
    return np.sqrt(_dot(a, a))


def _schedule(points, scale=100.0):
    if not points:
        return None
    pts = sorted((float(x), float(y) / scale) for x, y in points)
    return np.array([p[0] for p in pts]), np.array([p[1] for p in pts])


def simulate(missile: MissileSpec,
             shooter_speed, shooter_alt, target_speed, target_alt, aspect_deg, range_m,
             escape=False, escape_g=None, track_range_m=None,
             cfg: Optional[PhysicsConfig] = None) -> dict:
    """Fly N engagements. All scenario arguments broadcast to a common length N.

    shooter_speed / target_speed: true airspeed in m/s
    aspect_deg: 0 = target flying straight at the shooter (head-on), 90 = beam, 180 = tail chase
    range_m: slant range at launch
    escape: if True the target immediately turns away from the missile at escape_g and runs
    track_range_m: optional distance; the time when missile-target range first drops below it
                   is returned as 'track_time' (used for "seeker goes active" time)
    """
    cfg = cfg or PhysicsConfig()
    escape_g = cfg.escape_turn_g if escape_g is None else escape_g
    track = track_range_m is not None
    arrs = np.broadcast_arrays(
        *(np.atleast_1d(np.asarray(v, dtype=float)) for v in
          (shooter_speed, shooter_alt, target_speed, target_alt, aspect_deg, range_m,
           escape, escape_g, track_range_m if track else 0.0)))
    ss, sa, ts, ta, asp, rng, esc, eg, trk = (a.astype(float).copy() for a in arrs)
    esc = esc.astype(bool)
    n_total = ss.shape[0]

    # ---------------------------------------------------------------- missile constants
    m0 = float(missile.mass or 100.0)
    t1s = float(missile.start_delay1 or 0.0)
    burn1 = float(missile.burn1 or 0.0)
    t1e = t1s + burn1
    thrust1 = float(missile.thrust1 or 0.0)
    mdot1 = float(missile.fuel1 or 0.0) / burn1 if burn1 > 0 else 0.0
    has2 = missile.has_second_stage
    thrust2 = float(missile.thrust2 or 0.0)
    burn2 = float(missile.burn2 or 0.0)
    mdot2 = float(missile.fuel2 or 0.0) / burn2 if burn2 > 0 else 0.0
    s2_conds = [missile.s2_time, missile.s2_eta, missile.s2_speed, missile.s2_closure]
    s2_any_trigger = any(c is not None for c in s2_conds)
    area = missile.reference_area
    wing = float(missile.wing_area_mult or 1.0)
    cdk = float(missile.drag_mult or 1.0)
    max_acc = float(missile.max_g or 20.0) * G0
    vmax = float(missile.max_speed or 1e9)
    pn = float(missile.pn_mult or 4.0)
    pn_sched = _schedule(missile.pn_schedule)
    pull_sched = _schedule(missile.pull_schedule)
    g_start = float(missile.guidance_start_delay or 0.0)
    t_max = float(missile.guidance_duration or 60.0)
    range_limit = float(missile.flight_range_limit) * 1000.0 if missile.flight_range_limit else np.inf
    has_prox = bool(missile.proximity_fuze and missile.prox_range)
    prox_r = max(float(missile.prox_range or 0.0), cfg.impact_radius_m)
    prox_arm = float(missile.prox_delay or 0.0)
    mach_tab = np.array(cfg.mach_table)
    cd_tab = np.array(cfg.cd_table)
    dt = cfg.dt

    # ---------------------------------------------------------------- initial geometry
    dz = ta - sa
    slant = np.maximum(rng, np.abs(dz) + 1.0)
    horiz = np.sqrt(np.maximum(slant ** 2 - dz ** 2, 1.0))
    pm = np.stack([np.zeros(n_total), np.zeros(n_total), sa], axis=1)
    pt = np.stack([horiz, np.zeros(n_total), ta], axis=1)
    los = (pt - pm) / _norm(pt - pm)[:, None]
    vm = np.maximum(ss, 1.0)[:, None] * los
    a = np.radians(asp)
    vt = np.stack([-np.cos(a) * ts, np.sin(a) * ts, np.zeros(n_total)], axis=1)
    turn_rate = eg * G0 / np.maximum(ts, 1.0)

    # Loft decision at launch (only when the target is barely moving across the sky).
    R0 = pt - pm
    r0 = _norm(R0)
    los_rate0 = np.degrees(_norm(np.cross(R0, vt - vm)) / r0 ** 2)
    vc0 = -_dot(R0, vt - vm) / r0
    loft_on = np.full(n_total, bool(missile.loft) and bool(missile.loft_angle))
    if missile.loft_max_target_angular_change is not None:
        loft_on &= los_rate0 <= float(missile.loft_max_target_angular_change)
    loft_on &= (r0 / np.maximum(vc0 + 600.0, 1.0)) > cfg.loft_min_eta_s
    loft_ang = math.radians(float(missile.loft_angle or 0.0))
    loft_te = float(missile.loft_target_elevation if missile.loft_target_elevation is not None else -2.0)
    loft_max = math.radians(cfg.loft_max_climb_deg)

    # ---------------------------------------------------------------- outputs
    out_hit = np.zeros(n_total, dtype=bool)
    out_reason = np.zeros(n_total, dtype=int)
    out_tof = np.full(n_total, np.nan)
    out_miss = np.full(n_total, np.inf)
    out_vimp = np.full(n_total, np.nan)
    out_track = np.full(n_total, np.nan)

    # Active state (compacted as engagements finish).
    idx = np.arange(n_total)
    s2_start = np.full(n_total, np.nan)
    traveled = np.zeros(n_total)
    miss = np.full(n_total, np.inf)
    track_t = np.full(n_total, np.nan)
    state = dict(pm=pm, vm=vm, pt=pt, vt=vt, ts=ts, esc=esc, turn_rate=turn_rate, loft_on=loft_on,
                 s2_start=s2_start, traveled=traveled, miss=miss, track_t=track_t, trk=trk)

    t = 0.0
    while idx.size:
        pm, vm, pt, vt = state["pm"], state["vm"], state["pt"], state["vt"]
        R = pt - pm
        r = np.maximum(_norm(R), 1e-3)
        Vr = vt - vm
        vc = -_dot(R, Vr) / r
        speed = np.maximum(_norm(vm), 1.0)
        vhat = vm / speed[:, None]
        rho, a_snd = isa(pm[:, 2])
        mach = speed / a_snd
        q = 0.5 * rho * speed ** 2
        eta = r / np.maximum(vc, 1.0)

        # -- motors
        thr = np.full(idx.size, thrust1 if t1s <= t < t1e else 0.0)
        burned = mdot1 * min(max(t - t1s, 0.0), burn1)
        if has2:
            s2s = state["s2_start"]
            waiting = np.isnan(s2s) & (t >= t1e)
            if s2_any_trigger:
                cond = np.zeros(idx.size, dtype=bool)
                if missile.s2_time is not None:
                    cond |= t >= missile.s2_time
                if missile.s2_eta is not None:
                    cond |= eta <= missile.s2_eta
                if missile.s2_speed is not None:
                    cond |= speed <= missile.s2_speed
                if missile.s2_closure is not None:
                    cond |= vc <= missile.s2_closure
                waiting &= cond
            s2s[waiting] = t
            since2 = np.where(np.isnan(s2s), -1.0, t - s2s)
            burning2 = (since2 >= 0) & (since2 < burn2)
            thr = thr + thrust2 * burning2
            burned = burned + mdot2 * np.clip(since2, 0.0, burn2)
        mass = m0 - burned

        # -- guidance: proportional navigation with the sheet's ETA schedule
        pull = 1.0 if pull_sched is None else float(np.interp(t, *pull_sched))
        if t < g_start:
            pull = 0.0
        n_eff = pn * (np.interp(eta, *pn_sched) if pn_sched is not None else 1.0)
        w = np.cross(R, Vr) / (r ** 2)[:, None]
        a_cmd = (n_eff * vc)[:, None] * np.cross(w, R / r[:, None])

        lo = state["loft_on"]
        if lo.any():
            el = np.arcsin(np.clip(R[:, 2] / r, -1, 1))
            lo &= (np.degrees(el) > loft_te) & (eta > cfg.loft_min_eta_s)
            gam = np.arcsin(np.clip(vm[:, 2] / speed, -1, 1))
            gd = np.minimum(el + loft_ang, loft_max)
            a_cmd[:, 2] = np.where(lo, cfg.loft_gain * (gd - gam) * speed, a_cmd[:, 2])

        a_cmd[:, 2] += G0                                   # gravity compensation
        a_cmd -= _dot(a_cmd, vhat)[:, None] * vhat           # fins only push sideways
        lift_lim = cfg.cl_max * q * area * wing / mass
        lim = np.minimum(max_acc, lift_lim) * pull
        mag = _norm(a_cmd)
        a_lat = a_cmd * np.minimum(1.0, lim / np.maximum(mag, 1e-9))[:, None]

        # -- drag (parasitic + induced)
        cd = np.interp(mach, mach_tab, cd_tab) * cdk
        qa = np.maximum(q * area * wing, 1e-6)
        cl = mass * _norm(a_lat) / qa
        drag = q * area * (cd + cfg.induced_drag_k * cl ** 2)

        acc = ((thr - drag) / mass)[:, None] * vhat + a_lat
        acc[:, 2] -= G0
        vm_new = vm + acc * dt
        sp_new = _norm(vm_new)
        over = sp_new > vmax
        if over.any():
            vm_new[over] *= (vmax / sp_new[over])[:, None]
        pm_new = pm + vm_new * dt

        # -- target (straight line, or break turn away from the missile)
        vt_new = vt
        e = state["esc"]
        if e.any():
            psi = np.arctan2(vt[:, 1], vt[:, 0])
            away = np.arctan2(pt[:, 1] - pm[:, 1], pt[:, 0] - pm[:, 0])
            d = (away - psi + np.pi) % (2 * np.pi) - np.pi
            step = np.clip(d, -state["turn_rate"] * dt, state["turn_rate"] * dt)
            psi = psi + np.where(e, step, 0.0)
            tsp = state["ts"]
            vt_new = np.stack([np.cos(psi) * tsp, np.sin(psi) * tsp, np.zeros_like(tsp)], axis=1)
            vt_new = np.where(e[:, None], vt_new, vt)
        pt_new = pt + vt_new * dt

        # -- closest approach during this step
        R1 = pt_new - pm_new
        D = R1 - R
        s = np.clip(-_dot(R, D) / np.maximum(_dot(D, D), 1e-9), 0.0, 1.0)
        dmin = _norm(R + s[:, None] * D)
        state["miss"] = np.minimum(state["miss"], dmin)
        state["traveled"] = state["traveled"] + speed * dt
        t_new = t + dt
        r1 = _norm(R1)
        if track:
            tt = state["track_t"]
            first = np.isnan(tt) & (np.minimum(r1, dmin) <= state["trk"])
            tt[first] = t_new

        radius = prox_r if (has_prox and t_new >= prox_arm) else cfg.impact_radius_m
        hit = dmin <= radius
        vc1 = -_dot(R1, vt_new - vm_new) / np.maximum(r1, 1e-3)
        motor_off = (thr == 0) & (t_new > t1e)
        reason = np.zeros(idx.size, dtype=int)
        reason[(vc1 < 0) & motor_off & (t_new > g_start + 1.0)] = 5
        reason[state["traveled"] > range_limit] = 4
        if t_new >= t_max:
            reason[:] = 3
        reason[pm_new[:, 2] <= 0.0] = 2
        reason[hit] = 1
        done = reason > 0

        state.update(pm=pm_new, vm=vm_new, pt=pt_new, vt=vt_new)
        if done.any():
            gi = idx[done]
            out_hit[gi] = hit[done]
            out_reason[gi] = reason[done]
            out_tof[gi] = t_new
            out_miss[gi] = state["miss"][done]
            out_vimp[gi] = sp_new[done]
            out_track[gi] = state["track_t"][done]
            keep = ~done
            idx = idx[keep]
            for k in state:
                state[k] = state[k][keep]
        t = t_new

    return {"hit": out_hit, "reason": out_reason, "tof": out_tof, "miss": out_miss,
            "impact_speed": out_vimp, "track_time": out_track}

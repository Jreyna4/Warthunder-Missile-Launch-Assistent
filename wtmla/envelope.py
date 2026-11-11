"""Launch-envelope search: how far away can the missile still hit?

Two ranges are computed for every scenario:
  * R_max - longest launch range that hits a target that keeps flying straight
  * R_ne  - "no-escape" range: longest launch range that still hits when the target
            breaks away from the missile at launch (turns cold at escape_g and runs)

The search is a coarse logarithmic grid followed by two zoom-in passes. Every pass
simulates all scenarios and all candidate ranges in one batch.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .missile import MissileSpec
from .simulator import PhysicsConfig, simulate


def _search_bounds(missile: MissileSpec):
    lo = max(float(missile.min_range or 0.0), 500.0)
    limit_km = float(missile.flight_range_limit or 60.0)
    hi = min(max(limit_km * 1.6, 20.0), 200.0) * 1000.0
    return lo, hi


def max_launch_range(missile: MissileSpec, shooter_speed, shooter_alt, target_speed, target_alt,
                     aspect_deg, escape, escape_g=None, cfg: Optional[PhysicsConfig] = None,
                     n_grid: int = 20, n_refine: int = 10, levels: int = 2, chunk: int = 60000):
    """Return (range_m, at_search_edge) arrays for each scenario (0 m = never hits)."""
    cfg = cfg or PhysicsConfig()
    escape_g = cfg.escape_turn_g if escape_g is None else escape_g
    scen = [np.atleast_1d(np.asarray(v, dtype=float)) for v in
            (shooter_speed, shooter_alt, target_speed, target_alt, aspect_deg, escape, escape_g)]
    scen = [a.astype(float) for a in np.broadcast_arrays(*scen)]
    m = scen[0].shape[0]
    r_lo, r_hi = _search_bounds(missile)

    def run(ranges: np.ndarray, rows: np.ndarray) -> np.ndarray:
        """Simulate scenarios `rows`, each at the candidate ranges in the matching row of `ranges`."""
        k = ranges.shape[1]
        flat = [np.repeat(a[rows], k) for a in scen]
        rr = ranges.reshape(-1)
        hits = np.zeros(rr.shape[0], dtype=bool)
        for s in range(0, rr.shape[0], chunk):
            e = s + chunk
            res = simulate(missile, flat[0][s:e], flat[1][s:e], flat[2][s:e], flat[3][s:e],
                           flat[4][s:e], rr[s:e], escape=flat[5][s:e].astype(bool),
                           escape_g=flat[6][s:e], cfg=cfg)
            hits[s:e] = res["hit"]
        return hits.reshape(len(rows), k)

    all_rows = np.arange(m)
    grid = np.geomspace(r_lo, r_hi, n_grid)
    hits = run(np.tile(grid, (m, 1)), all_rows)
    any_hit = hits.any(axis=1)
    last = n_grid - 1 - np.argmax(hits[:, ::-1], axis=1)
    lo = np.where(any_hit, grid[last], 0.0)
    hi = np.where(any_hit, grid[np.minimum(last + 1, n_grid - 1)], grid[0])
    at_edge = any_hit & (last == n_grid - 1)
    refine = any_hit & ~at_edge

    frac = np.linspace(0.0, 1.0, n_refine + 2)[1:-1]
    for _ in range(levels):
        rows = np.nonzero(refine)[0]
        if rows.size == 0:
            break
        sub = lo[rows, None] + (hi[rows] - lo[rows])[:, None] * frac
        h = run(sub, rows)
        anyh = h.any(axis=1)
        j = n_refine - 1 - np.argmax(h[:, ::-1], axis=1)
        ar = np.arange(rows.size)
        new_lo = np.where(anyh, sub[ar, j], lo[rows])
        nxt = np.where(j + 1 < n_refine, sub[ar, np.minimum(j + 1, n_refine - 1)], hi[rows])
        new_hi = np.where(anyh, nxt, sub[:, 0])
        lo[rows], hi[rows] = new_lo, new_hi
    return lo, at_edge


def envelope(missile: MissileSpec, shooter_speed, shooter_alt, target_speed, target_alt, aspect_deg,
             escape_g=None, cfg: Optional[PhysicsConfig] = None, **kw):
    """R_max and R_ne (both in metres) for each scenario, computed in one batch."""
    s = [np.atleast_1d(np.asarray(v, dtype=float)) for v in
         (shooter_speed, shooter_alt, target_speed, target_alt, aspect_deg)]
    s = np.broadcast_arrays(*s)
    m = s[0].shape[0]
    both = [np.concatenate([a, a]) for a in s]
    esc = np.concatenate([np.zeros(m), np.ones(m)])
    rng, edge = max_launch_range(missile, *both, escape=esc, escape_g=escape_g, cfg=cfg, **kw)
    return {"r_max": rng[:m], "r_ne": rng[m:], "r_max_at_edge": edge[:m], "r_ne_at_edge": edge[m:]}

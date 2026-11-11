"""Train the neural network for one missile.

Pipeline ("learning from a simulator"):
  1. Draw random engagements (own speed/altitude, target speed/altitude/aspect).
  2. For each, the physics simulator finds R_max and R_ne (slow but accurate-ish).
  3. A small neural network learns the mapping  engagement -> (R_max, R_ne).
  4. The trained network answers in microseconds, which is what a live in-game
     helper (Part 2) needs.

Later you can add real in-game shot results to the dataset to correct the model.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from .envelope import envelope
from .missile import MissileSpec, slug
from .nn import Regressor
from .simulator import PhysicsConfig

FEATURES = ["shooter_speed_ms", "shooter_alt_m", "target_speed_ms", "target_alt_m", "cos_aspect", "sin_aspect"]
TARGETS = ["r_max_km", "r_ne_km"]

# Range of conditions the model is trained on (also used to check if a query is "in range").
SAMPLE_BOUNDS = {
    "shooter_speed_ms": (100.0, 700.0),
    "shooter_alt_m": (200.0, 15000.0),
    "target_speed_ms": (100.0, 700.0),
    "target_alt_m": (100.0, 15000.0),
    "aspect_deg": (0.0, 180.0),
}

MODEL_VERSION = 1


def make_features(shooter_speed, shooter_alt, target_speed, target_alt, aspect_deg) -> np.ndarray:
    a = np.radians(np.asarray(aspect_deg, float))
    cols = np.broadcast_arrays(*(np.atleast_1d(np.asarray(v, float)) for v in
                                 (shooter_speed, shooter_alt, target_speed, target_alt)), np.cos(a), np.sin(a))
    return np.stack(cols, axis=1)


def fingerprint(missile: MissileSpec, cfg: PhysicsConfig) -> str:
    blob = json.dumps({"m": missile.to_dict(), "c": asdict(cfg), "v": MODEL_VERSION}, sort_keys=True, default=str)
    return hashlib.sha1(blob.encode()).hexdigest()[:16]


def model_path(models_dir: Path, missile_name: str) -> Path:
    return Path(models_dir) / f"{slug(missile_name)}.npz"


def generate_dataset(missile: MissileSpec, n: int, cfg: Optional[PhysicsConfig] = None, seed: int = 0,
                     batch: int = 200, progress: Optional[Callable[[int, int], None]] = None):
    cfg = cfg or PhysicsConfig()
    rng = np.random.default_rng(seed)
    b = SAMPLE_BOUNDS
    ss = rng.uniform(*b["shooter_speed_ms"], n)
    sa = rng.uniform(*b["shooter_alt_m"], n)
    ts = rng.uniform(*b["target_speed_ms"], n)
    # Half the targets near the shooter's altitude (the common case), half anywhere.
    ta = np.where(rng.random(n) < 0.5,
                  np.clip(sa + rng.normal(0, 1500, n), *b["target_alt_m"]),
                  rng.uniform(*b["target_alt_m"], n))
    asp = rng.uniform(*b["aspect_deg"], n)
    r_max = np.zeros(n)
    r_ne = np.zeros(n)
    for s in range(0, n, batch):
        e = min(n, s + batch)
        env = envelope(missile, ss[s:e], sa[s:e], ts[s:e], ta[s:e], asp[s:e], cfg=cfg)
        r_max[s:e], r_ne[s:e] = env["r_max"], env["r_ne"]
        if progress:
            progress(e, n)
    X = make_features(ss, sa, ts, ta, asp)
    Y = np.stack([r_max, r_ne], axis=1) / 1000.0
    return X, Y


def train_model(missile: MissileSpec, models_dir: Path, n_samples: int = 1500,
                cfg: Optional[PhysicsConfig] = None, epochs: int = 3000, seed: int = 0,
                verbose: bool = True, progress: Optional[Callable[[str, float], None]] = None):
    """Generate data, train, save. Returns (Regressor, path)."""
    cfg = cfg or PhysicsConfig()
    t0 = time.time()
    if verbose:
        print(f"Simulating {n_samples} engagements for {missile.name} ...")

    def data_prog(done, total):
        if verbose:
            print(f"  {done}/{total} engagements simulated ({time.time() - t0:.0f}s)")
        if progress:
            progress("Simulating engagements", 0.8 * done / total)

    X, Y = generate_dataset(missile, n_samples, cfg, seed=seed, progress=data_prog)
    if verbose:
        print("Training neural network ...")

    def fit_prog(epoch, total, _loss):
        if progress:
            progress("Training network", 0.8 + 0.2 * epoch / total)

    reg = Regressor(X.shape[1], Y.shape[1], hidden=(64, 64), seed=seed)
    reg.fit(X, Y, epochs=epochs, verbose=verbose, progress=fit_prog)
    reg.meta.update({
        "missile": missile.name,
        "fingerprint": fingerprint(missile, cfg),
        "features": FEATURES,
        "targets": TARGETS,
        "bounds": SAMPLE_BOUNDS,
        "escape_g": cfg.escape_turn_g,
        "n_samples": n_samples,
        "trained_at": time.strftime("%Y-%m-%d %H:%M"),
        "train_seconds": round(time.time() - t0, 1),
    })
    path = model_path(models_dir, missile.name)
    reg.save(path)
    np.savez_compressed(path.with_suffix(".dataset.npz"), X=X, Y=Y)
    if verbose:
        mae = reg.meta["val_mae"]
        print(f"Saved {path}  (validation error: R_max ±{mae[0]:.2f} km, R_ne ±{mae[1]:.2f} km)")
    return reg, path

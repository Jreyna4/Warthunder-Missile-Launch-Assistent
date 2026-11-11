"""Where data lives on disk, and helpers to load all missiles / aircraft."""

from __future__ import annotations

import os
from pathlib import Path

from .aircraft import AircraftSpec
from .missile import MissileSpec
from .simulator import PhysicsConfig

ROOT = Path(os.environ.get("WTMLA_HOME", Path(__file__).resolve().parent.parent))
DATA_DIR = ROOT / "data"
MISSILE_DIR = DATA_DIR / "missiles"
AIRCRAFT_DIR = DATA_DIR / "aircraft"
MODELS_DIR = ROOT / "models"
PHYSICS_FILE = DATA_DIR / "physics.json"
WORKBOOK_FILE = DATA_DIR / "guided_weapons.xlsx"


def load_missiles(folder: Path = MISSILE_DIR) -> dict[str, MissileSpec]:
    out = {}
    for p in sorted(Path(folder).glob("*.json")):
        try:
            m = MissileSpec.load(p)
            out[m.name] = m
        except Exception as exc:  # keep going if one file is broken
            print(f"Skipping {p.name}: {exc}")
    return out


def load_aircraft(folder: Path = AIRCRAFT_DIR) -> dict[str, AircraftSpec]:
    out = {}
    for p in sorted(Path(folder).glob("*.json")):
        if p.name.startswith("_"):
            continue
        try:
            a = AircraftSpec.load(p)
            out[a.name] = a
        except Exception as exc:
            print(f"Skipping {p.name}: {exc}")
    return out


def load_physics() -> PhysicsConfig:
    return PhysicsConfig.load(PHYSICS_FILE)


def find(name: str, items: dict):
    """Case/punctuation-insensitive lookup ('aim9l' finds 'AIM-9L')."""
    if name in items:
        return items[name]
    key = "".join(c for c in name.upper() if c.isalnum())
    for k, v in items.items():
        if "".join(c for c in k.upper() if c.isalnum()) == key:
            return v
    raise KeyError(f"'{name}' not found. Available: {', '.join(items) or '(none)'}")

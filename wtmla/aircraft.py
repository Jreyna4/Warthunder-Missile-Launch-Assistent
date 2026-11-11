"""Aircraft "stat cards".

An aircraft card says which missiles a plane carries and what its radar can do.
Fill one in from the in-game stat card / wiki page and save it in data/aircraft/
(or use the "Aircraft card" editor in the GUI).

The radar values matter for radar missiles: a SARH missile (AIM-7) needs your
radar to keep tracking the target all the way, and ARH missiles with inertial
guidance / datalink (AIM-120, AAM-4) are fired from a radar lock, so your radar
range can be the real limit on how far away you can shoot.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

from .missile import slug


@dataclass
class Radar:
    name: str = ""
    search_range_km: Optional[float] = None
    track_range_km: Optional[float] = None     # range it can lock / track a fighter-sized target
    can_guide_sarh: bool = True
    can_guide_arh: bool = False                 # supports datalink / launches of ARH missiles


@dataclass
class AircraftSpec:
    name: str
    weapons: list[str] = field(default_factory=list)
    radar: Radar = field(default_factory=Radar)
    max_speed_kmh: Optional[float] = None
    service_ceiling_m: Optional[float] = None
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "AircraftSpec":
        radar = Radar(**{k: v for k, v in (d.get("radar") or {}).items() if k in Radar.__dataclass_fields__})
        return cls(name=d["name"], weapons=list(d.get("weapons") or []), radar=radar,
                   max_speed_kmh=d.get("max_speed_kmh"), service_ceiling_m=d.get("service_ceiling_m"),
                   notes=d.get("notes") or "")

    def save(self, folder: Path) -> Path:
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{slug(self.name)}.json"
        path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path) -> "AircraftSpec":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def carries(self, missile_name: str) -> bool:
        key = _key(missile_name)
        return any(_key(w) == key or key.startswith(_key(w)) or _key(w).startswith(key) for w in self.weapons)


def _key(name: str) -> str:
    return "".join(ch for ch in name.upper() if ch.isalnum())

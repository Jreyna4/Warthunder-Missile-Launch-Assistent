"""Read the "Guided weaponry data (in-game values)" spreadsheet.

The sheet is laid out with one *row per attribute* and one *column per missile*:

    Name:            | AAM-4 | AIM-54A Phoenix | ...
    Mass: [kg]       | 222   | 446,562         | ...
    ...

Three inputs are supported:
  * the whole workbook as .xlsx (File > Download > Microsoft Excel) - every AAM tab is read
  * one tab exported as .csv (File > Download > CSV)
  * a single missile pasted as two tab-separated columns (label <TAB> value)

The result is a list of MissileSpec objects, which can be saved as JSON in
data/missiles/ so the app doesn't need the spreadsheet afterwards.
"""

from __future__ import annotations

import csv
import io
import re
import urllib.request
from pathlib import Path
from typing import Iterable, Optional

from .missile import MissileSpec, classify_guidance, slug

DEFAULT_SHEET_ID = "1mrVx7RlntCrA-NUDcJeeY7HDHhrwr0hQjIjf1y-6WUM"

# Normalized row label -> MissileSpec field.
LABEL_MAP = {
    "mass": "mass",
    "fuel mass of first stage": "fuel1",
    "fuel mass of second stage": "fuel2",
    "calibre": "calibre_mm",
    "caliber": "calibre_mm",
    "length": "length",
    "force of first stage": "thrust1",
    "burn time of first stage": "burn1",
    "specific impulse of first stage": "isp1",
    "δv of first stage": "dv1",
    "first stage start delay": "start_delay1",
    "force of second stage": "thrust2",
    "burn time of second stage": "burn2",
    "specific impulse of second stage": "isp2",
    "δv of second stage": "dv2",
    "total δv": "total_dv",
    "time from launch above which second stage starts": "s2_time",
    "eta to impact below which second stage starts": "s2_eta",
    "minimum speed below which second stage starts": "s2_speed",
    "minimum closure rate below which second stage starts": "s2_closure",
    "explosive mass": "explosive_mass",
    "warhead": "warhead",
    "penetration": "penetration",
    "proximity fuze": "proximity_fuze",
    "proximity fuze range": "prox_range",
    "proximity fuze delay": "prox_delay",
    "guidance type": "guidance_type",
    "band": "band",
    "signal": "signal",
    "guidance start delay": "guidance_start_delay",
    "guidance duration": "guidance_duration",
    "gimbal limit": "gimbal_limit",
    "track rate": "track_rate",
    "uncaged seeker before launch": "uncaged_before_launch",
    "baseline lock range from all-aspect": "lock_range_all_aspect",
    "baseline lock range from rear-aspect": "lock_range_rear_aspect",
    "range to target at which the seeker is turned on": "seeker_on_range",
    "maximum lock range": "max_lock_range",
    "can lock after launch": "can_lock_after_launch",
    "inertial guidance drift speed": "inertial_drift",
    "datalink": "datalink",
    "proportional navigation multiplier": "pn_mult",
    "drag coefficient multiplier": "drag_mult",
    "simplified ballistic coefficient": "ballistic_coeff",
    "maximum fin angle of attack": "max_fin_aoa",
    "wing area multiplier": "wing_area_mult",
    "start speed": "start_speed",
    "maximum speed": "max_speed",
    "minimum range": "min_range",
    "flight range limit": "flight_range_limit",
    "maximum g-load": "max_g",
    "thrust vectoring": "thrust_vectoring",
    "loft": "loft",
    "loft angle": "loft_angle",
    "target elevation": "loft_target_elevation",
    "maximum target angular change": "loft_max_target_angular_change",
    "additional notes": "notes",
}

# Rows that appear several times and form a schedule of (x, percent) points.
SCHEDULE_ROWS = {
    "flight time when pull limit reaches x%": "pull_schedule",
    "eta to impact when prop multiplier reaches x%": "pn_schedule",
}

# Fields that hold text rather than numbers.
TEXT_FIELDS = {"warhead", "guidance_type", "band", "signal", "notes"}
BOOL_FIELDS = {"proximity_fuze", "uncaged_before_launch", "can_lock_after_launch",
               "datalink", "thrust_vectoring", "loft"}


def normalize_label(label: str) -> str:
    """'Mass: [kg]' -> 'mass'; drops [units], (comments), trailing colon, extra spaces."""
    s = str(label or "")
    s = re.sub(r"\[[^\]]*\]", " ", s)
    s = re.sub(r"\([^)]*\)", " ", s)
    s = s.replace(":", " ")
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s


def parse_value(raw):
    """Convert a sheet cell to float / bool / str / None (handles '3,667' decimal commas)."""
    if raw is None:
        return None
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, (int, float)):
        return float(raw)
    s = str(raw).strip()
    if s in ("", "-", "—", "–", "N/A", "n/a", "?"):
        return None
    low = s.lower()
    if low in ("yes", "true", "y"):
        return True
    if low in ("no", "false", "n"):
        return False
    num = s.replace(" ", "").replace(" ", "")
    if re.fullmatch(r"[-+]?\d+(,\d+)?", num):
        return float(num.replace(",", "."))
    if re.fullmatch(r"[-+]?\d{1,3}(\.\d{3})+(,\d+)?", num):  # 1.234,5 style
        return float(num.replace(".", "").replace(",", "."))
    if re.fullmatch(r"[-+]?\d*\.\d+|[-+]?\d+\.?", num):
        return float(num)
    return s


def parse_schedule_point(raw) -> Optional[tuple]:
    """'0,31 s / 100%' -> (0.31, 100.0)."""
    if raw is None:
        return None
    s = str(raw).strip()
    if s in ("", "-"):
        return None
    m = re.match(r"\s*([-+]?\d+(?:[.,]\d+)?)\s*s?\s*/\s*([-+]?\d+(?:[.,]\d+)?)\s*%?", s)
    if not m:
        return None
    return (float(m.group(1).replace(",", ".")), float(m.group(2).replace(",", ".")))


def _apply(spec: MissileSpec, label: str, raw) -> None:
    key = normalize_label(label)
    if not key or key == "name":
        return
    if key in SCHEDULE_ROWS:
        pt = parse_schedule_point(raw)
        if pt is not None:
            getattr(spec, SCHEDULE_ROWS[key]).append(pt)
        return
    value = parse_value(raw)
    fname = LABEL_MAP.get(key)
    if fname is None:
        # Fallbacks for small label differences between tabs.
        if "lock range from" in key and "rear" in key:
            fname = "lock_range_rear_aspect"
        elif "lock range from" in key and "all" in key:
            fname = "lock_range_all_aspect"
    if fname is None:
        if value is not None:
            spec.extra[label.strip()] = value
        return
    if value is None:
        return
    if fname in TEXT_FIELDS:
        value = str(value) if not isinstance(value, float) else f"{value:g}"
    elif fname in BOOL_FIELDS:
        value = bool(value) if isinstance(value, bool) else str(value).lower().startswith("y")
    elif not isinstance(value, float):
        spec.extra[label.strip()] = value
        return
    setattr(spec, fname, value)


def _finish(spec: MissileSpec) -> MissileSpec:
    spec.category = classify_guidance(spec.guidance_type, spec.source_tab)
    spec.pn_schedule.sort()
    spec.pull_schedule.sort()
    return spec


def parse_grid(rows: list[list], tab_name: Optional[str] = None) -> list[MissileSpec]:
    """Parse one tab given as a list of rows (first column = labels)."""
    header_idx = None
    for i, row in enumerate(rows):
        if row and normalize_label(row[0]) == "name":
            header_idx = i
            break
    if header_idx is None:
        return []
    header = rows[header_idx]
    specs: dict[int, MissileSpec] = {}
    for col in range(1, len(header)):
        name = header[col]
        if name is None or str(name).strip() in ("", "-"):
            continue
        specs[col] = MissileSpec(name=" ".join(str(name).split()), source_tab=tab_name)
    for row in rows[header_idx + 1:]:
        if not row or row[0] is None or str(row[0]).strip() == "":
            continue
        for col, spec in specs.items():
            raw = row[col] if col < len(row) else None
            _apply(spec, str(row[0]), raw)
    return [_finish(s) for s in specs.values()]


def parse_csv_text(text: str, tab_name: Optional[str] = None) -> list[MissileSpec]:
    dialect = "excel-tab" if text.count("\t") > text.count(",") / 2 and "\t" in text else "excel"
    rows = list(csv.reader(io.StringIO(text), dialect=dialect))
    return parse_grid(rows, tab_name)


def parse_single_tsv(text: str) -> list[MissileSpec]:
    """Parse one missile pasted as 'label<TAB>value' lines (like a copied sheet column)."""
    rows = [line.split("\t") for line in text.splitlines()]
    return parse_grid(rows)


def is_air_to_air_tab(tab_name: str) -> bool:
    return tab_name.strip().upper().startswith("AAM")


def parse_xlsx(path: Path, only_air_to_air: bool = True) -> list[MissileSpec]:
    try:
        import openpyxl
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Reading .xlsx needs openpyxl:  pip install openpyxl") from exc
    wb = openpyxl.load_workbook(path, data_only=True, read_only=True)
    result: list[MissileSpec] = []
    for ws in wb.worksheets:
        if only_air_to_air and not is_air_to_air_tab(ws.title):
            continue
        rows = [list(r) for r in ws.iter_rows(values_only=True)]
        result.extend(parse_grid(rows, ws.title))
    return result


def load_any(path: Path, tab_name: Optional[str] = None) -> list[MissileSpec]:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in (".xlsx", ".xlsm"):
        return parse_xlsx(path)
    text = path.read_text(encoding="utf-8-sig")
    if suffix == ".csv":
        return parse_csv_text(text, tab_name or path.stem)
    return parse_single_tsv(text)


def download_workbook(dest: Path, sheet_id: str = DEFAULT_SHEET_ID, timeout: float = 60) -> Path:
    """Download the whole public spreadsheet as .xlsx (run this on your own PC)."""
    url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=xlsx"
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 wtmla"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = resp.read()
    if not data.startswith(b"PK"):
        raise RuntimeError("Download did not return an .xlsx file. Is the sheet shared publicly? "
                           "You can also use File > Download > Microsoft Excel and import that file.")
    dest.write_bytes(data)
    return dest


def save_specs(specs: Iterable[MissileSpec], out_dir: Path) -> list[Path]:
    paths = []
    for spec in specs:
        p = Path(out_dir) / f"{slug(spec.name)}.json"
        spec.save(p)
        paths.append(p)
    return paths

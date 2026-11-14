"""Create an aircraft card from a War Thunder wiki page.

    https://wiki.warthunder.com/unit/su_30mkk   (or just "su_30mkk", or a saved .html page)

What the wiki page gives us:
  * the aircraft name
  * flight performance (max speed and its altitude, max altitude, limits) - Realistic, upgraded values
  * the radar popup: radar model, bands, TWS / search / track
  * "List of available weapons": every weapon; the air-to-air missiles are kept,
    each with its popup (guidance, lock ranges, launch range, max speed, overload ...)

What it does NOT give: radar search / lock ranges. Add those yourself with "Edit card".

Only Python's built-in html.parser is used, so no extra packages are needed.
"""

from __future__ import annotations

import re
import time
import urllib.request
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterator, Optional

from .aircraft import AircraftSpec, Radar
from .missile import ARH, SARH, classify_guidance, slug

WIKI_UNIT_URL = "https://wiki.warthunder.com/unit/"
VOID_TAGS = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta",
             "param", "source", "track", "wbr"}
SKIP_TEXT = {"script", "style", "svg"}


# --------------------------------------------------------------------------- tiny DOM
class Node:
    __slots__ = ("tag", "attrs", "children", "parent")

    def __init__(self, tag: str, attrs: dict, parent: Optional["Node"] = None):
        self.tag, self.attrs, self.children, self.parent = tag, attrs, [], parent

    @property
    def classes(self) -> set:
        return set((self.attrs.get("class") or "").split())

    def iter(self) -> Iterator["Node"]:
        for c in self.children:
            if isinstance(c, Node):
                yield c
                yield from c.iter()

    def find_all(self, tag: Optional[str] = None, cls: Optional[str] = None) -> list["Node"]:
        return [n for n in self.iter() if (tag is None or n.tag == tag) and (cls is None or cls in n.classes)]

    def find(self, tag: Optional[str] = None, cls: Optional[str] = None) -> Optional["Node"]:
        for n in self.iter():
            if (tag is None or n.tag == tag) and (cls is None or cls in n.classes):
                return n
        return None

    def element_children(self) -> list["Node"]:
        return [c for c in self.children if isinstance(c, Node)]

    def own_text(self) -> str:
        return " ".join(c.strip() for c in self.children if isinstance(c, str) and c.strip())

    def text(self) -> str:
        parts: list[str] = []

        def walk(n: Node):
            if n.tag in SKIP_TEXT:
                return
            for c in n.children:
                if isinstance(c, str):
                    parts.append(c)
                else:
                    walk(c)

        walk(self)
        return re.sub(r"\s+", " ", " ".join(parts)).strip()

    def has_check(self) -> Optional[bool]:
        """Wiki shows ✓ / ✗ as bootstrap icons."""
        for n in self.iter():
            c = n.classes
            if "bi-check-circle-fill" in c or "bi-check-circle" in c or "bi-check" in c:
                return True
            if any(x.startswith("bi-x") for x in c):
                return False
        return None


class _Builder(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.root = Node("#root", {})
        self.cur = self.root

    def handle_starttag(self, tag, attrs):
        node = Node(tag, {k: (v or "") for k, v in attrs}, self.cur)
        self.cur.children.append(node)
        if tag not in VOID_TAGS:
            self.cur = node

    def handle_startendtag(self, tag, attrs):
        self.cur.children.append(Node(tag, {k: (v or "") for k, v in attrs}, self.cur))

    def handle_endtag(self, tag):
        n = self.cur
        while n is not None and n.tag != tag:
            n = n.parent
        if n is not None and n.parent is not None:
            self.cur = n.parent

    def handle_data(self, data):
        if self.cur.tag not in ("script", "style"):
            self.cur.children.append(data)


def parse_html(html: str) -> Node:
    b = _Builder()
    b.feed(html)
    b.close()
    return b.root


# --------------------------------------------------------------------------- helpers
def clean_name(text: str) -> str:
    """Remove the game's icon glyphs (e.g. U+2417 shown before premium/squadron names)."""
    out = []
    for ch in text:
        cp = ord(ch)
        if 0x2400 <= cp <= 0x243F or 0xE000 <= cp <= 0xF8FF or cp < 32 or 0x25A0 <= cp <= 0x25FF:
            continue
        out.append(ch)
    return re.sub(r"\s+", " ", "".join(out)).strip()


def first_number(text: str) -> Optional[float]:
    m = re.search(r"-?\d[\d,\s]*(?:\.\d+)?", text or "")
    if not m:
        return None
    return float(m.group(0).replace(",", "").replace(" ", ""))


def all_numbers(text: str) -> list[float]:
    return [float(x.replace(",", "")) for x in re.findall(r"-?\d[\d,]*(?:\.\d+)?", text or "")]


def realistic_value(val: Node) -> str:
    """A value cell may hold four numbers (RB/AB, upgraded/stock): keep Realistic upgraded."""
    ref = val.find(cls="show-char-rb-mod-ref")
    if ref is not None:
        unit = val.own_text()
        return f"{ref.text()} {unit}".strip()
    return val.text()


def popover_info(attr_html: str) -> dict:
    """Turn a data-feature-popover HTML snippet into {label: value}."""
    root = parse_html(attr_html or "")
    info: dict = {}
    header = root.find(cls="game-unit_popover-header")
    if header is not None:
        info["_title"] = header.text()
    for line in root.iter():
        if not ({"game-unit_chars-line", "game-unit_chars-subline"} & line.classes):
            continue
        kids = line.element_children()
        if not kids:
            continue
        label = kids[0].text()
        val_node = line.find(cls="game-unit_chars-value") or (kids[1] if len(kids) > 1 else None)
        if not label or val_node is None or val_node is kids[0]:
            continue
        value = val_node.text()
        check = val_node.has_check()
        if check is not None:
            value = ("yes " if check else "no ") + value
        info[label] = value.strip()
    return info


def strip_weapon_name(text: str) -> str:
    """'2 × R-73E air-to-air missiles' -> 'R-73E'."""
    s = clean_name(text)
    s = re.sub(r"^\d+\s*[×xX]\s*", "", s)
    s = re.split(r"\s+air-to-air\b", s, flags=re.I)[0]
    return s.strip()


# --------------------------------------------------------------------------- page model
@dataclass
class WikiUnit:
    name: str
    url: str = ""
    weapons: list[str] = field(default_factory=list)              # air-to-air missiles
    weapon_details: dict = field(default_factory=dict)            # name -> popup info
    has_radar: bool = False
    radar_info: dict = field(default_factory=dict)
    max_speed_kmh: Optional[float] = None
    max_speed_alt_m: Optional[float] = None
    max_altitude_m: Optional[float] = None
    mach_limit: Optional[float] = None
    ias_limit_kmh: Optional[float] = None
    g_limit: Optional[float] = None
    characteristics: dict = field(default_factory=dict)

    @property
    def radar_name(self) -> str:
        return self.radar_info.get("Model", "")

    def guidance_categories(self) -> set:
        cats = set()
        for d in self.weapon_details.values():
            g = d.get("Guidance", "")
            cat = classify_guidance(g)
            low = g.lower()
            if "semi" in low:
                cat = SARH
            elif "active radar" in low:
                cat = ARH
            cats.add(cat)
        return cats


def parse_unit_html(html: str, url: str = "") -> WikiUnit:
    root = parse_html(html)
    h1 = root.find(cls="game-unit_name") or root.find("h1")
    name = clean_name(h1.text()) if h1 is not None else ""
    if not name:
        t = root.find("title")
        name = clean_name(t.text().split("|")[0]) if t is not None else "Unknown aircraft"
    unit = WikiUnit(name=name, url=url)

    # Characteristics: header -> realistic value text.
    for line in root.find_all(cls="game-unit_chars-line"):
        hdr = line.find(cls="game-unit_chars-header")
        val = line.find(cls="game-unit_chars-value")
        if hdr is None or val is None:
            continue
        unit.characteristics.setdefault(hdr.text(), realistic_value(val))
    for block in root.find_all(cls="game-unit_chars-block"):
        hdr = block.find(cls="game-unit_chars-header")
        if hdr is None or hdr.text() != "Max speed":
            continue
        sub = block.find(cls="game-unit_chars-subline")
        if sub is None:
            continue
        val = sub.find(cls="game-unit_chars-value")
        kids = sub.element_children()
        if val is not None:
            unit.max_speed_kmh = first_number(realistic_value(val))
        if kids and kids[0] is not val:
            unit.max_speed_alt_m = first_number(kids[0].text())
        break

    c = unit.characteristics
    unit.max_altitude_m = first_number(c.get("Max altitude", ""))
    unit.mach_limit = first_number(c.get("Mach Number Limit", ""))
    unit.ias_limit_kmh = first_number(c.get("Max Speed Limit (IAS)", ""))
    g = all_numbers(c.get("G limit", ""))
    unit.g_limit = max(g) if g else None

    # Radar popup.
    for btn in root.find_all(cls="game-unit_feature"):
        if "radar" in btn.text().lower():
            unit.radar_info = popover_info(btn.attrs.get("data-feature-popover", ""))
            radar = unit.radar_info.get("Radar", "")
            unit.has_radar = radar.startswith("yes") or bool(unit.radar_info.get("Model"))
            break

    # Weapons table.
    seen = set()
    table = root.find("table", cls="game-unit_slot-list")
    rows = table.find_all("tr") if table is not None else []
    for tr in rows:
        cells = [c for c in tr.element_children() if c.tag == "td"]
        if not cells:
            continue
        label = cells[0].text()
        if "air-to-air" not in label.lower():
            continue
        wname = strip_weapon_name(label)
        if not wname or wname.upper() in seen:
            continue
        seen.add(wname.upper())
        unit.weapons.append(wname)
        btn = cells[0].find("button") or cells[0]
        info = popover_info(btn.attrs.get("data-feature-popover", ""))
        info.pop("_title", None)
        if info:
            unit.weapon_details[wname] = info

    if table is None:  # fallback if the page layout changes
        for m in re.finditer(r">\s*(?:\d+\s*[×x]\s*)?([^<>]{2,40}?)\s+air-to-air missiles?\s*<", html, re.I):
            wname = clean_name(m.group(1))
            if wname.upper() not in seen:
                seen.add(wname.upper())
                unit.weapons.append(wname)
    return unit


# --------------------------------------------------------------------------- fetching
def normalize_source(text: str) -> str:
    """URL, unit id ('su_30mkk') or a path to a saved .html page."""
    s = text.strip().strip('"').strip("'")
    if re.match(r"https?://", s, re.I):
        return s.split("#")[0]
    if Path(s).suffix.lower() in (".html", ".htm") or Path(s).exists():
        return s
    if re.fullmatch(r"[A-Za-z0-9_\-]+", s):
        return WIKI_UNIT_URL + s.lower()
    raise ValueError("Give a wiki link like https://wiki.warthunder.com/unit/su_30mkk, "
                     "a unit name like su_30mkk, or a saved .html page.")


def fetch_html(source: str, timeout: float = 30) -> tuple[str, str]:
    src = normalize_source(source)
    if not re.match(r"https?://", src, re.I):
        return Path(src).read_text(encoding="utf-8", errors="replace"), ""
    req = urllib.request.Request(src, headers={
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) wtmla-missile-assistant",
        "Accept-Language": "en-US,en;q=0.9",
    })
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            charset = resp.headers.get_content_charset() or "utf-8"
    except Exception as exc:
        raise RuntimeError(f"Could not download {src}: {exc}\n\nTip: open the page in your browser, "
                           "press Ctrl+S, save it as 'Webpage, HTML only', and import that .html file.") from exc
    return raw.decode(charset, errors="replace"), src


def fetch_unit(source: str) -> WikiUnit:
    html, url = fetch_html(source)
    unit = parse_unit_html(html, url)
    if not unit.weapons and not unit.characteristics:
        raise RuntimeError("That page doesn't look like a War Thunder wiki vehicle page.")
    return unit


# --------------------------------------------------------------------------- card
def unit_to_aircraft(unit: WikiUnit, existing: Optional[AircraftSpec] = None) -> AircraftSpec:
    cats = unit.guidance_categories()
    old_radar = existing.radar if existing else Radar()
    radar = Radar(
        name=unit.radar_name or old_radar.name,
        search_range_km=old_radar.search_range_km,
        track_range_km=old_radar.track_range_km,
        can_guide_sarh=unit.has_radar and (SARH in cats or old_radar.can_guide_sarh and existing is not None),
        can_guide_arh=unit.has_radar and (ARH in cats or old_radar.can_guide_arh and existing is not None),
    )
    lines = [f"Imported from the War Thunder wiki on {time.strftime('%Y-%m-%d')}."]
    if unit.radar_info:
        bits = [f"{k}: {v}" for k, v in unit.radar_info.items() if not k.startswith("_")]
        lines.append("Radar/sensors: " + "; ".join(bits))
    if unit.has_radar and not radar.track_range_km:
        lines.append("The wiki doesn't list radar ranges - add the lock/track range for better SARH/ARH advice.")
    if unit.max_speed_kmh:
        lines.append(f"Max speed {unit.max_speed_kmh:g} km/h at {unit.max_speed_alt_m or 0:g} m"
                     + (f", Mach limit {unit.mach_limit:g}" if unit.mach_limit else "")
                     + (f", G limit {unit.g_limit:g}" if unit.g_limit else ""))
    for w in unit.weapons:
        d = unit.weapon_details.get(w, {})
        keys = ["Guidance", "Aspect", "Lock range in rear-aspect", "Lock range in all-aspect", "Launch range",
                "Maximum speed", "Maximum overload", "Missile guidance time"]
        desc = ", ".join(f"{k.lower()} {d[k]}" for k in keys if d.get(k, "-") not in ("-", ""))
        lines.append(f"{w}: {desc}" if desc else w)
    if existing and existing.notes and not existing.notes.startswith("Imported from the War Thunder wiki"):
        lines.append("Your earlier notes: " + existing.notes)
    return AircraftSpec(
        name=unit.name,
        weapons=list(unit.weapons),
        radar=radar,
        max_speed_kmh=unit.max_speed_kmh,
        service_ceiling_m=unit.max_altitude_m,
        notes="\n".join(lines),
        wiki_url=unit.url,
        weapon_details=unit.weapon_details,
    )


def import_aircraft(source: str, folder: Path) -> tuple[AircraftSpec, Path, WikiUnit, bool]:
    """Download + parse + save. Keeps radar ranges you typed in earlier for the same aircraft."""
    unit = fetch_unit(source)
    path = Path(folder) / f"{slug(unit.name)}.json"
    existing = AircraftSpec.load(path) if path.exists() else None
    card = unit_to_aircraft(unit, existing)
    return card, card.save(folder), unit, existing is not None

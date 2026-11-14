from pathlib import Path

from wtmla.missile import MissileSpec, names_match
from wtmla.sheets import dedupe
from wtmla.wiki import normalize_source, parse_unit_html, strip_weapon_name, unit_to_aircraft

FIX = Path(__file__).parent / "fixtures"


def test_parse_wiki_unit_page():
    html = (FIX / "wiki_su_30mkk.html").read_text(encoding="utf-8")
    u = parse_unit_html(html, "https://wiki.warthunder.com/unit/su_30mkk")
    assert u.name == "Su-30MKK"                      # icon glyph removed, <script> ignored
    assert u.weapons == ["R-73E", "PL-8B", "R-27ER1", "R-27ET1", "RVV-AE", "PL-12"]  # no rockets/bombs/ARMs
    assert (u.max_speed_kmh, u.max_speed_alt_m, u.max_altitude_m) == (2304, 12000, 16000)
    assert (u.mach_limit, u.ias_limit_kmh, u.g_limit) == (2.1, 1540, 11)
    assert u.has_radar and u.radar_name == "N001VE, ESM L-150"
    assert u.weapon_details["R-73E"]["Lock range in rear-aspect"] == "11 km"
    card = unit_to_aircraft(u)
    assert card.radar.can_guide_sarh and card.radar.can_guide_arh
    assert card.wiki_url.endswith("su_30mkk") and card.service_ceiling_m == 16000


def test_weapon_names_and_sources():
    assert strip_weapon_name("2 × R-73E air-to-air missiles") == "R-73E"
    assert normalize_source("su_30mkk") == "https://wiki.warthunder.com/unit/su_30mkk"
    assert normalize_source("https://wiki.warthunder.com/unit/su_30mkk#weapons").endswith("su_30mkk")


def test_names_match():
    assert names_match("AIM-7F", "AIM-7F Sparrow")
    assert names_match("aim9l", "AIM-9L")
    assert not names_match("R-27R", "R-27R1")
    assert not names_match("Super 530D", "Super 530F")


def test_dedupe_keeps_different_versions():
    a = MissileSpec("AIM-9L", source_tab="AAM (IR all-aspect)", mass=86)
    b = MissileSpec("AIM-9L", source_tab="AAM (IR all-aspect)", mass=86)
    c = MissileSpec("aim-9l", source_tab="AAM (IR rear-aspect)", mass=85)
    out, report = dedupe([a, b, c])
    assert [m.name for m in out] == ["AIM-9L", "aim-9l [AAM (IR rear-aspect)]"]
    assert len(report) == 2

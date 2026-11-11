from pathlib import Path

from wtmla.missile import ARH, IR, SARH, classify_guidance
from wtmla.sheets import load_any, normalize_label, parse_csv_text, parse_schedule_point, parse_value

FIX = Path(__file__).parent / "fixtures"


def test_parse_value_decimal_comma_and_flags():
    assert parse_value("3,667") == 3.667
    assert parse_value("27025") == 27025.0
    assert parse_value("-") is None
    assert parse_value("Yes") is True and parse_value("No") is False
    assert parse_value("SAPHE") == "SAPHE"


def test_labels_and_schedules():
    assert normalize_label("Mass: [kg]") == "mass"
    assert normalize_label("Maximum lock range (hard limit): [km]") == "maximum lock range"
    assert parse_schedule_point("0,31 s / 100%") == (0.31, 100.0)
    assert parse_schedule_point("-") is None


def test_classify_sarh_before_arh():
    assert classify_guidance("SARH") == SARH
    assert classify_guidance("ARH + IOG + DL") == ARH
    assert classify_guidance(None, "AAM (IR all-aspect)") == IR


def test_single_missile_tsv():
    (m,) = load_any(FIX / "AAM-4.tsv")
    assert m.name == "AAM-4" and m.category == ARH
    assert m.mass == 222 and m.length == 3.667 and m.thrust2 == 9200
    assert m.proximity_fuze is True and m.prox_range == 12
    assert m.pn_schedule == [(10.0, 100.0), (25.0, 80.0), (50.0, 50.0)]
    assert m.pull_schedule == [(0.3, 0.0), (0.31, 100.0)]
    # Burnout-mass ballistic coefficient matches the sheet's "simplified ballistic coefficient".
    bc = m.burnout_mass / (m.drag_mult * m.reference_area)
    assert abs(bc - m.ballistic_coeff) / m.ballistic_coeff < 0.01


def test_multi_column_csv():
    text = ('Name:,AIM-7X,AIM-9X\n'
            'Physical properties,,\n'
            'Mass: [kg],"230,5",86\n'
            'Guidance type:,SARH,IR\n'
            'Baseline lock range from rear-aspect: [km],-,"5,5"\n')
    a, b = parse_csv_text(text, "test")
    assert (a.name, a.mass, a.category) == ("AIM-7X", 230.5, SARH)
    assert (b.name, b.category, b.lock_range_rear_aspect) == ("AIM-9X", IR, 5.5)

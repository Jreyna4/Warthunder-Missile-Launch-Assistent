"""Command line interface.

    python -m wtmla                      open the GUI
    python -m wtmla download             download the spreadsheet and import every AAM
    python -m wtmla import FILE          import an .xlsx / .csv / pasted .txt export
    python -m wtmla add-aircraft URL     aircraft card from a wiki page, e.g. .../unit/su_30mkk
    python -m wtmla list                 show missiles, aircraft and trained models
    python -m wtmla advise AAM-4 --speed 1100 --alt 9000 [--aircraft "F/A-18C Early"]
    python -m wtmla train AAM-4          train the neural network for one missile ("all" for every one)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import library as lib
from .advisor import ModelCache, advise
from .missile import names_match
from .sheets import dedupe, download_workbook, load_any, save_specs
from .train import model_path, train_model
from .units import ALT_UNITS, SPEED_UNITS, to_metres, to_tas_ms


def cmd_download(args):
    dest = Path(args.out or lib.WORKBOOK_FILE)
    print(f"Downloading spreadsheet to {dest} ...")
    download_workbook(dest, args.sheet_id)
    args.file = str(dest)
    args.tab = None
    cmd_import(args)


def cmd_import(args):
    specs = load_any(Path(args.file), tab_name=args.tab)
    if not specs:
        print("No missiles found. Is the first column the row labels with a 'Name:' row?")
        return 1
    specs, dup_report = dedupe(specs)
    for line in dup_report:
        print(f"  note: {line}")
    paths = save_specs(specs, lib.MISSILE_DIR)
    for s, p in zip(specs, paths):
        print(f"  {s.name:28s} {s.category:6s} -> {p.relative_to(lib.ROOT)}")
    print(f"Imported {len(specs)} missiles.")


def cmd_list(args):
    missiles = lib.load_missiles()
    print("Missiles:")
    for name, m in missiles.items():
        trained = "trained" if model_path(lib.MODELS_DIR, name).exists() else ""
        print(f"  {name:28s} {m.category:6s} {m.guidance_type or '':22s} {trained}")
    print("Aircraft:")
    for name, a in lib.load_aircraft().items():
        print(f"  {name:28s} weapons: {', '.join(a.weapons)}")


def cmd_advise(args):
    missiles = lib.load_missiles()
    missile = lib.find(args.missile, missiles)
    aircraft = lib.find(args.aircraft, lib.load_aircraft()) if args.aircraft else None
    alt = to_metres(args.alt, args.alt_unit)
    talt = to_metres(args.target_alt, args.alt_unit) if args.target_alt is not None else alt
    spd = to_tas_ms(args.speed, args.speed_unit, alt)
    tspd = to_tas_ms(args.target_speed, args.speed_unit, talt) if args.target_speed is not None else spd
    adv = advise(missile, spd, alt, tspd, talt, args.aspect, aircraft=aircraft, cfg=lib.load_physics(),
                 models=ModelCache(lib.MODELS_DIR), force_simulator=args.sim)
    print(adv.text())


def cmd_add_aircraft(args):
    from .wiki import import_aircraft
    card, path, unit, merged = import_aircraft(args.source, lib.AIRCRAFT_DIR)
    missiles = lib.load_missiles()
    print(f"{'Updated' if merged else 'Saved'} aircraft card: {card.name} -> {path.relative_to(lib.ROOT)}")
    print(f"  Radar: {card.radar.name or 'none'}")
    for w in card.weapons:
        found = [n for n in missiles if card.carries(n) and names_match(w, n)]
        print(f"  {w:14s} {'-> ' + ', '.join(found) if found else 'NOT in the imported spreadsheet'}")
    if unit.has_radar and not card.radar.track_range_km:
        print("  Tip: the wiki has no radar ranges; add the lock range in the GUI with 'Edit card'.")


def cmd_train(args):
    missiles = lib.load_missiles()
    targets = list(missiles.values()) if args.missile.lower() == "all" else [lib.find(args.missile, missiles)]
    for m in targets:
        if m.category not in ("IR", "SARH", "ARH"):
            print(f"Skipping {m.name} ({m.category})")
            continue
        train_model(m, lib.MODELS_DIR, n_samples=args.samples, cfg=lib.load_physics(), epochs=args.epochs)


def main(argv=None):
    p = argparse.ArgumentParser(prog="wtmla", description="War Thunder missile launch assistant")
    sub = p.add_subparsers(dest="cmd")

    d = sub.add_parser("download", help="download the Google Sheet (.xlsx) and import it")
    d.add_argument("--sheet-id", default="1mrVx7RlntCrA-NUDcJeeY7HDHhrwr0hQjIjf1y-6WUM")
    d.add_argument("--out", default=None)
    d.set_defaults(func=cmd_download)

    i = sub.add_parser("import", help="import a spreadsheet export (.xlsx, .csv or pasted .txt)")
    i.add_argument("file")
    i.add_argument("--tab", default=None, help="tab name for a .csv export, e.g. 'AAM (SARH)'")
    i.set_defaults(func=cmd_import)

    w = sub.add_parser("add-aircraft", help="create an aircraft card from a War Thunder wiki page")
    w.add_argument("source", help="wiki link, unit name (su_30mkk) or a saved .html page")
    w.set_defaults(func=cmd_add_aircraft)

    sub.add_parser("list", help="list missiles, aircraft and trained models").set_defaults(func=cmd_list)

    a = sub.add_parser("advise", help="best launch range for one situation")
    a.add_argument("missile")
    a.add_argument("--speed", type=float, required=True, help="your speed")
    a.add_argument("--speed-unit", default="km/h TAS", choices=SPEED_UNITS)
    a.add_argument("--alt", type=float, required=True, help="your altitude")
    a.add_argument("--alt-unit", default="m", choices=ALT_UNITS)
    a.add_argument("--target-speed", type=float, default=None, help="default: same as yours")
    a.add_argument("--target-alt", type=float, default=None, help="default: same as yours")
    a.add_argument("--aspect", type=float, default=0.0, help="0 head-on, 90 beam, 180 tail chase")
    a.add_argument("--aircraft", default=None)
    a.add_argument("--sim", action="store_true", help="always use the physics simulator")
    a.set_defaults(func=cmd_advise)

    t = sub.add_parser("train", help="train the neural network for a missile (or 'all')")
    t.add_argument("missile")
    t.add_argument("--samples", type=int, default=1500)
    t.add_argument("--epochs", type=int, default=3000)
    t.set_defaults(func=cmd_train)

    sub.add_parser("gui", help="open the window (default)")

    args = p.parse_args(argv)
    if args.cmd in (None, "gui"):
        from .gui import run
        return run()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

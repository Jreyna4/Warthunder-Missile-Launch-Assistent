"""Tkinter desktop window for Part 1.

Pick your aircraft and missile, type your speed / altitude (and optionally the
target's), press Calculate. Heavy work (simulator, training) runs in a background
thread so the window never freezes.
"""

from __future__ import annotations

import queue
import threading
import tkinter as tk
import traceback
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from . import library as lib
from .advisor import Advice, ModelCache, advise
from .aircraft import AircraftSpec, Radar
from .missile import slug
from .sheets import download_workbook, load_any, save_specs
from .train import fingerprint, model_path, train_model
from .units import ALT_UNITS, SPEED_UNITS, to_metres, to_tas_ms

NO_AIRCRAFT = "(no aircraft card)"
ASPECTS = ["Head-on (0°)", "Flanking (45°)", "Beam (90°)", "Rear quarter (135°)", "Tail chase (180°)"]

GREEN, ORANGE, RED, GREY = "#3aa655", "#e8a33d", "#d9534f", "#888888"


def parse_aspect(text: str) -> float:
    t = text.strip()
    if "(" in t and "°" in t:
        t = t[t.index("(") + 1:t.index("°")]
    return float(t.replace("°", ""))


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("War Thunder Missile Launch Assistant")
        self.minsize(980, 620)
        self.jobs: "queue.Queue" = queue.Queue()
        self.busy = False
        self.models = ModelCache(lib.MODELS_DIR)
        self.physics = lib.load_physics()
        self.missiles = {}
        self.aircraft = {}
        self.last_advice: Advice | None = None

        self._build_menu()
        self._build_layout()
        self.reload_data()
        self.after(100, self._poll_jobs)

    # ------------------------------------------------------------------ layout
    def _build_menu(self):
        bar = tk.Menu(self)
        data = tk.Menu(bar, tearoff=False)
        data.add_command(label="Import spreadsheet file (.xlsx / .csv / .txt)...", command=self.import_file)
        data.add_command(label="Download spreadsheet from Google Sheets", command=self.download_sheet)
        data.add_command(label="Reload data folder", command=self.reload_data)
        data.add_separator()
        data.add_command(label="Quit", command=self.destroy)
        bar.add_cascade(label="Data", menu=data)
        help_menu = tk.Menu(bar, tearoff=False)
        help_menu.add_command(label="How to read the result", command=self.show_help)
        bar.add_cascade(label="Help", menu=help_menu)
        self.config(menu=bar)

    def _build_layout(self):
        root = ttk.Frame(self, padding=10)
        root.pack(fill="both", expand=True)
        root.columnconfigure(0, weight=0)
        root.columnconfigure(1, weight=1)
        root.rowconfigure(0, weight=1)

        left = ttk.Frame(root)
        left.grid(row=0, column=0, sticky="nsw", padx=(0, 10))

        # Aircraft + missile ------------------------------------------------
        setup = ttk.LabelFrame(left, text="1. Aircraft and missile", padding=8)
        setup.pack(fill="x")
        ttk.Label(setup, text="Aircraft:").grid(row=0, column=0, sticky="w")
        self.aircraft_var = tk.StringVar(value=NO_AIRCRAFT)
        self.aircraft_box = ttk.Combobox(setup, textvariable=self.aircraft_var, state="readonly", width=28)
        self.aircraft_box.grid(row=0, column=1, columnspan=2, sticky="ew", pady=2)
        self.aircraft_box.bind("<<ComboboxSelected>>", lambda e: self._refresh_missile_list())
        ttk.Button(setup, text="New card", command=lambda: self.edit_aircraft(None)).grid(row=1, column=1, sticky="ew")
        ttk.Button(setup, text="Edit card", command=self._edit_selected_aircraft).grid(row=1, column=2, sticky="ew")

        ttk.Label(setup, text="Missile:").grid(row=2, column=0, sticky="w", pady=(8, 0))
        self.missile_var = tk.StringVar()
        self.missile_box = ttk.Combobox(setup, textvariable=self.missile_var, state="readonly", width=28)
        self.missile_box.grid(row=2, column=1, columnspan=2, sticky="ew", pady=(8, 2))
        self.missile_box.bind("<<ComboboxSelected>>", lambda e: self._on_missile())
        self.only_carried = tk.BooleanVar(value=True)
        ttk.Checkbutton(setup, text="Only missiles on the aircraft card", variable=self.only_carried,
                        command=self._refresh_missile_list).grid(row=3, column=1, columnspan=2, sticky="w")

        self.missile_info = tk.Text(setup, width=44, height=7, wrap="word", font=("Consolas", 9),
                                    relief="flat", background=self.cget("background"))
        self.missile_info.grid(row=4, column=0, columnspan=3, sticky="ew", pady=(6, 0))
        self.missile_info.configure(state="disabled")

        self.model_status = ttk.Label(setup, text="", foreground=GREY, wraplength=320)
        self.model_status.grid(row=5, column=0, columnspan=2, sticky="w", pady=(6, 0))
        self.train_btn = ttk.Button(setup, text="Train AI model", command=self.train_selected)
        self.train_btn.grid(row=5, column=2, sticky="e", pady=(6, 0))

        # Situation -----------------------------------------------------------
        sit = ttk.LabelFrame(left, text="2. Situation", padding=8)
        sit.pack(fill="x", pady=(10, 0))
        self.speed = tk.StringVar(value="1000")
        self.speed_unit = tk.StringVar(value="km/h TAS")
        self.alt = tk.StringVar(value="8000")
        self.alt_unit = tk.StringVar(value="m")
        self.t_speed = tk.StringVar(value="")
        self.t_alt = tk.StringVar(value="")
        self.aspect = tk.StringVar(value=ASPECTS[0])
        self.exact = tk.BooleanVar(value=False)

        def row(r, label, var, unit_var=None, units=None, hint=""):
            ttk.Label(sit, text=label).grid(row=r, column=0, sticky="w", pady=2)
            ttk.Entry(sit, textvariable=var, width=10).grid(row=r, column=1, sticky="w")
            if unit_var is not None:
                ttk.Combobox(sit, textvariable=unit_var, values=units, state="readonly",
                             width=10).grid(row=r, column=2, sticky="w", padx=4)
            if hint:
                ttk.Label(sit, text=hint, foreground=GREY).grid(row=r, column=3, sticky="w")

        row(0, "Your speed", self.speed, self.speed_unit, SPEED_UNITS)
        row(1, "Your altitude", self.alt, self.alt_unit, ALT_UNITS)
        row(2, "Target speed", self.t_speed, hint="blank = same as yours")
        row(3, "Target altitude", self.t_alt, hint="blank = same as yours")
        ttk.Label(sit, text="Target aspect").grid(row=4, column=0, sticky="w", pady=2)
        ttk.Combobox(sit, textvariable=self.aspect, values=ASPECTS, width=20).grid(
            row=4, column=1, columnspan=3, sticky="w")
        ttk.Label(sit, text="(0° = coming at you, 180° = flying away; you can type a number)",
                  foreground=GREY).grid(row=5, column=0, columnspan=4, sticky="w")
        btns = ttk.Frame(sit)
        btns.grid(row=6, column=0, columnspan=4, sticky="ew", pady=(8, 0))
        self.calc_btn = ttk.Button(btns, text="Calculate", command=self.calculate)
        self.calc_btn.pack(side="left")
        ttk.Checkbutton(btns, text="Exact (use simulator, slower)", variable=self.exact).pack(side="left", padx=8)
        self.bind("<Return>", lambda e: self.calculate())

        # Result ----------------------------------------------------------------
        right = ttk.LabelFrame(root, text="3. Result", padding=8)
        right.grid(row=0, column=1, sticky="nsew")
        right.columnconfigure(0, weight=1)
        right.rowconfigure(2, weight=1)
        self.headline = ttk.Label(right, text="Choose a missile and press Calculate",
                                  font=("Segoe UI", 18, "bold"))
        self.headline.grid(row=0, column=0, sticky="w")
        self.canvas = tk.Canvas(right, height=110, background="white", highlightthickness=0)
        self.canvas.grid(row=1, column=0, sticky="ew", pady=8)
        self.canvas.bind("<Configure>", lambda e: self._draw_envelope())
        self.result_text = tk.Text(right, wrap="word", font=("Consolas", 10), height=14)
        self.result_text.grid(row=2, column=0, sticky="nsew")

        status = ttk.Frame(self, padding=(10, 0, 10, 8))
        status.pack(fill="x")
        self.progress = ttk.Progressbar(status, mode="determinate", maximum=1.0, length=220)
        self.progress.pack(side="right")
        self.status = ttk.Label(status, text="Ready")
        self.status.pack(side="left")

    # ------------------------------------------------------------------ data
    def reload_data(self):
        self.physics = lib.load_physics()
        self.missiles = lib.load_missiles()
        self.aircraft = lib.load_aircraft()
        self.aircraft_box["values"] = [NO_AIRCRAFT, *self.aircraft]
        if self.aircraft_var.get() not in self.aircraft:
            self.aircraft_var.set(next(iter(self.aircraft), NO_AIRCRAFT))
        self._refresh_missile_list()
        self.status.config(text=f"Loaded {len(self.missiles)} missiles, {len(self.aircraft)} aircraft cards")

    def _selected_aircraft(self) -> AircraftSpec | None:
        return self.aircraft.get(self.aircraft_var.get())

    def _refresh_missile_list(self):
        ac = self._selected_aircraft()
        names = list(self.missiles)
        if ac and self.only_carried.get() and ac.weapons:
            carried = [n for n in names if ac.carries(n)]
            if carried:
                names = carried
            else:
                self.status.config(text=f"No imported missile matches {ac.name}'s weapons "
                                        f"({', '.join(ac.weapons)}). Import the spreadsheet or untick the filter.")
                names = []
        self.missile_box["values"] = names
        if self.missile_var.get() not in names:
            self.missile_var.set(names[0] if names else "")
        self._on_missile()

    def _on_missile(self):
        m = self.missiles.get(self.missile_var.get())
        self.missile_info.configure(state="normal")
        self.missile_info.delete("1.0", "end")
        self.missile_info.insert("1.0", m.summary() if m else "No missile selected.")
        self.missile_info.configure(state="disabled")
        self._update_model_status()

    def _update_model_status(self):
        m = self.missiles.get(self.missile_var.get())
        if not m:
            self.model_status.config(text="")
            return
        reg = self.models.get(m.name)
        if reg is None:
            self.model_status.config(text="AI model: not trained (simulator is used, ~1-2 s per answer)",
                                     foreground=GREY)
        elif reg.meta.get("fingerprint") != fingerprint(m, self.physics):
            self.model_status.config(text="AI model: out of date for this data, retrain", foreground=ORANGE)
        else:
            mae = reg.meta.get("val_mae", [0, 0])
            self.model_status.config(
                text=f"AI model: trained {reg.meta.get('trained_at', '')} (±{mae[0]:.1f} km max, "
                     f"±{mae[1]:.1f} km no-escape)", foreground=GREEN)

    # ------------------------------------------------------------- actions
    def _run(self, label, work, done):
        """Run work(progress) in a thread, then done(result) back on the UI thread."""
        if self.busy:
            return
        self.busy = True
        self.calc_btn.state(["disabled"])
        self.train_btn.state(["disabled"])
        self.status.config(text=label)
        self.progress["value"] = 0

        def progress(text, frac):
            self.jobs.put(("progress", text, frac))

        def target():
            try:
                self.jobs.put(("done", done, work(progress)))
            except Exception as exc:  # show the error instead of crashing silently
                self.jobs.put(("error", f"{exc}\n\n{traceback.format_exc()}", None))

        threading.Thread(target=target, daemon=True).start()

    def _poll_jobs(self):
        try:
            while True:
                kind, a, b = self.jobs.get_nowait()
                if kind == "progress":
                    self.status.config(text=a)
                    self.progress["value"] = b
                    continue
                self.busy = False
                self.calc_btn.state(["!disabled"])
                self.train_btn.state(["!disabled"])
                self.progress["value"] = 0
                if kind == "done":
                    a(b)
                else:
                    self.status.config(text="Error")
                    messagebox.showerror("Error", a)
        except queue.Empty:
            pass
        self.after(100, self._poll_jobs)

    def _read_inputs(self):
        alt = to_metres(float(self.alt.get()), self.alt_unit.get())
        t_alt = to_metres(float(self.t_alt.get()), self.alt_unit.get()) if self.t_alt.get().strip() else alt
        spd = to_tas_ms(float(self.speed.get()), self.speed_unit.get(), alt)
        t_spd = (to_tas_ms(float(self.t_speed.get()), self.speed_unit.get(), t_alt)
                 if self.t_speed.get().strip() else spd)
        return spd, alt, t_spd, t_alt, parse_aspect(self.aspect.get())

    def calculate(self):
        m = self.missiles.get(self.missile_var.get())
        if not m:
            messagebox.showinfo("No missile", "Import the spreadsheet (Data menu) and pick a missile first.")
            return
        try:
            spd, alt, t_spd, t_alt, asp = self._read_inputs()
        except ValueError:
            messagebox.showerror("Input", "Speeds, altitudes and aspect must be numbers.")
            return
        ac = self._selected_aircraft()
        exact = self.exact.get()

        def work(progress):
            progress("Calculating ...", 0.3)
            return advise(m, spd, alt, t_spd, t_alt, asp, aircraft=ac, cfg=self.physics,
                          models=self.models, force_simulator=exact)

        self._run("Calculating ...", work, self._show_advice)

    def _show_advice(self, adv: Advice):
        self.last_advice = adv
        if adv.recommended_km:
            self.headline.config(text=f"Launch at {adv.recommended_km:.1f} km or closer", foreground=GREEN)
        elif adv.aggressive_km:
            self.headline.config(text=f"No escape-proof shot - max {adv.aggressive_km:.1f} km if unaware",
                                 foreground=ORANGE)
        else:
            self.headline.config(text="No valid shot in these conditions", foreground=RED)
        self.result_text.delete("1.0", "end")
        self.result_text.insert("1.0", adv.text())
        self.status.config(text=f"Done ({adv.source})")
        self._draw_envelope()

    def _draw_envelope(self):
        c = self.canvas
        c.delete("all")
        adv = self.last_advice
        if adv is None or adv.r_max_km <= 0:
            return
        w = max(c.winfo_width(), 300)
        x0, x1, y0, y1 = 20, w - 20, 30, 62
        span = max(adv.r_max_km, adv.sensor_limit_km or 0) * 1.12
        sx = lambda km: x0 + (x1 - x0) * km / span  # noqa: E731
        c.create_rectangle(x0, y0, x1, y1, fill="#f3d3d1", outline="")
        c.create_rectangle(x0, y0, sx(adv.r_max_km), y1, fill="#f8ddb0", outline="")
        c.create_rectangle(x0, y0, sx(adv.r_ne_km), y1, fill="#bfe5c8", outline="")
        c.create_rectangle(x0, y0, sx(adv.min_range_km), y1, fill=RED, outline="")
        c.create_text(x0, 12, text="0 km (you)", anchor="w", fill=GREY)
        c.create_text(x1, 12, text=f"{span:.0f} km", anchor="e", fill=GREY)
        if adv.r_ne_km > 0:
            c.create_text(sx(adv.r_ne_km), y1 + 12, text=f"no-escape {adv.r_ne_km:.1f}", fill=GREEN)
        c.create_text(sx(adv.r_max_km), y1 + 28, text=f"max {adv.r_max_km:.1f}", fill=ORANGE)
        if adv.sensor_limit_km:
            x = sx(adv.sensor_limit_km)
            c.create_line(x, y0 - 6, x, y1 + 6, dash=(4, 2), width=2, fill="#4a6fa5")
            c.create_text(x, y0 - 12, text=f"lock {adv.sensor_limit_km:.0f}", fill="#4a6fa5")
        if adv.recommended_km:
            x = sx(adv.recommended_km)
            c.create_polygon(x - 7, y0 - 2, x + 7, y0 - 2, x, y0 + 10, fill="black")
            c.create_line(x, y0, x, y1, width=2)

    def train_selected(self):
        m = self.missiles.get(self.missile_var.get())
        if not m:
            return
        if not messagebox.askyesno("Train AI model",
                                   f"Simulate about 1500 engagements with {m.name} and train the neural "
                                   "network on them? This takes about a minute."):
            return

        def work(progress):
            train_model(m, lib.MODELS_DIR, n_samples=1500, cfg=self.physics, verbose=False, progress=progress)
            return m.name

        def done(name):
            self._update_model_status()
            self.status.config(text=f"Model for {name} trained")

        self._run("Training ...", work, done)

    def import_file(self):
        path = filedialog.askopenfilename(
            title="Spreadsheet export",
            filetypes=[("Spreadsheet", "*.xlsx *.csv *.txt *.tsv"), ("All files", "*.*")])
        if not path:
            return
        try:
            specs = load_any(Path(path))
            save_specs(specs, lib.MISSILE_DIR)
        except Exception as exc:
            messagebox.showerror("Import failed", str(exc))
            return
        self.reload_data()
        messagebox.showinfo("Import", f"Imported {len(specs)} missiles.")

    def download_sheet(self):
        def work(progress):
            progress("Downloading spreadsheet ...", 0.3)
            download_workbook(lib.WORKBOOK_FILE)
            progress("Importing ...", 0.8)
            specs = load_any(lib.WORKBOOK_FILE)
            save_specs(specs, lib.MISSILE_DIR)
            return len(specs)

        def done(n):
            self.reload_data()
            messagebox.showinfo("Download", f"Imported {n} air-to-air missiles.")

        self._run("Downloading ...", work, done)

    def _edit_selected_aircraft(self):
        self.edit_aircraft(self._selected_aircraft())

    def edit_aircraft(self, ac: AircraftSpec | None):
        AircraftEditor(self, ac, sorted(self.missiles))

    def show_help(self):
        messagebox.showinfo("How to read the result", (
            "RECOMMENDED: launch at this range or closer. Even if the enemy turns away the moment you fire, "
            "the missile should still catch it (the no-escape range), as long as your sensors can lock that far.\n\n"
            "AGGRESSIVE: fine against a target that doesn't see the missile coming.\n\n"
            "MAX: the missile only just reaches a target that keeps flying straight at the same speed.\n\n"
            "Aspect: 0° = target flying straight at you, 90° = crossing, 180° = flying away.\n\n"
            "These are estimates from a simplified model of the game. Flares, chaff, notching and terrain "
            "are not modelled."))


class AircraftEditor(tk.Toplevel):
    """Small form to create or edit an aircraft stat card."""

    def __init__(self, app: App, ac: AircraftSpec | None, missile_names: list[str]):
        super().__init__(app)
        self.app = app
        self.original = ac
        self.title("Aircraft card" + (f" - {ac.name}" if ac else ""))
        self.transient(app)
        ac = ac or AircraftSpec(name="")
        r = ac.radar
        frm = ttk.Frame(self, padding=10)
        frm.pack(fill="both", expand=True)
        self.v = {
            "name": tk.StringVar(value=ac.name),
            "radar": tk.StringVar(value=r.name),
            "search": tk.StringVar(value="" if r.search_range_km is None else f"{r.search_range_km:g}"),
            "track": tk.StringVar(value="" if r.track_range_km is None else f"{r.track_range_km:g}"),
            "maxspd": tk.StringVar(value="" if ac.max_speed_kmh is None else f"{ac.max_speed_kmh:g}"),
            "ceiling": tk.StringVar(value="" if ac.service_ceiling_m is None else f"{ac.service_ceiling_m:g}"),
        }
        self.sarh = tk.BooleanVar(value=r.can_guide_sarh)
        self.arh = tk.BooleanVar(value=r.can_guide_arh)
        fields = [("Aircraft name", "name"), ("Radar name", "radar"), ("Radar search range [km]", "search"),
                  ("Radar lock / track range [km]", "track"), ("Max speed [km/h]", "maxspd"),
                  ("Service ceiling [m]", "ceiling")]
        for i, (label, key) in enumerate(fields):
            ttk.Label(frm, text=label).grid(row=i, column=0, sticky="w", pady=2)
            ttk.Entry(frm, textvariable=self.v[key], width=30).grid(row=i, column=1, sticky="ew")
        n = len(fields)
        ttk.Checkbutton(frm, text="Radar can guide SARH missiles (AIM-7 ...)", variable=self.sarh).grid(
            row=n, column=0, columnspan=2, sticky="w")
        ttk.Checkbutton(frm, text="Radar can launch ARH missiles (AIM-120 ...)", variable=self.arh).grid(
            row=n + 1, column=0, columnspan=2, sticky="w")

        ttk.Label(frm, text="Air-to-air missiles it carries\n(select imported ones and/or type names, "
                            "comma separated)").grid(row=n + 2, column=0, columnspan=2, sticky="w", pady=(8, 2))
        self.listbox = tk.Listbox(frm, selectmode="multiple", height=8, exportselection=False)
        self.listbox.grid(row=n + 3, column=0, columnspan=2, sticky="ew")
        for i, name in enumerate(missile_names):
            self.listbox.insert("end", name)
            if ac.carries(name) and ac.weapons:
                self.listbox.selection_set(i)
        extra = [w for w in ac.weapons if not any(_same(w, m) for m in missile_names)]
        self.extra = tk.StringVar(value=", ".join(extra))
        ttk.Entry(frm, textvariable=self.extra).grid(row=n + 4, column=0, columnspan=2, sticky="ew", pady=4)
        ttk.Label(frm, text="Notes").grid(row=n + 5, column=0, sticky="w")
        self.notes = tk.Text(frm, height=3, width=40)
        self.notes.insert("1.0", ac.notes)
        self.notes.grid(row=n + 6, column=0, columnspan=2, sticky="ew")
        b = ttk.Frame(frm)
        b.grid(row=n + 7, column=0, columnspan=2, sticky="e", pady=(8, 0))
        ttk.Button(b, text="Cancel", command=self.destroy).pack(side="right")
        ttk.Button(b, text="Save", command=self.save).pack(side="right", padx=6)

    def save(self):
        name = self.v["name"].get().strip()
        if not name:
            messagebox.showerror("Aircraft card", "Give the aircraft a name.", parent=self)
            return

        def num(key):
            s = self.v[key].get().strip().replace(",", ".")
            return float(s) if s else None

        try:
            weapons = [self.listbox.get(i) for i in self.listbox.curselection()]
            weapons += [w.strip() for w in self.extra.get().split(",") if w.strip()]
            ac = AircraftSpec(name=name, weapons=weapons,
                              radar=Radar(self.v["radar"].get().strip(), num("search"), num("track"),
                                          self.sarh.get(), self.arh.get()),
                              max_speed_kmh=num("maxspd"), service_ceiling_m=num("ceiling"),
                              notes=self.notes.get("1.0", "end").strip())
        except ValueError:
            messagebox.showerror("Aircraft card", "Ranges, speed and ceiling must be numbers.", parent=self)
            return
        if self.original and slug(self.original.name) != slug(name):
            old = lib.AIRCRAFT_DIR / f"{slug(self.original.name)}.json"
            if old.exists():
                old.unlink()
        ac.save(lib.AIRCRAFT_DIR)
        self.app.aircraft_var.set(name)
        self.app.reload_data()
        self.destroy()


def _same(a: str, b: str) -> bool:
    k = lambda s: "".join(c for c in s.upper() if c.isalnum())  # noqa: E731
    return k(a) == k(b)


def run():
    app = App()
    app.mainloop()
    return 0

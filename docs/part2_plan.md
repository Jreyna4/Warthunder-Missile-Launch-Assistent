# Part 2: live in-game launch cue (plan)

Goal: before a match, pick the aircraft card and missile in the GUI. During the match,
a small always-on-top window reads your speed and altitude about once per second and
shows the recommended launch range, with a sound or colour cue when the target is inside it.

> **Check the game's rules first.** Gaijin's EULA and rules limit third-party software.
> Read them before using anything live. Keep the helper *read-only*: it must never
> send keyboard/mouse input to the game or read game memory. Part 1 works offline and
> has none of these concerns.

## 1. Getting your own flight data

**Option A: the game's local telemetry page (recommended).** While a match runs, War
Thunder serves a telemetry page on your own PC at `http://localhost:8111`. Its JSON
endpoints (`/state`, `/indicators`) include airspeed (IAS/TAS), Mach and altitude.
Community dashboards use it. This is far more reliable than reading pixels.

```python
import json, urllib.request
state = json.load(urllib.request.urlopen("http://localhost:8111/state", timeout=0.5))
# Field names vary by game version; print(state.keys()) once and map them,
# e.g. TAS in km/h and altitude in metres.
```

**Option B: screen reading (OCR).** Capture the HUD region with `mss`, then read the
numbers with `pytesseract` (or a small digit-recognition network trained with
`wtmla/nn.py`). Slower and fragile when HUD layouts or resolutions change.

## 2. Target range and aspect

Your speed and altitude cover most of the answer. With no target data, the helper
assumes a co-altitude, same-speed, head-on target and shows that recommended range.
You compare it with the range shown on your radar screen. OCR of the radar range
readout can come later.

## 3. Architecture

```
telemetry.py   TelemetrySource (LocalhostSource, OcrSource) -> {tas_ms, alt_m, mach}
live.py        loop every 0.5-1 s:
                 data = source.read()
                 adv  = advise(missile, data.tas_ms, data.alt_m, ..., models=cache)  # NN: <1 ms
                 overlay.update(adv)
overlay.py     small Tk window: "-topmost", semi-transparent, big number + colour
```

The trained neural network is what makes this practical: the simulator takes about
a second per answer, while the network takes under a millisecond.

## 4. Steps

1. `telemetry.py` with `LocalhostSource`; print live values while flying a test flight.
2. Overlay window with the recommended range, updated every second.
3. Optional OCR of the radar's target range → green "IN RANGE" cue.
4. Shot logger: write situation + hit/miss to CSV, then fine-tune the network on real results.

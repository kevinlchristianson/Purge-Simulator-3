# Purge Simulator

A Python/Tkinter desktop app that models nitrogen-displacement ("purge") operations on
liquid pipelines — a pig driven by N2 gas pressure displaces the liquid product ahead of
it, station by station, down to a target exit condition.

Started as a rebuild of one reference pipeline and has since modeled 11 real client purges
across different diameters, fluids, and operating constraints without changing the core
engine.

## Running it

```
python main.py                        # GUI
python run_headless.py <scenario.json>  # no GUI; generates an xlsx report + animation GIF
```

Scenario JSONs (`scenarios/<ClientJob>/*.json`) are the full input state for a run — see
one for the shape, or build one from an ILI file, KMZ, or client Excel/CSV via the GUI's
Data Source panel.

## Project context

See [CLAUDE.md](CLAUDE.md) for the full architecture breakdown, data/privacy policy for
this repo, hard engineering rules the simulation depends on, and current open items.

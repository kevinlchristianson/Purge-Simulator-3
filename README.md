# Purge Simulator

A Python/Tkinter desktop app that models nitrogen-displacement ("purge") operations on
liquid pipelines — a pig driven by N2 gas pressure displaces the liquid product ahead of
it, station by station, down to a target exit condition.

Started as a rebuild of one reference pipeline and has since modeled 11 real client purges
across different diameters, fluids, and operating constraints without changing the core
engine.

## Running it

```
pip install -r requirements.txt
python app.py                           # standalone app in your browser, with the Claude assistant
python main.py                          # original Tkinter desktop GUI
python run_headless.py <scenario.json>  # no GUI; generates an xlsx report + animation GIF
```

### The standalone app (`app.py`)

`python app.py` starts a local server on this machine and opens the app in your default
browser: the scenario library, every input, Run, results and charts, a live pipeline
profile you can scrub through, the xlsx/log/GIF reports, and an **assistant** panel.
Nothing leaves the machine except assistant messages, which go to the Claude API.

- **API key.** Open Settings in the app and paste a Claude API key, or set
  `ANTHROPIC_API_KEY` before launching. The key is kept in `~/.purge_sim/settings.json`
  (never in the repo or the packaged app). The model defaults to `claude-opus-5-5` and can
  be changed in Settings.
- **Setting up a new job.** Import the route (or attach the KMZ in the assistant panel),
  then answer the questions on the **Job setup** tab: pipe size, product, MOP, drive cap,
  pig stop and where the liquid leaves, speeds, strategy. Blank questions use a default,
  and every default that is only a typical value is listed as an assumption to confirm and
  written into the scenario notes. A pre-run check classifies the job (speed- or
  drive-capped, friction-dominated, laminar, gravity-assisted) and sizes pack-and-coast
  before anything runs. Or just describe the job to the assistant, which fills in the same
  questions.
- **What the assistant can do.** Read and explain the open scenario, change inputs (the
  same validation as the form, and it can't break the hard engineering rules), run the
  simulation, sweep one input across several values, read results and the profile at any
  moment, and save a new scenario when asked. It never overwrites a scenario.
- **Your files.** Saved scenarios go to `~/PurgeSimScenarios` (shared with the Tkinter
  app); reports go to a new dated folder under `~/PurgeSimOutputs` on every export.

For how to use it well (the order to set up a job, what costs API credit, and what to
check before a number goes to a client), see the [user guide](docs/USER_GUIDE.md).

### Packaging it as a standalone program

```
pip install -r requirements.txt pyinstaller
pyinstaller packaging/purge_simulator.spec --noconfirm   # or packaging\build_windows.bat
```

This produces `dist/PurgeSimulator/` with `PurgeSimulator.exe` (on Windows): no Python
install needed on the machine that runs it. Build on the OS you're shipping for.

Scenario JSONs (`scenarios/<ClientJob>/*.json`) are the full input state for a run — see
one for the shape, or build one from an ILI file, KMZ, or client Excel/CSV via the GUI's
Data Source panel.

## Project context

See [CLAUDE.md](CLAUDE.md) for the full architecture breakdown, data/privacy policy for
this repo, hard engineering rules the simulation depends on, and current open items.

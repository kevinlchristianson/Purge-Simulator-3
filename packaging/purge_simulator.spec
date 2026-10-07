# PyInstaller spec for the standalone Purge Simulator app.
#
# Build from the repo root on the OS you want to ship for (PyInstaller does not
# cross-compile):
#     pip install -r requirements.txt pyinstaller
#     pyinstaller packaging/purge_simulator.spec --noconfirm
# Output: dist/PurgeSimulator/ (run PurgeSimulator.exe on Windows, PurgeSimulator elsewhere).
#
# The bundle carries the engine, the web UI and the scenarios/ library. It never
# carries an API key: the assistant reads it from Settings or ANTHROPIC_API_KEY.

import os

ROOT = os.path.abspath(os.path.join(SPECPATH, ".."))

a = Analysis(
    [os.path.join(ROOT, "app.py")],
    pathex=[ROOT],
    datas=[
        (os.path.join(ROOT, "purge_sim", "app", "static"), os.path.join("purge_sim", "app", "static")),
        (os.path.join(ROOT, "scenarios"), "scenarios"),
        (os.path.join(ROOT, "docs", "USER_GUIDE.md"), "docs"),
    ],
    hiddenimports=[
        "purge_sim.engine.formatted_report",
        "purge_sim.engine.purge_report",
        "purge_sim.engine.animation",
        "purge_sim.engine.log_export",
        "purge_sim.engine.hgl",
        "purge_sim.data.ili_parser",
        "purge_sim.data.profile_parser",
        "matplotlib.backends.backend_agg",
    ],
    excludes=["tkinter", "pytest", "IPython"],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz, a.scripts, [],
    exclude_binaries=True,
    name="PurgeSimulator",
    console=True,   # the console window shows the app URL and closes the app when closed
)
coll = COLLECT(exe, a.binaries, a.datas, name="PurgeSimulator")

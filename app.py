"""
Purge Simulator — standalone app launcher.

Starts the local app server and opens it in the default browser. The whole app
(scenario library, inputs, simulation, results, profile view, report exports
and the Claude assistant) runs on this machine; only assistant messages go out,
to the Claude API, using the key set in Settings or ANTHROPIC_API_KEY.

Usage:
    python app.py                 # pick a free port, open the browser
    python app.py --port 8765     # fixed port
    python app.py --no-browser    # just print the URL
"""

import argparse
import os
import sys
import threading
import webbrowser

# Headless matplotlib: reports and the GIF render on worker threads.
os.environ.setdefault("MPLBACKEND", "Agg")

if not getattr(sys, "frozen", False):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from purge_sim.app.server import App  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Purge Simulator standalone app")
    parser.add_argument("--port", type=int, default=0, help="port to listen on (default: any free port)")
    parser.add_argument("--no-browser", action="store_true", help="don't open a browser window")
    args = parser.parse_args()

    app = App()
    httpd = app.serve(port=args.port)
    url = app.url
    print(f"Purge Simulator is running at {url}")
    print("Close it with the Quit button in the app, or Ctrl+C here.")
    if not args.no_browser:
        threading.Timer(0.5, webbrowser.open, args=(url,)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()

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
    python app.py --phone         # also open it from a phone on the same Wi-Fi
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

PHONE_PORT = 8765   # fixed with --phone so a bookmark on the phone keeps working


def main() -> None:
    parser = argparse.ArgumentParser(description="Purge Simulator standalone app")
    parser.add_argument("--port", type=int, default=0, help="port to listen on (default: any free port)")
    parser.add_argument("--no-browser", action="store_true", help="don't open a browser window")
    parser.add_argument("--phone", action="store_true",
                        help="let a phone or tablet on the same Wi-Fi open the app (default port 8765)")
    args = parser.parse_args()

    app = App()
    httpd = app.serve(port=args.port or (PHONE_PORT if args.phone else 0), phone=args.phone)
    url = app.url
    print(f"Purge Simulator is running at {url}")
    if args.phone:
        print()
        print("Phone access is on. On a phone connected to the same Wi-Fi, open:")
        print(f"    {app.phone_url}")
        ts = app.tailscale_url
        if ts:
            print("Away from home, with Tailscale on the phone, open:")
            print(f"    {ts}")
        print("The link stays the same from one launch to the next, so you can bookmark it.")
        print("If Windows asks whether to allow Python on networks, allow Private networks.")
        print("Anyone on this Wi-Fi with the link can use the app; only turn this on on a network you trust.")
        if not ts:
            print("To use it away from home, install Tailscale on this PC and the phone, then restart the app.")
        print()
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

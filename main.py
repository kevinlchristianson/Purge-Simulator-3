"""
Purge Simulator v30 — entry point.

Usage:
    python main.py
"""

import sys
import os

# Ensure the project root is on the path when running directly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from purge_sim.ui.main_window import MainWindow


def main():
    app = MainWindow()
    app.run()


if __name__ == "__main__":
    main()

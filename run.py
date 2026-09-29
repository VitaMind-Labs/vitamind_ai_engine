"""Start Mira and Lumina together, each in its own process on its own port.

    python run.py                   # both agents, restart-on-crash
    python run.py --only lumina     # one agent
    python run.py --reload          # dev autoreload

Thin entry point over scripts/run_all.py, which holds the launcher and all its
flags. Ports come from MIRA_PORT / LUMINA_PORT (.env.example).
"""
import sys

from scripts.run_all import main

if __name__ == "__main__":
    sys.exit(main())

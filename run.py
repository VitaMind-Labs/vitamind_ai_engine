"""Start Mira, Lumina and Spark together, each in its own process on its own port.

    python run.py                   # all three agents, restart-on-crash
    python run.py --only lumina     # one agent
    python run.py --gateway         # + one public port routing /mira /lumina /spark
    python run.py --reload          # dev autoreload

Thin entry point over scripts/run_all.py, which holds the launcher and all its
flags. Ports come from MIRA_PORT / LUMINA_PORT / SPARK_PORT (.env.example).

On Render this file is the whole start command (`python run.py`): PORT is set, so
the gateway comes up on it and the three agents stay on their own internal ports.
"""
import sys

from scripts.run_all import main

if __name__ == "__main__":
    sys.exit(main())

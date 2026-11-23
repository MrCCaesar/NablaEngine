"""The benchmark harness (result schema dk-bench/1): one driver (protocol.py) that starts a fresh worker process
(worker.py) per benchmark unit in the Python environment the engine needs (benchmarks/envs/envs.toml), and adapters
(adapters/) that wrap each engine behind one interface (adapters/base.py)."""

from pathlib import Path

SCHEMA = "dk-bench/1"
REPO = Path(__file__).resolve().parents[2]

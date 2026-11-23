"""Where the default model and corpus are (standard library only: torch_ffi uses it without mujoco).

Lookup order of the fixtures directory (nablaengine.fixtures.fixtures_dir) and of the benchmark corpora's directory
(fixtures.bench_dir):
  1. $DK_FIXTURES / $DK_BENCH_DATA when set and not empty;
  2. the source checkout this package is imported from: <checkout>/fixtures/mjlab and <checkout>/benchmarks/data
     (PYTHONPATH=<checkout>/src, a .pth file or pip install -e), when they exist;
  3. the copy packaged with the wheel: nablaengine/_fixtures/mjlab (the G1 model: g1_train.mjb, g1_meta.json, its
     LICENSE and README.md) and nablaengine/_fixtures/data (the corpus mf_s0.npz), _fixtures/README.md.
When neither exists, the checkout paths are returned, so error messages name the expected location."""

from pathlib import Path

PACKAGE = Path(__file__).resolve().parent
CHECKOUT = PACKAGE.parents[1]
PACKAGED = PACKAGE / "_fixtures"
CHECKOUT_FIXTURES = CHECKOUT / "fixtures" / "mjlab"
CHECKOUT_BENCH = CHECKOUT / "benchmarks" / "data"
PACKAGED_FIXTURES = PACKAGED / "mjlab"
PACKAGED_BENCH = PACKAGED / "data"


def default_fixtures():
    """The default fixtures directory: the checkout's fixtures/mjlab, else the packaged copy (where g1_meta.json
    exists), else the checkout's path."""
    for d in (CHECKOUT_FIXTURES, PACKAGED_FIXTURES):
        if (d / "g1_meta.json").is_file():
            return d
    return CHECKOUT_FIXTURES


def default_bench():
    """The default benchmark corpora's directory: the checkout's benchmarks/data, else the packaged copy (where the
    directory exists), else the checkout's path."""
    for d in (CHECKOUT_BENCH, PACKAGED_BENCH):
        if d.is_dir():
            return d
    return CHECKOUT_BENCH

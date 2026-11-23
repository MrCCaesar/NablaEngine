"""Packages the default fixtures into the wheel (the rest of the configuration is in pyproject.toml)."""
import shutil
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py

HERE = Path(__file__).resolve().parent
PACKAGED_FIXTURES = {
    "fixtures/mjlab/g1_train.mjb": "_fixtures/mjlab/g1_train.mjb",
    "fixtures/mjlab/g1_meta.json": "_fixtures/mjlab/g1_meta.json",
    "fixtures/mjlab/LICENSE": "_fixtures/mjlab/LICENSE",
    "fixtures/mjlab/README.md": "_fixtures/mjlab/README.md",
    "benchmarks/data/mf_s0.npz": "_fixtures/data/mf_s0.npz",
}


class BuildPyWithFixtures(build_py):
    def run(self):
        super().run()
        if self.dry_run:
            return
        pkg = Path(self.build_lib) / "nablaengine"
        for src, dst in PACKAGED_FIXTURES.items():
            out = pkg / dst
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(HERE / src, out)

    def get_outputs(self, include_bytecode=True):
        pkg = Path(self.build_lib) / "nablaengine"
        return super().get_outputs(include_bytecode) + [str(pkg / dst) for dst in PACKAGED_FIXTURES.values()]


setup(cmdclass={"build_py": BuildPyWithFixtures})

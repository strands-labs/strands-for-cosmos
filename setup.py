"""Minimal setup shim: bundle the root `justfile` into the package at build time.

pyproject.toml holds all real config (PEP 621). This shim only ensures the
justfile (single source of truth at the repo root) is copied into
`strands_cosmos/` so it ships in the wheel/sdist — the tools shell out to it.
"""
import shutil
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py as _build_py


class build_py(_build_py):
    def run(self):
        root_justfile = Path(__file__).parent / "justfile"
        pkg_justfile = Path(__file__).parent / "strands_cosmos" / "justfile"
        if root_justfile.is_file():
            shutil.copyfile(root_justfile, pkg_justfile)
        super().run()


setup(cmdclass={"build_py": build_py})

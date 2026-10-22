"""Modelling operations of the studio (web/modeller.js), checked in Node.

Floor loads must reproduce the generator's tributary loads exactly, splitting
members must not change any nodal result, a translational repeat must build the
same frame as the generator, and node/member/support/load-case edits behave.
"""
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
NODE = shutil.which("node")


@pytest.mark.skipif(NODE is None, reason="node.js not available")
def test_modeller_operations():
    out = subprocess.run([NODE, str(ROOT / "tests" / "js" / "modeller_test.js")], capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr + out.stdout
    assert "modeller ok" in out.stdout

"""The solver agrees with two independent programs on random frames.

PyNite frames include hinges, partial and trapezoidal loads; OpenSees frames
use full-length and point loads only.
"""
import pytest

SEEDS = list(range(12))


@pytest.mark.parametrize("seed", SEEDS)
def test_against_pynite(seed):
    pytest.importorskip("Pynite")
    from tragwerk.verification.crosscheck import compare_pynite
    r = compare_pynite(seed)
    assert r["displacement"] < 1e-9
    assert r["reaction"] < 1e-9
    assert r["moment"] < 1e-6


@pytest.mark.parametrize("seed", SEEDS)
def test_against_opensees(seed):
    pytest.importorskip("openseespy.opensees")
    from tragwerk.verification.crosscheck import compare_opensees
    r = compare_opensees(seed)
    assert r["displacement"] < 1e-9
    assert r["reaction"] < 1e-9

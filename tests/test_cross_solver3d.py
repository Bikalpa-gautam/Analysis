"""The 3D solver agrees with PyNite (square sections, hinges, partial loads) and
OpenSees (rectangular sections with rolled members, so local axes must match)."""
import pytest

SEEDS = list(range(8))


@pytest.mark.parametrize("seed", SEEDS)
def test_against_pynite3d(seed):
    pytest.importorskip("Pynite")
    from tragwerk.verification.crosscheck3d import compare_pynite3d
    r = compare_pynite3d(seed)
    assert r["displacement"] < 1e-9 and r["reaction"] < 1e-9


@pytest.mark.parametrize("seed", SEEDS)
def test_against_opensees3d(seed):
    pytest.importorskip("openseespy.opensees")
    from tragwerk.verification.crosscheck3d import compare_opensees3d
    r = compare_opensees3d(seed)
    assert r["displacement"] < 1e-9 and r["reaction"] < 1e-9

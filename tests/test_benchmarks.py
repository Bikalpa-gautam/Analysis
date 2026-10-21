"""Every published verification benchmark must pass."""
import pytest

from tragwerk.verification.benchmarks import BENCHMARKS


@pytest.mark.parametrize("bm", BENCHMARKS, ids=[b.id for b in BENCHMARKS])
def test_benchmark(bm):
    comps = bm.run()
    assert comps, "benchmark produced no comparisons"
    failed = [c for c in comps if not c.passed]
    assert not failed, "; ".join(f"{c.quantity}: {c.computed} vs {c.expected}" for c in failed)

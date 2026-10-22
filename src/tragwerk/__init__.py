"""Tragwerk: verifiable structural analysis and Eurocode 2 checks for RC frames.

    import tragwerk as tw
    model = tw.load_model("examples/two_span_beam.json")
    result = tw.run(model)
    print(result.overall_status(), result.governing().utilisation)
    html = tw.report(result)
"""
__version__ = "0.3.0"

from .model import Model, ModelError, load_model, parse_model  # noqa: E402,F401


def run(model):
    from .project import run as _run
    return _run(model)


def report(result, verify: bool = True, **info) -> str:
    from .report.html import render
    from .verification.benchmarks import run_all
    return render(result, verification=run_all() if verify else None, project_info=info)

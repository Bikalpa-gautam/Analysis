"""SIA 262:2013 checks: not implemented yet.

The interface matches ``tragwerk.codes.ec2.checks.check_member``. The rules
must be written against the standard text by an engineer who holds it;
until then every member reports "not covered" so no result is silently
taken for a SIA verification.
"""
from __future__ import annotations

from ..base import NOT_COVERED, CheckResult


def check_member(member, sec, rc, forces, params) -> list[CheckResult]:
    if member.design.type == "none":
        return []
    return [CheckResult(
        id="sia262.pending", title="SIA 262 checks", clause="SIA 262:2013", member=member.id,
        status=NOT_COVERED,
        notes=["SIA 262 rules are not implemented in this version. Run the model with "
               "design_code 'EC2' or verify the member by another method."],
    )]

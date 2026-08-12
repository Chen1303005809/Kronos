"""V5 target-only, direction-guided path research workflow.

V5 deliberately starts with the target-contract-only audit, zero-shot path
bank, and frozen direction probe.  The learned path phases stay unavailable
until their preceding persisted gates have been reviewed.
"""

from csj.v5.config import PRODUCTION_ELIGIBLE, RESULT_SCOPE

__all__ = ["PRODUCTION_ELIGIBLE", "RESULT_SCOPE"]

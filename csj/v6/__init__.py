"""V6 target-only active-risk-control research workflow.

P0 risk labels, leakage checks, support gates, and visual evidence are
implemented. Later stages remain unavailable while the persisted P0 gate is
failed.
"""

STRATEGY_VERSION = 6
RESULT_SCOPE = "retrospective_observed_contracts"
PRODUCTION_ELIGIBLE = False

__all__ = ["PRODUCTION_ELIGIBLE", "RESULT_SCOPE", "STRATEGY_VERSION"]

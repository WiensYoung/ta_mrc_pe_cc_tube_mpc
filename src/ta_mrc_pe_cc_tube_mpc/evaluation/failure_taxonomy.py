"""Failure taxonomy: structured failure classification and reporting.

Failure types:
    F1: Collision
    F2: Near-miss
    F3: Rule violation
    F4: Channel boundary violation
    F5: Bank clearance violation
    F6: Inter-ship clearance violation
    F7: MPC infeasibility
    F8: CBF infeasible / over-intervention
    F9: Fallback failure
    F10: Runtime deadline miss
"""

from ..constants import FAILURE_CODES


def build_failure_taxonomy_table(results: list) -> dict:
    """Build a structured failure taxonomy summary from episode results.

    Args:
        results: List of EpisodeResult objects.

    Returns:
        dict suitable for DataFrame conversion or report table.
    """
    rows = []
    for result in results:
        row = {
            "episode_id": result.episode_id,
            "scenario_id": result.scenario_id,
            "method": result.method,
            "seed": result.seed,
        }
        for code in FAILURE_CODES:
            row[code] = result.failure_flags.get(code, False)
        row["primary_failure"] = result.failure_type or "None"
        rows.append(row)
    return {"rows": rows, "n_total": len(results)}


def compute_failure_rates_by_scenario(results: list) -> dict:
    """Compute failure rates broken down by scenario and method.

    Args:
        results: List of EpisodeResult objects.

    Returns:
        Nested dict: scenario -> method -> failure code -> rate.
    """
    from collections import defaultdict

    counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    totals = defaultdict(lambda: defaultdict(int))

    for r in results:
        key = (r.scenario_id, r.method)
        totals[r.scenario_id][r.method] += 1
        for code in FAILURE_CODES:
            if r.failure_flags.get(code, False):
                counts[r.scenario_id][r.method][code] += 1

    rates = {}
    # Iterate totals (includes ALL methods, even those with zero failures)
    for scenario, methods in totals.items():
        rates[scenario] = {}
        for method, n in methods.items():
            failures = counts.get(scenario, {}).get(method, {})
            rates[scenario][method] = {
                code: failures.get(code, 0) / max(n, 1)
                for code in FAILURE_CODES
            }
    return rates


def compute_failure_rates_by_method(results: list) -> dict:
    """Compute overall failure rates per method across all scenarios.

    Returns:
        dict: method -> {failure_code: rate, ...}
    """
    from collections import defaultdict

    counts = defaultdict(lambda: defaultdict(int))
    totals = defaultdict(int)

    for r in results:
        totals[r.method] += 1
        for code in FAILURE_CODES:
            if r.failure_flags.get(code, False):
                counts[r.method][code] += 1

    # Iterate totals (includes all methods, even those with zero failures)
    return {
        method: {code: counts.get(method, {}).get(code, 0) / max(totals[method], 1)
                 for code in FAILURE_CODES}
        for method in totals
    }

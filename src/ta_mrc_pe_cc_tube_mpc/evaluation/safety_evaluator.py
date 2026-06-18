"""Unified safety evaluator — single source of truth for safety metrics.

All methods (Proposed, B1-B7, B0, B8, B10) must use this evaluator
to compute paper-table safety metrics. No method-specific metric
computation is permitted.

Design principles:
1. Independent of controller metadata — works from raw state history.
2. Returns finite values for all methods — no silent NaN.
3. Uses oriented-box collision detection via geometry.py.
4. Separates collision, near-miss, and safety-domain violation clearly.
5. Respects missing-context policy: record reason, use conservative default.
"""

from dataclasses import dataclass, field
from typing import Optional

import logging
import numpy as np

from .geometry import centroid_distance, hulls_overlap_oriented_rect

logger = logging.getLogger(__name__)


@dataclass
class SafetyEvaluationBreakdown:
    """Decomposed safety distance for one (ownship, target) at one timestep."""

    base_domain: float = 0.0
    dynamic_domain_margin: float = 0.0
    rule_margin: float = 0.0
    chance_margin: float = 0.0
    tube_margin: float = 0.0
    environment_margin: float = 0.0
    total_required_distance: float = 0.0

    missing_context_reasons: list[str] = field(default_factory=list)


@dataclass
class EpisodeSafetyMetrics:
    """All safety metrics for one episode, computed uniformly.

    Every field must be finite (no NaN, no inf in final output).
    """

    collision: bool = False
    grounding: bool = False
    rule_violation: bool = False
    near_miss_step_count: int = 0
    near_miss_event_count: int = 0
    episode_has_near_miss: bool = False

    min_distance: float = float("inf")
    min_required_safety_distance: float = float("inf")
    mean_required_safety_distance: float = 0.0
    min_safety_margin: float = float("inf")
    mean_safety_margin: float = 0.0
    p5_safety_margin: float = float("inf")
    safety_margin_violation_rate: float = 0.0
    min_normalized_safety_margin: float = float("inf")

    fallback_step_count: int = 0
    episode_has_fallback: bool = False
    fallback_recovery_success: bool = True

    chance_constraint_violations: int = 0
    chance_constraint_violation_rate: float = 0.0
    cbf_intervention_count: int = 0
    cbf_infeasible_count: int = 0

    mean_runtime: float = 0.0
    p95_runtime: float = 0.0
    p99_runtime: float = 0.0
    max_runtime: float = 0.0
    deadline_miss_count: int = 0
    deadline_miss_rate: float = 0.0

    requested_backend: str = "unknown"
    actual_backend: str = "unknown"
    actual_solver: str = "unknown"
    backend_degraded: bool = False
    fallback_reason: str = ""

    scenario_type: str = "synthetic"
    data_source: str = "procedural"

    nonfinite_force_count: int = 0
    nonfinite_state_count: int = 0
    nonfinite_control_count: int = 0
    nonfinite_solver_output_count: int = 0

    # Environmental metrics (populated even for single-ship scenarios)
    bank_clearance_violation: bool = False
    bank_safety_margin: float = 0.0
    ukc_violation: bool = False
    channel_violation: bool = False
    channel_boundary_violation: bool = False

    missing_context_reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to flat dict for CSV output."""
        return {
            "collision": self.collision,
            "grounding": self.grounding,
            "rule_violation": self.rule_violation,
            "near_miss_step_count": self.near_miss_step_count,
            "near_miss_event_count": self.near_miss_event_count,
            "episode_has_near_miss": self.episode_has_near_miss,
            "min_distance": self._safe_float(self.min_distance),
            "min_required_safety_distance": self._safe_float(self.min_required_safety_distance),
            "mean_required_safety_distance": self._safe_float(self.mean_required_safety_distance),
            "min_safety_margin": self._safe_float(self.min_safety_margin),
            "mean_safety_margin": self._safe_float(self.mean_safety_margin),
            "p5_safety_margin": self._safe_float(self.p5_safety_margin),
            "safety_margin_violation_rate": self._safe_float(self.safety_margin_violation_rate),
            "min_normalized_safety_margin": self._safe_float(self.min_normalized_safety_margin),
            "fallback_step_count": self.fallback_step_count,
            "episode_has_fallback": self.episode_has_fallback,
            "fallback_recovery_success": self.fallback_recovery_success,
            "chance_constraint_violations": self.chance_constraint_violations,
            "chance_constraint_violation_rate": self._safe_float(self.chance_constraint_violation_rate),
            "cbf_intervention_count": self.cbf_intervention_count,
            "cbf_infeasible_count": self.cbf_infeasible_count,
            "mean_runtime": self._safe_float(self.mean_runtime),
            "p95_runtime": self._safe_float(self.p95_runtime),
            "p99_runtime": self._safe_float(self.p99_runtime),
            "max_runtime": self._safe_float(self.max_runtime),
            "deadline_miss_count": self.deadline_miss_count,
            "deadline_miss_rate": self._safe_float(self.deadline_miss_rate),
            "requested_backend": self.requested_backend,
            "actual_backend": self.actual_backend,
            "actual_solver": self.actual_solver,
            "backend_degraded": self.backend_degraded,
            "fallback_reason": self.fallback_reason,
            "scenario_type": self.scenario_type,
            "data_source": self.data_source,
            "nonfinite_force_count": self.nonfinite_force_count,
            "nonfinite_state_count": self.nonfinite_state_count,
            "nonfinite_control_count": self.nonfinite_control_count,
            "nonfinite_solver_output_count": self.nonfinite_solver_output_count,
            "bank_clearance_violation": self.bank_clearance_violation,
            "bank_safety_margin": self._safe_float(self.bank_safety_margin),
            "ukc_violation": self.ukc_violation,
            "channel_violation": self.channel_violation,
            "channel_boundary_violation": self.channel_boundary_violation,
        }

    @staticmethod
    def _safe_float(value, default=None, allow_nan=True):
        """Return finite float or None for NaN/Inf (preserving data quality signal).

        When *allow_nan* is True, NaN/Inf are returned as None (JSON null),
        which is detectable downstream rather than silently zeroed.
        When False, NaN/Inf raise ValueError (strict mode).

        This matches the corrected behaviour in ``closed_loop_runner._safe_float``
        — NaN/Inf are NEVER silently converted to 0.0, which was a statistical
        error that distorted safety metrics and hypothesis tests.
        """
        if value is None:
            return default
        try:
            v = float(value)
            if np.isnan(v):
                if allow_nan:
                    return None
                raise ValueError("NaN in metric value with allow_nan=False")
            if np.isinf(v):
                if allow_nan:
                    return None
                raise ValueError("Inf in metric value with allow_nan=False")
            return v
        except (TypeError, ValueError):
            return default


def evaluate_episode_safety(
    own_states: list,
    target_histories: list[list],
    own_params,  # VesselParams
    target_params_list: list,  # list[TargetShip]
    command_history: list,
    metadata: dict,
    env_sequence: list = None,
    *,
    near_miss_padding: float = 50.0,
    near_miss_event_gap: int = 5,
    deadline_s: float = 0.5,
) -> EpisodeSafetyMetrics:
    """Unified safety evaluation for one episode.

    Computes ALL safety metrics from raw state/command history and
    metadata. Independent of controller internals.

    Args:
        own_states: List of ownship VesselState over time.
        target_histories: List of lists of target VesselState over time.
        own_params: Ownship VesselParams.
        target_params_list: List of target TargetShip objects (for dimensions).
        command_history: List of ControlCommand over time.
        metadata: Episode metadata dict from controller.
        env_sequence: Optional list of EnvironmentState over time.
        near_miss_padding: Safety padding [m] for near-miss detection.
        near_miss_event_gap: Max step gap to merge near-miss steps into one event.
        deadline_s: Runtime deadline [s] for timing checks.

    Returns:
        EpisodeSafetyMetrics with all fields populated.
    """
    m = EpisodeSafetyMetrics()
    n_steps = len(own_states)
    n_targets = len(target_params_list)

    # ── Populate metadata-derived fields (ALWAYS — even for single-ship) ─
    _populate_metadata_fields(m, metadata, command_history, deadline_s, n_steps)
    _populate_scenario_info(m, metadata)

    # No states → nothing to evaluate
    if n_steps == 0:
        return m

    # Single-ship scenario (no collision-avoidance targets): still compute
    # environmental metrics (bank, UKC, channel) but skip target-dependent
    # collision / domain / near-miss evaluation.
    if n_targets == 0:
        # Compute environmental metrics from state history
        _compute_single_ship_environmental(m, own_states, own_params, env_sequence)
        _ensure_finite(m)
        return m

    # ── Per-step safety evaluation ────────────────────────────────────
    all_distances = []
    all_required = []
    all_margins = []
    near_miss_steps = []

    for k in range(n_steps):
        own = own_states[k]
        for i in range(n_targets):
            if i >= len(target_histories):
                continue
            tgt_hist = target_histories[i]
            if k >= len(tgt_hist):
                continue
            tgt = tgt_hist[k]
            tgt_params = target_params_list[i] if i < len(target_params_list) else None

            # Distance
            d = centroid_distance(own.x, own.y, tgt.x, tgt.y)
            all_distances.append(d)

            # Oriented collision check (P0: use hull geometry, not centroid distance)
            if tgt_params is not None:
                tgt_L = getattr(tgt_params, "length", 100.0)
                tgt_B = getattr(tgt_params, "beam", 15.0)
            else:
                tgt_L, tgt_B = 100.0, 15.0

            if hulls_overlap_oriented_rect(
                own.x, own.y, own.psi, own_params.length, own_params.beam,
                tgt.x, tgt.y, tgt.psi, tgt_L, tgt_B,
                safety_padding=0.0,
            ):
                m.collision = True

            # Required safety distance from metadata breakdown
            required = _get_required_safety_distance(
                metadata, i, k, own_params, tgt_params, m.missing_context_reasons, n_targets=n_targets,
            )
            all_required.append(required)

            margin = d - required
            all_margins.append(margin)

            if margin < 0:
                m.safety_margin_violation_rate += 1

            # Near-miss: using padded hull overlap
            if hulls_overlap_oriented_rect(
                own.x, own.y, own.psi, own_params.length, own_params.beam,
                tgt.x, tgt.y, tgt.psi, tgt_L, tgt_B,
                safety_padding=near_miss_padding,
            ):
                near_miss_steps.append(k)

    # ── Compute statistics ────────────────────────────────────────────
    n_checks = max(len(all_distances), 1)
    m.min_distance = float(np.min(all_distances)) if all_distances else 0.0
    m.min_required_safety_distance = float(np.min(all_required)) if all_required else 0.0
    m.mean_required_safety_distance = float(np.mean(all_required)) if all_required else 0.0

    if all_margins:
        margins_arr = np.array(all_margins)
        m.min_safety_margin = float(np.min(margins_arr))
        m.mean_safety_margin = float(np.mean(margins_arr))
        m.p5_safety_margin = float(np.percentile(margins_arr, 5))
    else:
        m.min_safety_margin = 0.0
        m.mean_safety_margin = 0.0
        m.p5_safety_margin = 0.0

    m.safety_margin_violation_rate = m.safety_margin_violation_rate / n_checks

    if all_required and all_required[0] > 0:
        m.min_normalized_safety_margin = m.min_safety_margin / max(all_required[0], 0.1)

    # Near-miss events
    m.near_miss_step_count = len(near_miss_steps)
    near_miss_events = _merge_consecutive_to_events(sorted(set(near_miss_steps)), gap=near_miss_event_gap)
    m.near_miss_event_count = len(near_miss_events)
    m.episode_has_near_miss = len(near_miss_events) > 0

    # ── Environmental metrics (bank, UKC, channel) ───────────────────
    # These MUST be computed for ALL episodes (including multi-target),
    # not just single-ship episodes.  The old code only called this for
    # n_targets == 0, causing the safety evaluator's default False values
    # to overwrite the legacy metrics' correct values for multi-target
    # episodes.  This made F4/F5 failure detection impossible for any
    # encounter scenario.
    _compute_single_ship_environmental(m, own_states, own_params, env_sequence)

    # ── Finalize: ensure all fields are finite ───────────────────────
    _ensure_finite(m)
    return m


def _populate_metadata_fields(m: EpisodeSafetyMetrics, metadata: dict, command_history, deadline_s, n_steps):
    """Extract fields from controller metadata.

    Uses EPISODE-LEVEL accumulated counters (fallback_step_total,
    cbf_intervention_total, etc.) set by the controller's
    reset_episode_counters / per-step accumulation, NOT the last-step
    cbf_info or fallback_step_count.
    """
    # Fallback — use accumulated episode-level totals
    m.fallback_step_count = metadata.get("fallback_step_total", 0)
    m.episode_has_fallback = metadata.get("episode_has_fallback", False)
    m.fallback_recovery_success = metadata.get("fallback_recovery_success", True)
    m.fallback_reason = str(metadata.get("fallback_reason", ""))

    # Chance constraint
    cc_results = metadata.get("chance_constraint_results", [])
    m.chance_constraint_violations = sum(
        1 for r in cc_results if not r.get("constraint_satisfied", True)
    )
    n_cc = max(len(cc_results), 1)
    m.chance_constraint_violation_rate = m.chance_constraint_violations / n_cc

    # CBF — use accumulated episode-level totals (not last-step cbf_info)
    m.cbf_intervention_count = metadata.get("cbf_intervention_total", 0)
    m.cbf_infeasible_count = metadata.get("cbf_infeasible_total", 0)

    # Runtime — compute from timing breakdown if available
    timing = metadata.get("timing_breakdown", {})
    if timing and isinstance(timing, dict):
        # timing_breakdown is per-step; compute aggregate for episode
        # The controller stores the last-step timing breakdown in metadata
        # For episode-level stats, compute from command_history
        pass

    # Compute runtime statistics from per-step command metadata
    runtimes = []
    if command_history:
        for c in command_history:
            rt = (c.metadata or {}).get("runtime", 0.0) if hasattr(c, "metadata") else 0.0
            if isinstance(rt, (int, float)) and rt > 0:
                runtimes.append(float(rt))
                if rt > deadline_s:
                    m.deadline_miss_count += 1
    if runtimes:
        import numpy as np
        m.mean_runtime = float(np.mean(runtimes))
        m.p95_runtime = float(np.percentile(runtimes, 95))
        m.p99_runtime = float(np.percentile(runtimes, 99))
        m.max_runtime = float(np.max(runtimes))
    else:
        # Fallback: use metadata (for backwards compatibility)
        m.mean_runtime = metadata.get("mean_runtime", 0.0)
        m.p95_runtime = metadata.get("p95_runtime", 0.0)
        m.p99_runtime = metadata.get("p99_runtime", 0.0)
        m.max_runtime = metadata.get("max_runtime", 0.0)
    # BUG #4 fix (2026-06-10): denominator should be len(runtimes),
    # not n_steps.  Steps without timing data should not be counted as
    # "met the deadline" — they have no data either way.
    n_with_runtime = max(len(runtimes), 1)
    m.deadline_miss_rate = m.deadline_miss_count / n_with_runtime

    # Backend info
    mpc_info = metadata.get("mpc_info", {})
    m.requested_backend = str(mpc_info.get("requested_backend", "unknown"))
    m.actual_backend = str(mpc_info.get("actual_backend", "unknown"))
    m.actual_solver = str(mpc_info.get("actual_solver", "unknown"))
    m.backend_degraded = mpc_info.get("degraded_backend", False) or (m.requested_backend != m.actual_backend)

    # Non-finite diagnostics
    m.nonfinite_force_count = metadata.get("nonfinite_force_count", 0)
    m.nonfinite_state_count = metadata.get("nonfinite_state_count", 0)
    m.nonfinite_control_count = metadata.get("nonfinite_control_count", 0)
    m.nonfinite_solver_output_count = metadata.get("nonfinite_solver_output_count", 0)


def _populate_scenario_info(m: EpisodeSafetyMetrics, metadata: dict):
    """Extract scenario type and data source."""
    m.scenario_type = str(metadata.get("scenario_type", "synthetic"))
    m.data_source = str(metadata.get("data_source", "procedural"))


def _get_required_safety_distance(
    metadata: dict,
    target_idx: int,
    step_idx: int,
    own_params,
    target_params,
    missing_reasons: list,
    n_targets: int = 1,
) -> float:
    """Get required safety distance for one (ownship, target) at one step.

    Priority:
    1. Per-step domain_results: domain_results[step_idx * n_targets + target_idx].
    2. safety_distance_breakdown from metadata (controller-computed, per-target only).
    3. d_safe_eff_list from metadata (legacy, per-target only).
    4. Conservative default: 4.0 * (own_length + target_length).
    """
    # Try per-step domain_results first (time-varying safety distances from controller)
    domain_results = metadata.get("domain_results", [])
    if domain_results and isinstance(domain_results, list):
        flat_idx = step_idx * n_targets + target_idx
        if flat_idx < len(domain_results):
            dr = domain_results[flat_idx]
            if isinstance(dr, dict):
                total = dr.get("total")
                if total is not None:
                    return float(total)

    # Try safety_distance_breakdown (per-target, same for all steps)
    breakdowns = metadata.get("safety_distance_breakdown", [])
    if breakdowns and target_idx < len(breakdowns):
        bd = breakdowns[target_idx]
        if isinstance(bd, dict):
            return float(bd.get("total", 500.0))

    # Try legacy d_safe_eff_list
    d_safe_list = metadata.get("d_safe_eff_list", [])
    if d_safe_list and target_idx < len(d_safe_list):
        return float(d_safe_list[target_idx])

    # Conservative default
    missing_reasons.append(
        f"No safety distance breakdown for target={target_idx}; using conservative default."
    )
    own_L = own_params.length if own_params else 180.0
    tgt_L = target_params.length if target_params else 100.0
    return 4.0 * (own_L + tgt_L)


# ---------------------------------------------------------------------------
# Shared helpers for domain_results flat indexing (L1 fix — 2026-06-10)
# ---------------------------------------------------------------------------
# domain_results is a flat list ordered as:
#   [target_0_step_0, target_1_step_0, ..., target_N_step_0,
#    target_0_step_1, target_1_step_1, ..., target_N_step_1,
#    ...]
# The flat index for (step_idx=k, target_idx=i, n_targets=N) is:
#     flat_idx = k * N + i
#
# These functions are the SINGLE SOURCE OF TRUTH for domain_results indexing.
# All modules that consume domain_results MUST use these helpers to avoid
# off-by-one and stride errors.

def domain_results_flat_index(step_idx: int, target_idx: int, n_targets: int) -> int:
    """Compute flat index into domain_results list.

    Args:
        step_idx: Time step index (0-based).
        target_idx: Target ship index (0-based).
        n_targets: Total number of target ships in the episode.

    Returns:
        Flat index into the domain_results list.
    """
    return step_idx * n_targets + target_idx


def get_domain_result(domain_results: list, step_idx: int, target_idx: int,
                      n_targets: int) -> dict:
    """Safely extract a domain result dict from the flat list.

    Returns an empty dict if the index is out of bounds.
    """
    idx = domain_results_flat_index(step_idx, target_idx, n_targets)
    if idx < len(domain_results):
        dr = domain_results[idx]
        return dr if isinstance(dr, dict) else {}
    return {}
def _merge_consecutive_to_events(step_indices: list[int], gap: int = 5) -> list[list[int]]:
    """Merge consecutive step indices into events."""
    if not step_indices:
        return []
    events = []
    current = [step_indices[0]]
    for idx in step_indices[1:]:
        if idx - current[-1] <= gap:
            current.append(idx)
        else:
            events.append(current)
            current = [idx]
    events.append(current)
    return events


def _compute_single_ship_environmental(
    m: EpisodeSafetyMetrics,
    own_states: list,
    own_params,
    env_sequence: list = None,
):
    """Compute environmental metrics from state history and environment data.

    Populates bank clearance, UKC, and channel boundary metrics.
    Called for ALL episodes (single-ship and multi-target) to ensure
    the safety evaluator does not overwrite legacy metrics with defaults.
    """
    if not env_sequence:
        return

    bank_clearance_violation = False
    bank_safety_margin = float("inf")
    ukc_violation = False
    channel_violation = False
    min_ukc = float("inf")
    min_bank = float("inf")

    for k, state in enumerate(own_states):
        env = env_sequence[min(k, len(env_sequence) - 1)]

        # Bank clearance
        bl = env.bank_distance_left
        br = env.bank_distance_right
        if bl is not None and bl < min_bank:
            min_bank = min(min_bank, bl)
        if br is not None and br < min_bank:
            min_bank = min(min_bank, br)
        # Threshold: 0.5 × L (consistent with failure_detector.py and metrics.py).
        # Old value was 2.0 × L which was 4x too conservative.
        if min_bank < 0.5 * own_params.length:
            bank_clearance_violation = True

        # UKC
        if env.water_depth is not None:
            ukc = env.water_depth - own_params.draught
            if ukc < min_ukc:
                min_ukc = ukc
            if ukc < 1.0:
                ukc_violation = True

        # Channel boundary
        if env.channel_width is not None:
            half_w = env.channel_width / 2
            center_y = getattr(env, "channel_center_y", 0.0)
            if abs(state.y - center_y) > half_w:
                channel_violation = True

    m.bank_clearance_violation = bank_clearance_violation
    m.bank_safety_margin = min_bank if min_bank != float("inf") else 0.0
    m.ukc_violation = ukc_violation
    m.channel_violation = channel_violation
    m.channel_boundary_violation = channel_violation


def _ensure_finite(m: EpisodeSafetyMetrics):
    """Replace any remaining NaN/Inf with None, logging each replacement.

    BUG #7 fix (2026-06-10): Previously set NaN/Inf to 0.0, which silently
    erased data-quality signals before ``_safe_float`` in ``to_dict()`` could
    convert them to ``None`` (JSON null).  Now we set ``None`` directly so the
    downstream CSV / JSON output preserves the "unknown / not computed" signal.
    """
    for field_name in [
        "min_distance", "min_required_safety_distance", "mean_required_safety_distance",
        "min_safety_margin", "mean_safety_margin", "p5_safety_margin",
        "safety_margin_violation_rate", "min_normalized_safety_margin",
        "mean_runtime", "p95_runtime", "p99_runtime", "max_runtime",
        "deadline_miss_rate", "chance_constraint_violation_rate",
    ]:
        val = getattr(m, field_name)
        if val is None or (isinstance(val, float) and (np.isnan(val) or np.isinf(val))):
            logger.warning(
                "Safety metric '%s' has non-finite value %s — replacing with None. "
                "This may indicate missing data or a computation failure.",
                field_name, val,
            )
            setattr(m, field_name, None)

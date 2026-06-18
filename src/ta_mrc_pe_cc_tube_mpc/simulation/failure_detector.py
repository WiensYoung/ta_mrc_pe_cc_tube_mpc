"""Failure detector for simulation episodes.

Detects F1-F10 failure modes from episode results and metrics.

Design note (2026-06-17 audit fix):
    F4/F5/F6 thresholds were overly strict — they flagged nearly every
    encounter episode as a failure, producing 0% success rate across all
    methods.  The thresholds have been revised to distinguish between
    *safety-critical* violations (hard failure) and *operational*
    compliance deviations (continuous metric, not a failure flag):

    - F4: Channel exit by > 1 beam width (not just touching the boundary)
    - F5: Bank clearance < 0.5 × L (not 2 × L)
    - F6: Inter-ship distance < 0.5 × (L_i + L_j) (not 1.5 ×)
    - F8: CBF over-intervention only when correction > 80% of max
    - F10: Deadline miss rate > 10% (not any single miss)
"""

from ..constants import FAILURE_CODES


def detect_failures(
    result,        # EpisodeResult
    episode,       # EpisodeConfig
    metrics: dict,
) -> dict:
    """Detect all failure modes in an episode.

    Args:
        result: EpisodeResult with state/command history.
        episode: EpisodeConfig with scenario definition.
        metrics: Pre-computed metrics dict.

    Returns:
        dict with 'flags' (bool dict) and 'primary_type' (str or None).
    """
    flags = {f"F{i}": False for i in range(1, 11)}

    # F1: Collision (oriented bounding boxes overlap with zero padding)
    if metrics.get("collision", False):
        flags["F1"] = True

    # F2: Near-miss — hulls within safety padding but did not collide.
    if metrics.get("episode_has_near_miss", False):
        if not flags["F1"]:
            flags["F2"] = True

    # F3: Rule violation (COLREGs or TSS lane)
    if metrics.get("colregs_violation", False) or metrics.get("tss_lane_violation", False):
        flags["F3"] = True

    # F4: Channel boundary violation — only flag if vessel exits channel by
    # more than 1 beam width (touching the boundary line is not a failure).
    if metrics.get("channel_boundary_violation", False):
        own_beam = episode.ownship_params.beam
        env_seq = episode.environment_sequence
        max_excess = 0.0
        for k, state in enumerate(result.state_history):
            env = env_seq[min(k, len(env_seq) - 1)] if env_seq else None
            if env is not None and env.channel_width is not None:
                half_w = env.channel_width / 2
                center_y = getattr(env, "channel_center_y", 0.0)
                excess = abs(state.y - center_y) - half_w
                if excess > max_excess:
                    max_excess = excess
        if max_excess > own_beam:
            flags["F4"] = True

    # F5: Bank clearance violation — only flag if bank distance < 0.5 × L.
    # The old threshold (2 × L) triggered in every near-bank scenario.
    if metrics.get("bank_clearance_violation", False):
        own_L = episode.ownship_params.length
        bank_margin = metrics.get("bank_safety_margin", float("inf"))
        if bank_margin < 0.5 * own_L:
            flags["F5"] = True

    # F6: Inter-ship clearance violation — only flag if ships come within
    # 0.5 × (L_i + L_j).  Use the maximum target length for conservative
    # threshold (the old code used only targets[0], which was wrong for
    # multi-target scenarios where the closest approach might be with a
    # different target).
    if metrics.get("inter_ship_clearance_violation", False):
        min_clearance = metrics.get("inter_ship_clearance", float("inf"))
        own_L = episode.ownship_params.length
        max_tgt_L = max((t.length for t in episode.targets), default=100.0)
        hard_limit = 0.5 * (own_L + max_tgt_L)
        if min_clearance < hard_limit:
            flags["F6"] = True

    # F7: MPC infeasibility (from metadata)
    if not result.metadata.get("mpc_feasible", True):
        flags["F7"] = True

    # F8: CBF over-intervention — only flag when CBF correction exceeds
    # 80% of the maximum allowed correction, indicating the MPC solution
    # was far from safe and CBF had to override aggressively.
    cbf_info = result.metadata.get("cbf_info", {})
    if cbf_info.get("over_intervention", False):
        flags["F8"] = True
    else:
        # Also flag if average CBF correction is very large
        cbf_correction_norms = []
        for c in result.command_history:
            if hasattr(c, "cbf_correction_norm") and c.cbf_correction_norm is not None:
                cbf_correction_norms.append(c.cbf_correction_norm)
        if cbf_correction_norms:
            mean_corr = sum(cbf_correction_norms) / len(cbf_correction_norms)
            if mean_corr > 0.4:  # 80% of max_correction_norm=0.5
                flags["F8"] = True

    # F9: Fallback failure — fallback was active AND a safety-critical
    # failure (collision, grounding) still occurred.
    if result.metadata.get("fallback_active", False):
        if flags.get("F1", False) or flags.get("F5", False):
            flags["F9"] = True

    # F10: Runtime deadline miss — only flag if miss rate > 10%.
    # A single deadline miss is an operational event, not a system failure.
    rt_stats = result.runtime_stats or {}
    n_commands = max(len(result.command_history), 1)
    deadline_misses = rt_stats.get("deadline_misses", 0)
    if isinstance(deadline_misses, (int, float)) and deadline_misses > 0:
        miss_rate = deadline_misses / n_commands
        if miss_rate > 0.10:
            flags["F10"] = True

    # Determine primary failure type (worst first)
    primary = None
    for fi in range(1, 11):
        if flags[f"F{fi}"]:
            primary = f"F{fi}"
            break

    return {"flags": flags, "primary_type": primary}


def summarize_failures(results: list) -> dict:
    """Summarize failure counts across multiple episodes.

    Args:
        results: List of EpisodeResult objects.

    Returns:
        dict with failure counts and rates.
    """
    n_total = len(results)
    summary = {"n_total": n_total, "failures": {}}
    for code, name in FAILURE_CODES.items():
        count = sum(1 for r in results if r.failure_flags.get(code, False))
        summary["failures"][code] = {
            "name": name,
            "count": count,
            "rate": count / max(n_total, 1),
        }
    summary["any_failure"] = sum(
        1 for r in results if r.failure_type is not None
    )
    summary["any_failure_rate"] = summary["any_failure"] / max(n_total, 1)
    return summary

"""Unified baseline and ablation registry — single source of truth.

Every experiment script, the closed-loop runner, and analysis tools MUST
source method definitions from this module.  No script-local overrides are
permitted for baseline / ablation feature flags.

Schema version is bumped whenever the registry changes so that old result
CSV files can be audited.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Optional

# Increment whenever baseline or ablation definitions change.
REGISTRY_VERSION = "3.0"

# ═══════════════════════════════════════════════════════════════════════════
# Feature-flag vocabulary (must match controller.flags keys)
# ═══════════════════════════════════════════════════════════════════════════

_ALL_FEATURE_KEYS = frozenset({
    "enable_multi_rule",
    "enable_target_size",
    "enable_speed_sensitive_domain",
    "enable_vessel_type_weight",
    "enable_behavior_uncertainty",
    "enable_shallow_water",
    "enable_wind_current",
    "enable_bank_effect",
    "enable_ship_interaction",
    "enable_chance_constraint",
    "enable_tube_mpc",
    "enable_cbf_filter",
    "enable_fallback",
})

# Every feature defaults to False.  Each baseline / Proposed MUST explicitly
# set every key it cares about.  This prevents "accidental True" from
# controller defaults or config files.
ALL_FEATURES_DEFAULT = {k: False for k in sorted(_ALL_FEATURE_KEYS)}

# ═══════════════════════════════════════════════════════════════════════════
# Baseline definitions
# ═══════════════════════════════════════════════════════════════════════════

def _make_features(**overrides: bool) -> dict:
    """Build a features dict with ALL keys explicit (default False)."""
    d = dict(ALL_FEATURES_DEFAULT)
    d.update(overrides)
    return d


# ═══════════════════════════════════════════════════════════════════════════
# Baseline definitions  (every key explicit — no implicit defaults)
# ═══════════════════════════════════════════════════════════════════════════

BASELINE_REGISTRY: dict[str, dict] = {
    # ── Heuristic / reactive baselines (no MPC) ─────────────────────────
    "B1": {
        "label": "COLREGs + VO/OZT",
        "controller": "velocity_obstacle",
        "description": "COLREGs rule-based velocity obstacle heuristic.",
        "features": _make_features(),
        "backend_override": None,
    },
    "B2": {
        "label": "Rule-based DWA",
        "controller": "dynamic_window",
        "description": "Dynamic Window Approach with basic COLREGs cost terms.",
        "features": _make_features(),
        "backend_override": None,
    },
    # ── Progressive MPC baselines ────────────────────────────────────────
    "B3": {
        "label": "Deterministic MMG-MPC (sampling)",
        "controller": "mpc",
        "description": "Sampling-based MPC with basic COLREGs directional heuristic. NO CasADi/IPOPT, NO multi-rule P0-P5, NO chance constraints. This is the computationally lightweight baseline — does NOT use the nonlinear MPC described in the paper.",
        "features": _make_features(
            enable_target_size=True,
            enable_speed_sensitive_domain=True,
            enable_vessel_type_weight=True,
            enable_behavior_uncertainty=True,
            # NOTE: enable_multi_rule is FALSE here because sampling_based_mpc()
            # only uses a COLREGs directional heuristic (prefer_starboard).
            # It does NOT support the full P0-P5 rule hierarchy.
        ),
        "backend_override": "sampling",
    },
    "B4": {
        "label": "CC-MPC",
        "controller": "mpc",
        "description": "B3 + chance constraint (Σ_rel).",
        "features": _make_features(
            enable_multi_rule=True,
            enable_target_size=True,
            enable_speed_sensitive_domain=True,
            enable_vessel_type_weight=True,
            enable_behavior_uncertainty=True,
            enable_chance_constraint=True,
        ),
        "backend_override": None,
    },
    "B5": {
        "label": "PE-CC-MPC",
        "controller": "mpc",
        "description": "B4 + physics (shallow / bank / ship / wind/current).",
        "features": _make_features(
            enable_multi_rule=True,
            enable_target_size=True,
            enable_speed_sensitive_domain=True,
            enable_vessel_type_weight=True,
            enable_behavior_uncertainty=True,
            enable_chance_constraint=True,
            enable_shallow_water=True,
            enable_wind_current=True,
            enable_bank_effect=True,
            enable_ship_interaction=True,
        ),
        "backend_override": None,
    },
    "B6": {
        "label": "PE-CC-Tube-MPC",
        "controller": "mpc",
        "description": "B5 + robust tube + adaptive scaling.",
        "features": _make_features(
            enable_multi_rule=True,
            enable_target_size=True,
            enable_speed_sensitive_domain=True,
            enable_vessel_type_weight=True,
            enable_behavior_uncertainty=True,
            enable_chance_constraint=True,
            enable_shallow_water=True,
            enable_wind_current=True,
            enable_bank_effect=True,
            enable_ship_interaction=True,
            enable_tube_mpc=True,
        ),
        "backend_override": None,
    },
    "B7": {
        "label": "PE-CC-Tube-MPC + CBF",
        "controller": "mpc",
        "description": "B6 + CBF-QP runtime safety filter.  No fallback.",
        "features": _make_features(
            enable_multi_rule=True,
            enable_target_size=True,
            enable_speed_sensitive_domain=True,
            enable_vessel_type_weight=True,
            enable_behavior_uncertainty=True,
            enable_chance_constraint=True,
            enable_shallow_water=True,
            enable_wind_current=True,
            enable_bank_effect=True,
            enable_ship_interaction=True,
            enable_tube_mpc=True,
            enable_cbf_filter=True,
        ),
        "backend_override": None,
    },
    # ── CBF-only baseline (G1 — 2026-06-10) ─────────────────────────────
    "B8": {
        "label": "CBF-QP only (no MPC)",
        "controller": "mpc",
        "description": "CBF-QP safety filter with single-step MPC (horizon=1). "
                       "No tube MPC, no chance constraints.  Verifies that CBF "
                       "alone cannot replace predictive planning — MPC provides "
                       "the anticipatory behaviour that CBF lacks.",
        "features": _make_features(
            enable_multi_rule=True,
            enable_target_size=True,
            enable_speed_sensitive_domain=True,
            enable_vessel_type_weight=True,
            enable_behavior_uncertainty=True,
            enable_shallow_water=True,
            enable_wind_current=True,
            enable_bank_effect=True,
            enable_ship_interaction=True,
            enable_cbf_filter=True,
        ),
        "backend_override": "scipy",
        "mpc_overrides": {"horizon": 1},
    },
    # ── Proposed (full system — the ONLY method with fallback) ────────────
    "Proposed": {
        "label": "TA-MRC-PE-CC-Tube-MPC",
        "controller": "mpc",
        "description": "B7 + fallback.  Complete system.",
        "features": _make_features(
            enable_multi_rule=True,
            enable_target_size=True,
            enable_speed_sensitive_domain=True,
            enable_vessel_type_weight=True,
            enable_behavior_uncertainty=True,
            enable_chance_constraint=True,
            enable_shallow_water=True,
            enable_wind_current=True,
            enable_bank_effect=True,
            enable_ship_interaction=True,
            enable_tube_mpc=True,
            enable_cbf_filter=True,
            enable_fallback=True,
        ),
        "backend_override": None,
    },
}

# ═══════════════════════════════════════════════════════════════════════════
# Planned baselines (NOT YET IMPLEMENTED — research roadmap only)
# ═══════════════════════════════════════════════════════════════════════════
#
# These are documented to show what a "strong Q1" submission should include.
# They are NOT registered in BASELINE_REGISTRY and cannot be selected via
# --method.  See docs/SCI_Q1_EXPERIMENT_PLAN.md Section 3.2.
#
#   B0  — Artificial Potential Field (APF)
#         Classic reactive method.  Low implementation effort.
#         Status: PLANNED, not implemented.
#
#   B8  — CBF-only (CBF-QP without MPC)
#         Verify CBF vs MPC contribution to safety.
#         Status: PLANNED, achievable via controller config (Proposed with
#                 enable_tube_mpc=False, enable_chance_constraint=False,
#                 horizon=1).
#
#   B9  — Deep RL baseline (PPO/SAC trained on same scenarios)
#         External dependency (stable-baselines3, gymnasium).
#         Status: PLANNED, high implementation effort.
#
#   B10 — COLREGs-MPC (Villagómez et al. 2025 style)
#         COLREGs-only MPC without multi-rule hierarchy.
#         Status: PLANNED, achievable via B3 with enable_multi_rule=False.
#
#   B11 — Intention-aware MPC
#         Uses IMM-filter multi-modal prediction in MPC cost.
#         Status: PLANNED, infrastructure exists in risk/intent_predictor.py.
#
#   B12 — Distributed MPC
#         Multi-vessel cooperative control.
#         Status: PLANNED, out of scope for single-agent framework.
# ═══════════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════════
# Ablation definitions  (each removes ONE primary feature from Proposed)
# ═══════════════════════════════════════════════════════════════════════════

ABLATION_REGISTRY: dict[str, dict] = {
    "A1": {
        "label": "w/o target vessel awareness",
        "removes": "enable_target_size",
        "hypothesis": "H1 — target-as-point degrades safety",
    },
    "A2": {
        "label": "w/o multi-source rules",
        "removes": "enable_multi_rule",
        "hypothesis": "H2 — COLREGs-only insufficient for restricted waters",
    },
    "A3": {
        "label": "w/o physics enhancement",
        "removes": ["enable_shallow_water", "enable_wind_current",
                     "enable_bank_effect", "enable_ship_interaction"],
        "hypothesis": "H3 — physics ignorance degrades feasibility in restricted waters",
    },
    "A4": {
        "label": "w/o chance constraints",
        "removes": "enable_chance_constraint",
        "hypothesis": "H4 — no probabilistic margin degrades AIS-uncertainty robustness",
    },
    "A5": {
        "label": "w/o tube MPC",
        "removes": "enable_tube_mpc",
        "hypothesis": "H4 — no robust tube degrades disturbance rejection",
    },
    "A6": {
        "label": "w/o CBF-QP safety filter",
        "removes": "enable_cbf_filter",
        "hypothesis": "H5 — no runtime safety filter increases collision rate",
    },
    "A7": {
        "label": "w/o fallback",
        "removes": "enable_fallback",
        "hypothesis": "H5 — no fallback reduces recovery from infeasibility",
    },
    "A8": {
        "label": "w/o bank effect",
        "removes": "enable_bank_effect",
        "hypothesis": "H3 — no bank-effect model increases near-bank violations",
    },
    "A9": {
        "label": "w/o shallow water effect",
        "removes": "enable_shallow_water",
        "hypothesis": "H3 — no shallow-water model degrades UKC safety",
    },
    "A10": {
        "label": "w/o ship interaction",
        "removes": "enable_ship_interaction",
        "hypothesis": "H3 — no ship-interaction model increases close-quarters risk",
    },
    "A11": {
        "label": "w/o dynamic ship domain (scalar only)",
        "removes": ["enable_speed_sensitive_domain", "enable_vessel_type_weight",
                     "enable_behavior_uncertainty"],
        "hypothesis": "H1 — scalar domain insufficient for heterogeneous traffic",
    },
    "A12": {
        "label": "w/o multi-modal target prediction",
        "removes": "enable_behavior_uncertainty",
        "hypothesis": "H1/H4 — constant-velocity prediction degrades safety with non-cooperative targets",
    },
}

# ═══════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════

def get_baseline(method: str) -> dict:
    """Return a *deep copy* of the baseline definition for *method*."""
    if method not in BASELINE_REGISTRY:
        raise KeyError(
            f"Unknown baseline '{method}'. "
            f"Valid: {sorted(BASELINE_REGISTRY.keys())}"
        )
    return deepcopy(BASELINE_REGISTRY[method])


def get_ablation(ablation_id: str) -> dict:
    """Return a *deep copy* of the ablation definition."""
    if ablation_id not in ABLATION_REGISTRY:
        raise KeyError(
            f"Unknown ablation '{ablation_id}'. "
            f"Valid: {sorted(ABLATION_REGISTRY.keys())}"
        )
    return deepcopy(ABLATION_REGISTRY[ablation_id])


def build_ablation_features(ablation_id: str) -> dict:
    """Build controller feature flags for *ablation_id*: Proposed minus one feature."""
    ablation = get_ablation(ablation_id)
    features = deepcopy(BASELINE_REGISTRY["Proposed"]["features"])
    removes = ablation["removes"]
    if isinstance(removes, str):
        removes = [removes]
    for key in removes:
        features[key] = False
    return features


def list_baseline_methods() -> list[str]:
    """Return ordered list of all baseline method IDs."""
    return list(BASELINE_REGISTRY.keys())


def list_ablation_ids() -> list[str]:
    """Return ordered list of all ablation IDs."""
    return list(ABLATION_REGISTRY.keys())


def validate_registry() -> list[str]:
    """Return list of issues (empty = valid). Checks key constraints."""
    issues = []

    # 1. Every baseline MUST list ALL _ALL_FEATURE_KEYS explicitly
    for method, entry in BASELINE_REGISTRY.items():
        features = entry.get("features", {})
        for k in _ALL_FEATURE_KEYS:
            if k not in features:
                issues.append(f"{method}: missing feature key '{k}' (must be explicit)")
        for k in features:
            if k not in _ALL_FEATURE_KEYS:
                issues.append(f"{method}: unknown feature '{k}'")

    # 2. Proposed must have fallback=True (ONLY baseline with it)
    proposed = BASELINE_REGISTRY["Proposed"]["features"]
    if not proposed.get("enable_fallback"):
        issues.append("Proposed.enable_fallback must be True")

    # 3. B3-B8 must NOT have fallback (Proposed only)
    for m in ["B3", "B4", "B5", "B6", "B7", "B8"]:
        if BASELINE_REGISTRY[m]["features"].get("enable_fallback"):
            issues.append(f"{m} must NOT have enable_fallback (Proposed only)")

    # 4. B3 must have backend_override="sampling"
    if BASELINE_REGISTRY["B3"].get("backend_override") != "sampling":
        issues.append("B3 must have backend_override='sampling'")

    # 5. Every ablation must reference valid feature keys
    for aid, entry in ABLATION_REGISTRY.items():
        removes = entry["removes"]
        if isinstance(removes, str):
            removes = [removes]
        for k in removes:
            if k not in _ALL_FEATURE_KEYS:
                issues.append(f"{aid}: unknown feature '{k}'")

    return issues

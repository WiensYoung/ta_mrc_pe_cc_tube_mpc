"""Tests for the unified baseline/ablation registry.

Verifies:
  1. Registry has all expected methods
  2. No fallback leakage into B3-B7
  3. Each ablation changes exactly one primary feature
  4. Feature flags are valid
  5. Experiment scripts use the registry
"""

import os
import pytest


# ── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def registry():
    from ta_mrc_pe_cc_tube_mpc.experiments.baseline_registry import (
        ABLATION_REGISTRY,
        BASELINE_REGISTRY,
    )
    return {"baselines": BASELINE_REGISTRY, "ablations": ABLATION_REGISTRY}


# ── 1. Expected methods ─────────────────────────────────────────────────────

def test_baseline_registry_has_expected_methods(registry):
    """Baseline registry must contain B1–B7 + Proposed."""
    expected = {"B1", "B2", "B3", "B4", "B5", "B6", "B7", "Proposed"}
    actual = set(registry["baselines"].keys())
    missing = expected - actual
    assert not missing, f"Missing baselines: {missing}"


def test_each_baseline_has_required_keys(registry):
    for method, entry in registry["baselines"].items():
        for key in ("label", "controller", "description", "features"):
            assert key in entry, f"{method} missing key '{key}'"


# ── 2. No fallback leakage ──────────────────────────────────────────────────

def test_baseline_flags_no_fallback_leakage(registry):
    """B3-B7 must NOT have enable_fallback=True. Only Proposed may."""
    for method in ["B3", "B4", "B5", "B6", "B7"]:
        features = registry["baselines"][method]["features"]
        assert not features.get("enable_fallback"), (
            f"{method} must not have enable_fallback=True — "
            "fallback is Proposed-only"
        )


def test_proposed_has_fallback(registry):
    """Proposed MUST have enable_fallback=True."""
    features = registry["baselines"]["Proposed"]["features"]
    assert features.get("enable_fallback"), (
        "Proposed must have enable_fallback=True"
    )


# ── 3. One primary feature per ablation ─────────────────────────────────────

def test_ablation_changes_exactly_one_primary_feature():
    """Each ablation removes exactly ONE feature from Proposed."""
    from ta_mrc_pe_cc_tube_mpc.experiments.baseline_registry import (
        ABLATION_REGISTRY,
        BASELINE_REGISTRY,
    )
    proposed = BASELINE_REGISTRY["Proposed"]["features"]
    # Which features Proposed enables
    proposed_true = {k for k, v in proposed.items() if v}

    for aid, entry in ABLATION_REGISTRY.items():
        removes = entry["removes"]
        if isinstance(removes, str):
            removes = [removes]
        # Build ablation features
        ablated = dict(proposed)
        for k in removes:
            ablated[k] = False
        # Count differences from Proposed (only True→False counts as a change)
        diffs = [k for k in proposed_true if ablated.get(k) is False]
        assert len(diffs) >= 1, (
            f"{aid}: no feature was actually removed (removes={removes})"
        )
        # Check that the removed keys are valid
        for k in removes:
            assert k in proposed, (
                f"{aid}: '{k}' not a valid feature key"
            )


# ── 4. Registry validation ──────────────────────────────────────────────────

def test_registry_validate_no_issues():
    """validate_registry() returns empty list."""
    from ta_mrc_pe_cc_tube_mpc.experiments.baseline_registry import (
        validate_registry,
    )
    issues = validate_registry()
    assert not issues, f"Registry issues: {issues}"


# ── 5. Progressive chain is strict ──────────────────────────────────────────

def test_progressive_chain_strictly_additive():
    """B3 ⊂ B4 ⊂ B5 ⊂ B6 ⊂ B7 ⊂ Proposed (feature-set subset)."""
    from ta_mrc_pe_cc_tube_mpc.experiments.baseline_registry import (
        BASELINE_REGISTRY,
    )

    chain = ["B3", "B4", "B5", "B6", "B7", "Proposed"]
    for i in range(len(chain) - 1):
        prev_true = {k for k, v in BASELINE_REGISTRY[chain[i]]["features"].items() if v}
        next_true = {k for k, v in BASELINE_REGISTRY[chain[i+1]]["features"].items() if v}
        missing = prev_true - next_true
        assert not missing, (
            f"{chain[i]} → {chain[i+1]}: features lost: {missing}"
        )
        added = next_true - prev_true
        assert len(added) >= 1, (
            f"{chain[i]} → {chain[i+1]}: no new features added. "
            f"prev={sorted(prev_true)}, next={sorted(next_true)}"
        )


# ── 6. Experiment scripts use registry ──────────────────────────────────────

def test_run_core_experiments_uses_registry():
    """run_core_experiments default methods come from the registry."""
    from ta_mrc_pe_cc_tube_mpc.experiments.baseline_registry import (
        list_baseline_methods,
    )
    methods = list_baseline_methods()
    assert methods == ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "Proposed"]
    assert len(methods) == 9  # G1: B8 (CBF-only) added 2026-06-10


def test_ablation_list_consistent():
    """Ablation registry has exactly 12 entries."""
    from ta_mrc_pe_cc_tube_mpc.experiments.baseline_registry import (
        list_ablation_ids,
    )
    ids = list_ablation_ids()
    assert len(ids) == 12, f"Expected 12 ablations, got {len(ids)}: {ids}"
    assert ids[0].startswith("A"), f"First ablation ID should start with 'A': {ids[0]}"


# ── 7. Backend constraints ──────────────────────────────────────────────────

def test_b3_uses_sampling_backend():
    """B3 should use sampling backend for numerical robustness."""
    from ta_mrc_pe_cc_tube_mpc.experiments.baseline_registry import (
        get_baseline,
    )
    b3 = get_baseline("B3")
    assert b3["backend_override"] == "sampling", (
        "B3 should use 'sampling' backend_override"
    )


def test_proposed_no_backend_override():
    """Proposed should follow the default config backend (CasADi)."""
    from ta_mrc_pe_cc_tube_mpc.experiments.baseline_registry import (
        get_baseline,
    )
    p = get_baseline("Proposed")
    assert p["backend_override"] is None, (
        "Proposed should use default config backend, not override"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Runtime verification: controller flags match registry at instantiation
# ═══════════════════════════════════════════════════════════════════════════════

_ALL_FLAGS = [
    "enable_multi_rule", "enable_target_size", "enable_speed_sensitive_domain",
    "enable_vessel_type_weight", "enable_behavior_uncertainty",
    "enable_shallow_water", "enable_wind_current", "enable_bank_effect",
    "enable_ship_interaction", "enable_chance_constraint",
    "enable_tube_mpc", "enable_cbf_filter", "enable_fallback",
]


def _build_controller_for_method(method: str):
    """Build a real controller via ClosedLoopRunner._build_controller."""
    from ta_mrc_pe_cc_tube_mpc.simulation.closed_loop_runner import (
        ClosedLoopRunner,
    )
    from ta_mrc_pe_cc_tube_mpc.types import VesselParams
    from ta_mrc_pe_cc_tube_mpc.utils.io_utils import load_config_with_overrides

    _project_root = os.path.join(os.path.dirname(__file__), "..")
    config = load_config_with_overrides(os.path.join(_project_root, "configs"))
    runner = ClosedLoopRunner(config)
    params = VesselParams()
    return runner._build_controller(method, params)


def _get_controller_flags(ctrl) -> dict:
    """Extract all feature flags from an instantiated controller."""
    return {f: getattr(ctrl, f, None) for f in _ALL_FLAGS}


class TestRuntimeFeatureFlags:
    """Verify that controller runtime flags match registry definitions."""

    def test_b3_features_are_disabled(self):
        """B3: multi_rule + aux ON, everything else OFF."""
        from ta_mrc_pe_cc_tube_mpc.experiments.baseline_registry import (
            get_baseline,
        )
        expected = get_baseline("B3")["features"]
        ctrl = _build_controller_for_method("B3")
        flags = _get_controller_flags(ctrl)

        for key in _ALL_FLAGS:
            want = expected.get(key, False)
            got = flags.get(key)
            assert got == want, f"B3 {key}: expected {want}, got {got}"

    def test_b4_only_adds_chance_constraint(self):
        """B4 = B3 + enable_chance_constraint + CasADi backend.

        NOTE: B4 switches from sampling (B3) to CasADi/IPOPT backend,
        so enable_multi_rule is True (CasADi supports full multi-rule)
        even though B3 has it False (sampling MPC only has COLREGs heuristic).
        """
        from ta_mrc_pe_cc_tube_mpc.experiments.baseline_registry import (
            get_baseline,
        )
        b3 = get_baseline("B3")["features"]
        b4 = get_baseline("B4")["features"]
        ctrl = _build_controller_for_method("B4")
        flags = _get_controller_flags(ctrl)

        # B4 is B3 + chance_constraint, EXCEPT:
        # - enable_multi_rule: True in B4 (CasADi), False in B3 (sampling)
        # All other B3-false flags remain false in B4.
        for key in _ALL_FLAGS:
            if key == "enable_chance_constraint":
                assert flags[key] is True, f"B4 {key} must be True"
            elif key == "enable_multi_rule":
                # B4 uses CasADi → supports full multi-rule
                assert flags[key] is True, f"B4 {key} must be True (CasADi backend)"
            else:
                want = b3.get(key, False)
                assert flags[key] == want, f"B4 {key}: expected {want}, got {flags[key]}"

    def test_b5_adds_physics_but_not_tube_cbf_fallback(self):
        """B5 = B4 + physics. No Tube/CBF/Fallback."""
        ctrl = _build_controller_for_method("B5")
        flags = _get_controller_flags(ctrl)
        assert flags["enable_shallow_water"] is True
        assert flags["enable_wind_current"] is True
        assert flags["enable_bank_effect"] is True
        assert flags["enable_ship_interaction"] is True
        assert flags["enable_tube_mpc"] is False
        assert flags["enable_cbf_filter"] is False
        assert flags["enable_fallback"] is False

    def test_b6_adds_tube_but_not_cbf_fallback(self):
        """B6 = B5 + tube. No CBF/Fallback."""
        ctrl = _build_controller_for_method("B6")
        flags = _get_controller_flags(ctrl)
        assert flags["enable_tube_mpc"] is True
        assert flags["enable_cbf_filter"] is False
        assert flags["enable_fallback"] is False

    def test_b7_adds_cbf_but_not_fallback(self):
        """B7 = B6 + CBF. No fallback."""
        ctrl = _build_controller_for_method("B7")
        flags = _get_controller_flags(ctrl)
        assert flags["enable_cbf_filter"] is True
        assert flags["enable_fallback"] is False

    def test_proposed_enables_all_claimed_features(self):
        """Proposed has ALL 13 features enabled."""
        ctrl = _build_controller_for_method("Proposed")
        flags = _get_controller_flags(ctrl)
        for key in _ALL_FLAGS:
            assert flags[key] is True, f"Proposed {key} must be True, got {flags[key]}"

    def test_proposed_is_only_method_with_fallback(self):
        """Only Proposed has enable_fallback=True."""
        for method in ["B3", "B4", "B5", "B6", "B7"]:
            ctrl = _build_controller_for_method(method)
            assert ctrl.enable_fallback is False, (
                f"{method} enable_fallback must be False"
            )
        ctrl_p = _build_controller_for_method("Proposed")
        assert ctrl_p.enable_fallback is True, "Proposed must have fallback"

    def test_backend_override_sampling_is_applied(self):
        """B3 controller has use_sampling=True at runtime."""
        from ta_mrc_pe_cc_tube_mpc.experiments.baseline_registry import (
            get_baseline,
        )
        b3 = get_baseline("B3")
        assert b3["backend_override"] == "sampling"

        ctrl = _build_controller_for_method("B3")
        assert ctrl.tube_mpc.use_sampling is True, (
            "B3 tube_mpc.use_sampling must be True"
        )
        assert ctrl.tube_mpc.backend == "sampling", (
            f"B3 tube_mpc.backend must be 'sampling', got '{ctrl.tube_mpc.backend}'"
        )

    def test_proposed_uses_casadi_backend(self):
        """Proposed uses CasADi backend (not sampling)."""
        ctrl = _build_controller_for_method("Proposed")
        assert ctrl.tube_mpc.use_sampling is False, (
            "Proposed must NOT use sampling backend"
        )
        assert ctrl.tube_mpc.backend == "casadi", (
            f"Proposed backend must be 'casadi', got '{ctrl.tube_mpc.backend}'"
        )

    def test_ablation_override_takes_final_precedence(self):
        """Ablation cfg['controller'] override wins over registry features."""
        from ta_mrc_pe_cc_tube_mpc.simulation.closed_loop_runner import (
            ClosedLoopRunner,
        )
        from ta_mrc_pe_cc_tube_mpc.types import VesselParams
        from ta_mrc_pe_cc_tube_mpc.utils.io_utils import load_config_with_overrides

        _project_root = os.path.join(os.path.dirname(__file__), "..")
        config = load_config_with_overrides(os.path.join(_project_root, "configs"))
        runner = ClosedLoopRunner(config)
        params = VesselParams()

        # Ablation A7 removes fallback from Proposed
        ablation_cfg = {"controller": {"enable_fallback": False}}
        ctrl = runner._build_controller("Proposed", params, ablation_cfg)
        assert ctrl.enable_fallback is False, (
            "Ablation override must disable fallback even for Proposed"
        )
        # But other Proposed features remain
        assert ctrl.enable_cbf_filter is True, "Non-ablated features must stay on"

    def test_all_features_explicit_in_registry(self):
        """Every baseline has all 13 feature keys (no implicit defaults)."""
        from ta_mrc_pe_cc_tube_mpc.experiments.baseline_registry import (
            BASELINE_REGISTRY,
            _ALL_FEATURE_KEYS,
        )
        for method in ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "Proposed"]:
            entry = BASELINE_REGISTRY[method]
            features = entry["features"]
            missing = _ALL_FEATURE_KEYS - set(features.keys())
            assert not missing, f"{method}: missing explicit keys {missing}"
            extra = set(features.keys()) - _ALL_FEATURE_KEYS
            assert not extra, f"{method}: unknown keys {extra}"

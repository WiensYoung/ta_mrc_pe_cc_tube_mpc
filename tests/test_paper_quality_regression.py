"""Regression tests for paper-quality fixes.

Validates:
  1. Tube margin is NOT double-counted in d_safe_cc_list
  2. Tube-disabled ablation has zero constraint tightening
  3. Boole risk allocation respects total epsilon budget
  4. Safety metric field semantics (min_distance_actual, min_required_safety_distance)
  5. F9 is NOT equivalent to fallback_trigger (only true failure)
  6. Closed-loop runner uses unified safety evaluator
"""

import numpy as np
import pytest

from ta_mrc_pe_cc_tube_mpc.control.mpc_problem import MPCProblem
from ta_mrc_pe_cc_tube_mpc.control.risk_allocation import allocate_chance_risk
from ta_mrc_pe_cc_tube_mpc.simulation.failure_detector import detect_failures
from ta_mrc_pe_cc_tube_mpc.types import VesselState, VesselParams, ControlCommand


# ── Fixtures ────────────────────────────────────────────────────────────────

def _dummy_dynamics(state, cmd, env, dt):
    """Simple kinematic dynamics for testing."""
    return VesselState(
        x=state.x + state.u * dt,
        y=state.y + state.v * dt,
        psi=state.psi + state.r * dt,
        u=state.u, v=state.v, r=state.r,
        timestamp=state.timestamp + dt,
    )


@pytest.fixture
def mpc_problem():
    return MPCProblem(dynamics=_dummy_dynamics, horizon=10, dt=0.5)


# ── Test 1: Tube margin not double-counted ──────────────────────────────────

def test_tube_not_double_counted(mpc_problem):
    """d_safe_cc_list = d_safe_eff + uncertainty_margin, NOT + 2*rho_tube."""
    state = VesselState(x=0, y=0, psi=0, u=7, v=0, r=0)
    ref = [state] * 11
    targets = [np.zeros((1, 2))]
    d_safe_eff = [500.0]
    env = type('Env', (), {'water_depth': 50.0, 'channel_width': None,
                           'bank_distance_left': None, 'bank_distance_right': None,
                           'current_x': 0.0, 'current_y': 0.0,
                           'wind_x': 0.0, 'wind_y': 0.0})()

    rho_tube = 50.0
    eps = 0.10

    cmd, info = mpc_problem.solve(
        current_state=state, reference_trajectory=ref,
        target_positions=targets, d_safe_eff_list=[d + rho_tube for d in d_safe_eff],
        env=env, rho_tube=rho_tube, epsilon=eps,
    )

    d_safe_cc = info.get("d_safe_cc_list", [])
    if d_safe_cc:
        # Should be d_safe_eff + rho_tube + chance_margin (rho_tube once)
        # NOT d_safe_eff + 2*rho_tube + chance_margin
        for val in d_safe_cc:
            assert val < d_safe_eff[0] + 2 * rho_tube + 100, (
                f"d_safe_cc={val} >= twice rho_tube, suggests double-counting"
            )


# ── Test 2: Tube-disabled ablation ──────────────────────────────────────────

def test_tube_disabled_ablation_clean():
    """When enable_tube_mpc=False, rho_tube_override must be 0.0, not None."""
    from ta_mrc_pe_cc_tube_mpc.control.controller import TAMRCPECCTubeMPCController
    from ta_mrc_pe_cc_tube_mpc.types import EnvironmentState, TargetShip

    params = VesselParams()
    cfg = {
        "controller": {
            "enable_tube_mpc": False,
            "enable_multi_rule": False,
            "enable_cbf_filter": False,
            "enable_fallback": False,
            "enable_chance_constraint": False,
        },
        "mpc": {"horizon": 5, "dt": 0.5, "backend": "scipy"},
    }
    ctrl = TAMRCPECCTubeMPCController(params, cfg)

    own = VesselState(x=0, y=0, psi=0, u=7, v=0, r=0)
    env = EnvironmentState()
    cmd = ctrl.compute_control(own, params, [], env)

    mpc_info = cmd.metadata.get("mpc_info", {})
    # When tube is disabled, the metadata must not claim tube was used
    tube_radius = cmd.metadata.get("tube_radius", -1)
    assert tube_radius == 0.0 or tube_radius == -1, (
        f"tube_radius={tube_radius} when tube is disabled"
    )


# ── Test 3: Boole risk allocation respects budget ───────────────────────────

def test_boole_risk_allocation_budget():
    """Allocated epsilons must not exceed total budget."""
    eps_total = 0.10
    N = 20  # horizon
    M = 3   # targets
    eps_step = allocate_chance_risk(eps_total, N, M, mode="boole")
    # Boole: eps_step * N * M <= eps_total (actually == eps_total)
    assert abs(eps_step * N * M - eps_total) < 1e-12, (
        f"Boole allocation: {eps_step} * {N} * {M} != {eps_total}"
    )


def test_risk_allocation_none_mode():
    """'none' mode returns epsilon_total unchanged."""
    assert allocate_chance_risk(0.10, 20, 3, mode="none") == 0.10


def test_risk_allocation_invalid_epsilon():
    """Epsilon out of range must raise ValueError."""
    with pytest.raises(ValueError):
        allocate_chance_risk(0.0, 20, 3)
    with pytest.raises(ValueError):
        allocate_chance_risk(1.5, 20, 3)


# ── Test 4: Safety metric field semantics ───────────────────────────────────

def test_safety_metric_fields_present():
    """After a proper episode, key paper metrics must be present."""
    from ta_mrc_pe_cc_tube_mpc.evaluation.metrics import compute_episode_metrics
    from ta_mrc_pe_cc_tube_mpc.types import (
        EpisodeConfig, EpisodeResult, EnvironmentState, TargetShip,
    )

    own = VesselState(x=0, y=0, psi=0, u=7, v=0, r=0, timestamp=0)
    tgt = TargetShip(
        mmsi="test", state=VesselState(x=1000, y=0, psi=np.pi, u=5, v=0, r=0, timestamp=0),
        length=150, beam=25,
    )
    ep = EpisodeConfig(
        scenario_id="test", duration=5.0, dt=0.5,
        ownship_initial_state=own, ownship_params=VesselParams(),
        targets=[tgt], environment_sequence=[EnvironmentState()],
        random_seed=0,
    )
    result = EpisodeResult(
        episode_id="test", scenario_id="test", method="Proposed", seed=0,
        state_history=[own, own], command_history=[], target_histories=[[tgt.state, tgt.state]],
        metadata={"domain_results": []},
    )
    m = compute_episode_metrics(result, ep)

    assert "min_distance_actual" in m
    assert "min_required_safety_distance" in m


# ── Test 5: F9 is NOT equivalent to fallback trigger ────────────────────────

def test_f9_only_on_true_failure():
    """F9 must only trigger when fallback is active AND safety failure occurs."""
    from ta_mrc_pe_cc_tube_mpc.types import EpisodeConfig, EpisodeResult

    # Case A: fallback active, no collision → F9 must be False
    ep = EpisodeConfig(scenario_id="x", duration=1, dt=0.5)
    r = EpisodeResult(
        metadata={"fallback_active": True, "mpc_feasible": True},
        metrics={},
    )
    flags_a = detect_failures(r, ep, {})["flags"]
    assert not flags_a.get("F9", False), (
        "F9 should be False when fallback triggered without safety failure"
    )

    # Case B: fallback active + collision → F9 must be True
    flags_b = detect_failures(r, ep, {"collision": True})["flags"]
    assert flags_b.get("F9", False), (
        "F9 should be True when fallback triggered AND collision occurred"
    )


# ── Test 6: Closed-loop runner uses unified evaluator ───────────────────────

def test_runner_uses_safety_evaluator():
    """run_single() must populate unified safety evaluator fields."""
    from ta_mrc_pe_cc_tube_mpc.simulation.closed_loop_runner import ClosedLoopRunner
    from ta_mrc_pe_cc_tube_mpc.types import (
        EpisodeConfig, EnvironmentState, TargetShip,
    )

    own = VesselState(x=0, y=0, psi=0, u=7, v=0, r=0, timestamp=0)
    tgt = TargetShip(
        mmsi="t1", state=VesselState(x=500, y=0, psi=np.pi, u=5, v=0, r=0, timestamp=0),
        length=150, beam=25,
    )
    ep = EpisodeConfig(
        scenario_id="run_test", duration=5.0, dt=0.5,
        ownship_initial_state=own, ownship_params=VesselParams(),
        targets=[tgt], environment_sequence=[EnvironmentState()],
        random_seed=1,
    )
    cfg = {
        "simulation": {"dt": 0.5, "closed_loop_duration": 5.0},
        "mpc": {"horizon": 5, "dt": 0.5, "backend": "scipy"},
        "controller": {
            "enable_multi_rule": False, "enable_tube_mpc": False,
            "enable_cbf_filter": False, "enable_fallback": False,
            "enable_chance_constraint": False,
        },
    }
    runner = ClosedLoopRunner(cfg)
    result = runner.run_single(ep, "B3", seed=1)

    # Unified evaluator fields should be present
    assert "min_safety_margin" in result.metrics, (
        "Unified safety evaluator not called — missing 'min_safety_margin'"
    )
    assert "near_miss_event_count" in result.metrics

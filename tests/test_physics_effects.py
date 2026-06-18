"""Tests for physics effects: shallow water, bank, ship interaction, wind/current, tube."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest

from ta_mrc_pe_cc_tube_mpc.physics.shallow_water import assess_shallow_water
from ta_mrc_pe_cc_tube_mpc.physics.bank_effect import assess_bank_effect
from ta_mrc_pe_cc_tube_mpc.physics.ship_interaction import assess_ship_interaction
from ta_mrc_pe_cc_tube_mpc.physics.wind_current import assess_wind_current
from ta_mrc_pe_cc_tube_mpc.physics.tube_boundary import compute_tube_radius
from ta_mrc_pe_cc_tube_mpc.types import EnvironmentState, TargetShip, VesselState


class TestShallowWater:
    """Test shallow water depth/draught classification."""

    def test_deep_water(self):
        """h/T > 4.0 is deep water."""
        result = assess_shallow_water(water_depth=50.0, vessel_draught=10.0,
                                       vessel_speed=5.0, vessel_length=180.0)
        assert result.shallow_risk_level == "deep"
        assert result.I_shallow == 0.0
        assert result.rho_shallow == 0.0

    def test_classification_levels(self):
        """Each h/T range maps to correct risk level."""
        cases = [
            (50.0, "deep"),       # h/T = 5.0
            (35.0, "mild_shallow"),  # h/T = 3.5
            (25.0, "moderate_shallow"),  # h/T = 2.5
            (17.0, "strong_shallow"),  # h/T = 1.7
            (12.0, "forbidden"),   # h/T = 1.2
        ]
        for depth, expected in cases:
            result = assess_shallow_water(depth, 10.0, 5.0, 180.0)
            assert result.shallow_risk_level == expected, f"h={depth}, T=10"

    def test_ukc_violation(self):
        """UKC < UKC_min triggers violation."""
        result = assess_shallow_water(water_depth=11.0, vessel_draught=10.0,
                                       vessel_speed=5.0, vessel_length=180.0,
                                       ukc_min=1.5)
        assert result.ukc_violation

    def test_ukc_sufficient(self):
        """Sufficient UKC does not trigger violation."""
        result = assess_shallow_water(water_depth=20.0, vessel_draught=10.0,
                                       vessel_speed=5.0, vessel_length=180.0,
                                       ukc_min=1.5)
        assert not result.ukc_violation


class TestBankEffect:
    """Test bank effect conservative bounded disturbance."""

    def test_bank_distance_smaller_rho_larger(self):
        """Closer bank -> larger rho_bank (tube contribution)."""
        far = assess_bank_effect(5.0, 180.0, 30.0, 500.0, 500.0)
        near = assess_bank_effect(5.0, 180.0, 30.0, 50.0, 500.0)

        assert near.rho_bank > far.rho_bank

    def test_bank_distance_increases_disturbance(self):
        """Closer bank -> larger disturbance bound."""
        far = assess_bank_effect(5.0, 180.0, 30.0, 1000.0, 1000.0)
        near = assess_bank_effect(5.0, 180.0, 30.0, 60.0, 1000.0)

        assert near.bank_disturbance_bound > far.bank_disturbance_bound

    def test_no_bank_no_effect(self):
        """No bank information -> zero effect."""
        result = assess_bank_effect(5.0, 180.0, 30.0, None, None)
        assert result.bank_disturbance_bound == 0.0
        assert result.rho_bank == 0.0

    def test_bank_safety_margin_positive(self):
        """Bank safety margin is positive."""
        result = assess_bank_effect(5.0, 180.0, 30.0, 100.0, 200.0)
        assert result.bank_safety_margin > 0


class TestShipInteraction:
    """Test ship-ship interaction assessment."""

    def test_distant_target_no_interaction(self):
        """Distant target should not trigger interaction."""
        own = VesselState(x=0, y=0, psi=0, u=5, v=0, r=0)
        tgt = TargetShip(
            mmsi="T001", state=VesselState(x=5000, y=500, psi=np.pi, u=5, v=0, r=0),
            length=100, beam=20, draught=5,
        )

        result = assess_ship_interaction(own, 180.0, 30.0, [tgt])
        assert not result.active
        assert result.rho_ship == 0.0

    def test_close_target_triggers_interaction(self):
        """Close parallel target triggers ship interaction."""
        own = VesselState(x=0, y=0, psi=0, u=5, v=0, r=0)
        tgt = TargetShip(
            mmsi="T002", state=VesselState(x=50, y=20, psi=0, u=4, v=0, r=0),
            length=100, beam=20, draught=5,
        )

        result = assess_ship_interaction(own, 180.0, 30.0, [tgt])
        assert result.active
        assert result.rho_ship > 0.0


class TestWindCurrent:
    """Test wind/current disturbance assessment."""

    def test_current_contributes_to_rho(self):
        """Current adds to rho_current tube contribution."""
        no_current = assess_wind_current(0, 0, 0, 0, 180, 30, 5.0)
        with_current = assess_wind_current(0, 1.0, 0, 0, 180, 30, 5.0)

        assert with_current.rho_current > no_current.rho_current

    def test_stress_test_detection(self):
        """3 kn current flagged as stress test."""
        low = assess_wind_current(0, 1.0, 0, 0, 180, 30, 5.0)
        high = assess_wind_current(0, 1.5, 0, 0, 180, 30, 5.0)

        assert not low.current_stress_test
        assert high.current_stress_test


class TestTubeBoundary:
    """Test tube boundary additive formulation."""

    def test_tube_non_negative(self):
        """All tube components and radius are non-negative."""
        own = VesselState(x=0, y=0, psi=0, u=5, v=0, r=0)
        env = EnvironmentState(water_depth=50.0)
        targets = []

        result = compute_tube_radius(own, 180, 30, 10, targets, env)

        # Radius must be non-negative
        assert result["radius"] >= 0

        # All component values are float and non-negative
        for key, val in result["components"].items():
            assert isinstance(val, (int, float, np.floating)), (
                f"component '{key}' is {type(val).__name__}, expected numeric"
            )
            assert val >= 0, f"component '{key}'={val} should be >= 0"

        # Metadata contains the expected string keys
        assert isinstance(result["metadata"]["encounter_type"], str)
        assert isinstance(result["metadata"]["waterway_type"], str)

    def test_tube_decomposable(self):
        """Tube radius equals sum of components."""
        own = VesselState(x=0, y=0, psi=0, u=5, v=0, r=0)
        tgt = TargetShip(
            mmsi="T001", state=VesselState(x=200, y=0, psi=np.pi, u=3, v=0, r=0),
            length=100, beam=20, draught=5,
        )
        env = EnvironmentState(
            water_depth=20.0, current_x=0, current_y=0.5,
            bank_distance_left=80, bank_distance_right=200,
        )

        result = compute_tube_radius(own, 180, 30, 10, [tgt], env)

        # Radius must equal sum of component values
        total_from_components = sum(result["components"].values())
        assert result["radius"] == pytest.approx(total_from_components, rel=1e-6)

    def test_targets_increase_tube(self):
        """More targets -> larger tube (from ship interaction and AIS)."""
        own = VesselState(x=0, y=0, psi=0, u=5, v=0, r=0)
        env = EnvironmentState(water_depth=50.0)

        no_targets = compute_tube_radius(own, 180, 30, 10, [], env)

        tgt = TargetShip(
            mmsi="T001", state=VesselState(x=100, y=10, psi=0, u=4, v=0, r=0),
            length=100, beam=20, draught=5,
        )
        with_target = compute_tube_radius(own, 180, 30, 10, [tgt], env)

        assert with_target["radius"] >= no_targets["radius"]


class TestTubeRadiusSensitivity:
    """Sensitivity: each physics effect monotonically increases tube radius."""

    def _make_baseline(self):
        own = VesselState(x=0, y=0, psi=0, u=5, v=0, r=0)
        env = EnvironmentState(water_depth=50.0)
        return own, env, []

    def test_wind_current_increases_tube(self):
        """Stronger current → larger tube radius."""
        own, env_calm, targets = self._make_baseline()
        env_current = EnvironmentState(water_depth=50.0, current_x=0, current_y=1.5)

        r_calm = compute_tube_radius(own, 180, 30, 10, targets, env_calm,
                                      use_adaptive=False)
        r_curr = compute_tube_radius(own, 180, 30, 10, targets, env_current,
                                      use_adaptive=False)
        assert r_curr["components"]["current"] >= r_calm["components"]["current"]

    def test_bank_effect_increases_near_bank(self):
        """Closer bank → larger tube radius."""
        own, env_far, targets = self._make_baseline()
        # Far bank
        env_far = EnvironmentState(water_depth=50.0,
                                    bank_distance_left=500, bank_distance_right=500)
        # Near bank
        env_near = EnvironmentState(water_depth=50.0,
                                     bank_distance_left=30, bank_distance_right=500)

        r_far = compute_tube_radius(own, 180, 30, 10, targets, env_far,
                                     use_adaptive=False)
        r_near = compute_tube_radius(own, 180, 30, 10, targets, env_near,
                                     use_adaptive=False)
        assert r_near["components"]["bank_effect"] >= r_far["components"]["bank_effect"], (
            f"near bank={r_near['components']['bank_effect']:.3f} should be >= "
            f"far bank={r_far['components']['bank_effect']:.3f}"
        )

    def test_shallow_water_increases_at_low_ukc(self):
        """Shallow water → larger tube radius."""
        own, _, targets = self._make_baseline()
        env_deep = EnvironmentState(water_depth=50.0)
        env_shallow = EnvironmentState(water_depth=15.0)

        r_deep = compute_tube_radius(own, 180, 30, 10.5, targets, env_deep,
                                      use_adaptive=False)
        r_shallow = compute_tube_radius(own, 180, 30, 10.5, targets, env_shallow,
                                         use_adaptive=False)
        assert r_shallow["components"]["shallow_water"] >= r_deep["components"]["shallow_water"]

    def test_ship_interaction_increases_with_close_target(self):
        """Close target → ship interaction component increases."""
        own = VesselState(x=0, y=0, psi=0, u=5, v=0, r=0)
        env = EnvironmentState(water_depth=50.0)

        tgt_far = TargetShip(
            mmsi="T_far", state=VesselState(x=5000, y=0, psi=np.pi, u=3, v=0, r=0),
            length=200, beam=32, draught=11,
        )
        tgt_close = TargetShip(
            mmsi="T_close", state=VesselState(x=80, y=5, psi=0, u=4, v=0, r=0),
            length=200, beam=32, draught=11,
        )

        r_far = compute_tube_radius(own, 180, 30, 10, [tgt_far], env,
                                     use_adaptive=False)
        r_close = compute_tube_radius(own, 180, 30, 10, [tgt_close], env,
                                       use_adaptive=False)
        assert r_close["components"]["ship_interaction"] >= r_far["components"]["ship_interaction"]

    def test_all_components_non_negative(self):
        """Every tube component is >= 0."""
        own = VesselState(x=0, y=0, psi=0, u=7, v=0, r=0)
        tgt = TargetShip(
            mmsi="T001", state=VesselState(x=100, y=10, psi=np.pi, u=5, v=0, r=0),
            length=200, beam=32, draught=11,
        )
        env = EnvironmentState(water_depth=20.0, current_x=0, current_y=1.0,
                                bank_distance_left=80, bank_distance_right=200)

        result = compute_tube_radius(own, 180, 30, 10, [tgt], env,
                                      use_adaptive=False)
        for key, val in result["components"].items():
            assert val >= 0, f"component '{key}' is negative: {val}"

    def test_radius_equals_sum_of_components(self):
        """Total radius == sum of all named components."""
        own = VesselState(x=0, y=0, psi=0, u=6, v=0, r=0)
        tgt = TargetShip(
            mmsi="T001", state=VesselState(x=150, y=20, psi=np.pi, u=5, v=0, r=0),
            length=200, beam=32, draught=11,
        )
        env = EnvironmentState(water_depth=18.0, current_x=0.5, current_y=0.5,
                                bank_distance_left=60, bank_distance_right=250)

        result = compute_tube_radius(own, 180, 30, 10, [tgt], env,
                                      use_adaptive=False)
        comp_sum = sum(result["components"].values())
        assert result["radius"] == pytest.approx(comp_sum, rel=1e-6)

    def test_metadata_has_confidence(self):
        """Tube metadata includes confidence_level and method."""
        own, env, targets = self._make_baseline()
        result = compute_tube_radius(own, 180, 30, 10, targets, env)
        assert result["metadata"]["confidence_level"] == 0.95
        assert result["metadata"]["method"] == "additive_conservative"
        assert "clipped" in result["metadata"]

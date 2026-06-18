"""Tests for dynamic ship domain computation."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest

from ta_mrc_pe_cc_tube_mpc.risk.dynamic_ship_domain import DynamicShipDomain
from ta_mrc_pe_cc_tube_mpc.types import EnvironmentState, TargetShip, VesselParams, VesselState


def make_own_state(u=6.0):
    return VesselState(x=0, y=0, psi=0, u=u, v=0, r=0)


def make_target_state(x=500, y=0, u=5.0, psi=3.14):
    return VesselState(x=x, y=y, psi=psi, u=u, v=0, r=0)


def make_target(vessel_type="cargo", length=180, beam=30, draught=10.5,
                u=5.0, x=500, y=0, psi=3.14, nav_status="underway",
                ais_delay=0.0, dropout_prob=0.0):
    state = make_target_state(x=x, y=y, u=u, psi=psi)
    return TargetShip(
        mmsi="TEST001", state=state,
        length=length, beam=beam, draught=draught,
        vessel_type=vessel_type, nav_status=nav_status,
        ais_quality={"delay": ais_delay, "dropout_prob": dropout_prob,
                     "position_noise_std": 10.0},
    )


class TestDynamicShipDomain:
    """Test dynamic ship domain additive formulation."""

    def setup_method(self):
        self.domain = DynamicShipDomain()
        self.own_params = VesselParams()
        self.env = EnvironmentState()

    def test_larger_ship_larger_domain(self):
        """Larger target ship should produce larger safety domain."""
        own = make_own_state()
        tgt_small = make_target(length=50, beam=10)
        tgt_large = make_target(length=300, beam=50)

        result_small = self.domain.compute(own, self.own_params, tgt_small, self.env)
        result_large = self.domain.compute(own, self.own_params, tgt_large, self.env)

        assert result_large["total"] > result_small["total"]

    def test_faster_target_larger_domain(self):
        """Faster closing speed should increase domain."""
        own = make_own_state(u=6.0)
        tgt_slow = make_target(u=2.0)  # slow closing
        tgt_fast = make_target(u=12.0)  # fast closing

        result_slow = self.domain.compute(
            own, self.own_params, tgt_slow, self.env,
            encounter_role="head_on",
        )
        result_fast = self.domain.compute(
            own, self.own_params, tgt_fast, self.env,
            encounter_role="head_on",
        )

        assert result_fast["d_speed"] > result_slow["d_speed"]

    def test_vessel_type_weight(self):
        """Ferry, tug-barge, tanker should have different domain weights."""
        own = make_own_state()
        tgt_cargo = make_target(vessel_type="cargo")
        tgt_tanker = make_target(vessel_type="tanker")
        tgt_ferry = make_target(vessel_type="ferry")
        tgt_tug = make_target(vessel_type="tug_barge")

        r_cargo = self.domain.compute(own, self.own_params, tgt_cargo, self.env)
        r_tanker = self.domain.compute(own, self.own_params, tgt_tanker, self.env)
        r_ferry = self.domain.compute(own, self.own_params, tgt_ferry, self.env)
        r_tug = self.domain.compute(own, self.own_params, tgt_tug, self.env)

        # Tanker > cargo due to weight
        assert r_tanker["d_vessel_type"] > r_cargo["d_vessel_type"]
        # Ferry < cargo
        assert r_ferry["d_vessel_type"] < r_cargo["d_vessel_type"]
        # Tug-barge > cargo
        assert r_tug["d_vessel_type"] > r_cargo["d_vessel_type"]

    def test_ais_delay_enlarges_domain(self):
        """AIS delay should increase uncertainty contribution."""
        own = make_own_state()
        tgt = make_target()

        r_no = self.domain.compute(own, self.own_params, tgt, self.env, ais_delay=0.0)
        r_delayed = self.domain.compute(own, self.own_params, tgt, self.env, ais_delay=10.0)

        assert r_delayed["d_unc"] > r_no["d_unc"]

    def test_output_is_length_dimension(self):
        """All domain terms must be in meters (length dimension)."""
        own = make_own_state()
        tgt = make_target()

        result = self.domain.compute(own, self.own_params, tgt, self.env)

        for key, val in result.items():
            if key == "total":
                continue
            assert val >= 0, f"{key} should be non-negative"
        assert result["total"] > 0
        assert isinstance(result["total"], float)

    def test_bank_proximity_affects_domain(self):
        """Closer bank should increase bank safety term."""
        own = make_own_state()

        env_far = EnvironmentState(bank_distance_left=500, bank_distance_right=500)
        env_near = EnvironmentState(bank_distance_left=50, bank_distance_right=50)

        result_far = self.domain.compute(
            own, self.own_params, make_target(), env_far,
            bank_effect_active=True,
        )
        result_near = self.domain.compute(
            own, self.own_params, make_target(), env_near,
            bank_effect_active=True,
        )

        # Near bank should have larger d_bank contribution
        assert result_near["d_bank"] >= result_far["d_bank"]

    def test_ablations_remove_terms(self):
        """Disabling features should zero out corresponding terms."""
        own = make_own_state()
        tgt = make_target()

        full = self.domain.compute(own, self.own_params, tgt, self.env)

        no_speed = self.domain.compute(
            own, self.own_params, tgt, self.env,
            enable_speed_sensitive=False,
        )
        assert no_speed["d_speed"] == 0.0

        no_size = self.domain.compute(
            own, self.own_params, tgt, self.env,
            enable_target_size=False,
        )
        # d_base changes but d_size goes to 0
        assert no_size["d_size"] == 0.0

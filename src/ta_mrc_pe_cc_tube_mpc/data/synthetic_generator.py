"""Synthetic data generator for scenarios, AIS tracks, and environments.

Provides reproducible synthetic data when real AIS/ENC data is unavailable.
"""

from typing import Optional

import numpy as np

from ..constants import KNOTS_TO_MPS
from ..types import EnvironmentState, TargetShip, VesselParams, VesselState


class SyntheticGenerator:
    """Generate synthetic vessel states, environments, and AIS records.

    All random processes accept a seed for reproducibility.
    """

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    def make_target_ship(
        self,
        mmsi: str,
        vessel_type: str = "cargo",
        initial_state: Optional[dict] = None,
        length: float = 180.0,
        beam: float = 30.0,
        draught: float = 10.5,
        nav_status: str = "underway",
        ais_delay: float = 0.0,
        ais_dropout_prob: float = 0.0,
        position_noise_std: float = 10.0,
    ) -> TargetShip:
        """Create a synthetic target ship with optional AIS degradation.

        Args:
            mmsi: Vessel identifier.
            vessel_type: One of cargo, tanker, ferry, tug_barge, small_craft.
            initial_state: Dict with keys x, y, psi, u, v, r.
            length, beam, draught: Dimensions [m].
            nav_status: Navigation status string.
            ais_delay: AIS reporting delay [s].
            ais_dropout_prob: Probability of AIS dropout at each step.
            position_noise_std: AIS position noise standard deviation [m].

        Returns:
            Configured TargetShip.
        """
        if initial_state is None:
            initial_state = {}
        state = VesselState(
            x=initial_state.get("x", 0.0),
            y=initial_state.get("y", 0.0),
            psi=initial_state.get("psi", 0.0),
            u=initial_state.get("u", 0.0),
            v=initial_state.get("v", 0.0),
            r=initial_state.get("r", 0.0),
        )
        speed_kn = state.speed() / KNOTS_TO_MPS
        heading_deg = np.rad2deg(state.psi) % 360.0

        return TargetShip(
            mmsi=mmsi,
            state=state,
            length=length,
            beam=beam,
            draught=draught,
            vessel_type=vessel_type,
            nav_status=nav_status,
            cog=heading_deg,
            sog=speed_kn,
            heading=heading_deg,
            covariance=np.eye(2) * position_noise_std**2,
            ais_quality={
                "delay": ais_delay,
                "dropout_prob": ais_dropout_prob,
                "position_noise_std": position_noise_std,
            },
        )

    def make_environment(
        self,
        water_depth: float = 50.0,
        current_x: float = 0.0,
        current_y: float = 0.0,
        wind_x: float = 0.0,
        wind_y: float = 0.0,
        bank_distance_left: Optional[float] = None,
        bank_distance_right: Optional[float] = None,
        channel_width: Optional[float] = None,
        waterway_id: str = "synthetic",
    ) -> EnvironmentState:
        """Create an environment state."""
        return EnvironmentState(
            water_depth=water_depth,
            current_x=current_x,
            current_y=current_y,
            wind_x=wind_x,
            wind_y=wind_y,
            bank_distance_left=bank_distance_left,
            bank_distance_right=bank_distance_right,
            channel_width=channel_width,
            waterway_id=waterway_id,
        )

    def propagate_target_straight(
        self,
        target: TargetShip,
        dt: float,
        duration: float,
    ) -> list[VesselState]:
        """Generate a straight-line trajectory for a target ship.

        Args:
            target: Initial target ship configuration.
            dt: Time step [s].
            duration: Total duration [s].

        Returns:
            List of VesselState at each time step.
        """
        n_steps = int(duration / dt) + 1
        states = []
        current = target.state
        for k in range(n_steps):
            t = k * dt
            s = VesselState(
                x=current.x + current.u * np.cos(current.psi) * t,
                y=current.y + current.u * np.sin(current.psi) * t,
                psi=current.psi,
                u=current.u,
                v=current.v,
                r=current.r,
                timestamp=t,
            )
            states.append(s)
        return states

    def add_ais_noise(
        self, state: VesselState, std: float = 10.0
    ) -> VesselState:
        """Add Gaussian noise to simulate AIS position error.

        Args:
            state: True vessel state.
            std: Position noise standard deviation [m].

        Returns:
            Noisy VesselState.
        """
        noisy = VesselState(
            x=state.x + self.rng.normal(0, std),
            y=state.y + self.rng.normal(0, std),
            psi=state.psi + self.rng.normal(0, 1.0) * np.pi / 180.0,
            u=state.u + self.rng.normal(0, 0.1),
            v=state.v + self.rng.normal(0, 0.1),
            r=state.r,
            timestamp=state.timestamp,
        )
        return noisy

    def generate_environment_sequence(
        self,
        base_env: EnvironmentState,
        duration: float,
        dt: float,
        current_variation: float = 0.0,
        wind_variation: float = 0.0,
    ) -> list[EnvironmentState]:
        """Generate a time sequence of environment states.

        Args:
            base_env: Base environment state.
            duration: Total duration [s].
            dt: Time step [s].
            current_variation: Std of random current variation [m/s].
            wind_variation: Std of random wind variation [m/s].

        Returns:
            List of EnvironmentStates.
        """
        n_steps = int(duration / dt) + 1
        seq = []
        for k in range(n_steps):
            env = EnvironmentState(
                water_depth=base_env.water_depth,
                current_x=base_env.current_x + self.rng.normal(0, current_variation),
                current_y=base_env.current_y + self.rng.normal(0, current_variation),
                wind_x=base_env.wind_x + self.rng.normal(0, wind_variation),
                wind_y=base_env.wind_y + self.rng.normal(0, wind_variation),
                bank_distance_left=base_env.bank_distance_left,
                bank_distance_right=base_env.bank_distance_right,
                channel_width=base_env.channel_width,
                waterway_id=base_env.waterway_id,
            )
            seq.append(env)
        return seq

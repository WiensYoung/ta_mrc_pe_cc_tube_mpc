"""Ownship vessel model that wraps MMG dynamics and state management."""

import numpy as np

from ..types import ControlCommand, EnvironmentState, VesselParams, VesselState
from .mmg_3dof import MMG3DOFModel


class Vessel:
    """Ownship representation with 3-DOF MMG dynamics.

    Wraps the MMG3DOFModel and manages state, constraints, and
    disturbance accumulation.
    """

    def __init__(self, params: VesselParams, mmg_params: dict = None):
        """
        Args:
            params: Vessel physical parameters.
            mmg_params: Optional MMG model parameter overrides.
        """
        self.params = params
        self.mmg = MMG3DOFModel({
            "length": params.length,
            "beam": params.beam,
            "draught": params.draught,
            "mass": params.mass,
            "Iz": params.Iz,
            "max_rudder": params.max_rudder,
            "max_rudder_rate": params.max_rudder_rate,
            "max_yaw_rate": params.max_yaw_rate,
            "min_turning_radius": params.min_turning_radius_multiplier,
            **(mmg_params or {}),
        })
        self.state = VesselState()
        self.command_history: list[ControlCommand] = []

    def set_state(self, state: VesselState):
        """Set the current vessel state."""
        self.state = state

    def step(
        self,
        command: ControlCommand,
        env: EnvironmentState,
        dt: float,
        disturbance_force: np.ndarray = None,
    ) -> VesselState:
        """Advance dynamics by one time step.

        Args:
            command: Control command.
            env: Environment state.
            dt: Time step [s].
            disturbance_force: External disturbance [X, Y, N] in body frame.

        Returns:
            New vessel state.
        """
        self.state = self.mmg.step(self.state, command, env, dt, disturbance_force)
        self.command_history.append(command)
        return self.state

    def position(self) -> np.ndarray:
        """Current position [x, y]."""
        return self.state.position()

    def velocity_world(self) -> np.ndarray:
        """Current velocity in world frame."""
        return self.state.velocity()

    def speed(self) -> float:
        """Current speed [m/s]."""
        return self.state.speed()

    def heading_deg(self) -> float:
        """Current heading [deg]."""
        return float(np.rad2deg(self.state.psi) % 360.0)

"""Environment model: manages environment state and provides lookups."""

from ..types import EnvironmentState


class Environment:
    """Environment manager for waterway state.

    Wraps EnvironmentState and provides utilities for depth queries,
    bank distance lookups, and waterway zone checks.
    """

    def __init__(self, initial_state: EnvironmentState = None, dt: float = 0.5):
        self.state = initial_state or EnvironmentState()
        self.history: list[EnvironmentState] = []
        self.dt = dt

    def set_state(self, state: EnvironmentState):
        self.state = state

    def get_state(self, t: float, sequence: list[EnvironmentState] = None) -> EnvironmentState:
        """Get environment state at time t from a sequence, or return current."""
        if sequence is not None and len(sequence) > 0:
            idx = min(int(t / self.dt), len(sequence) - 1)
            if 0 <= idx < len(sequence):
                return sequence[idx]
        return self.state

    def depth_to_draught_ratio(self, draught: float = 1.0) -> float:
        """Return h / T (depth to draught ratio)."""
        if self.state.water_depth <= 0:
            return float("inf")
        if draught <= 0:
            return float("inf")
        return self.state.water_depth / draught

    def is_navigable(self, min_depth: float = 1.0) -> bool:
        """Check if the current position has sufficient depth."""
        return self.state.water_depth >= min_depth

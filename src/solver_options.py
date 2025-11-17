from dataclasses import dataclass, field
from typing import Dict, Any


@dataclass
class SolverOptions:
    """A data class to hold all solver configuration options."""

    # Dataclass Fields (The ones you want to initialize automatically)
    limiter_type: str = "barth_jespersen"
    flux_type: str = "roe"
    over_relaxation: float = 1.2
    use_adaptive_dt: bool = True
    cfl: float = 0.5
    dt_initial: float = 0.01
    time_integration_method: str = "rk2"

    # Non-dataclass properties can be stored here using field(init=False)
    t_end: float = field(init=False)
    gamma: float = field(init=False)

    def __post_init__(self):
        # This runs after the fields are set. We use it here just for illustration.
        print(f"SolverOptions correctly read")

    @classmethod
    def from_config(cls, config: Dict[str, Any]):
        """Creates a SolverOptions instance from a configuration dictionary."""

        # 1. Extract non-dataclass parameters (t_end, gamma)
        t_end = config.get("t_end", 0.25)
        gamma = config.get("gamma", 1.4)
        # 2. Get the solver_options sub-dictionary, or use an empty dict
        solver_config = config.get("solver_options", {})

        # 3. Create the instance using dictionary unpacking (The Fix!)
        # Any key in solver_config matching a dataclass field name is set automatically.
        instance = cls(**solver_config)

        # 4. Manually set the non-dataclass properties
        instance.t_end = t_end
        instance.gamma = gamma

        return instance

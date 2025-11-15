from dataclasses import dataclass


@dataclass
class SolverOptions:
    """A data class to hold all solver configuration options."""

    limiter_type: str = "barth_jespersen"
    flux_type: str = "roe"
    over_relaxation: float = 1.2
    use_adaptive_dt: bool = True
    cfl: float = 0.5
    dt_initial: float = 0.01
    time_integration_method: str = "rk2"

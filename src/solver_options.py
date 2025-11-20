from dataclasses import dataclass, field
from typing import Dict, Any


@dataclass
class SolverOptions:
    """A data class to hold all solver configuration options."""

    # Time integration
    time_integration_method: str = "rk2"
    cfl: float = 0.5
    use_adaptive_dt: bool = True
    dt_initial: float = 1.0e-4

    # Spatial discretization
    flux_type: str = "roe"
    limiter_type: str = "minmod"
    gradient_over_relaxation: float = 1.0

    # I/O
    output_format: str = "vtk"
    output_interval: int = 100
    output_filename_prefix: str = "solution"

    # Simulation settings that are not solver-specific but useful to have here
    mesh_file: str = field(init=False)
    t_end: float = field(init=False)
    equation: str = field(init=False)
    case: str = field(init=False)
    gamma: float = field(init=False)
    g: float = field(init=False)


    @classmethod
    def from_config(cls, config: Dict[str, Any]):
        """Creates a SolverOptions instance from a nested configuration dictionary."""
        input_config = config.get("input", {})
        sim_config = config.get("simulation", {})
        physics_config = config.get("physics", {})
        solver_config = config.get("solver", {})
        time_config = solver_config.get("time_integration", {})
        spatial_config = solver_config.get("spatial", {})
        output_config = config.get("output", {})

        # Create an instance with values from the config, falling back to defaults
        instance = cls(
            time_integration_method=time_config.get("method", "rk2"),
            cfl=time_config.get("cfl", 0.5),
            use_adaptive_dt=time_config.get("use_adaptive_dt", True),
            dt_initial=time_config.get("dt_initial", 1.0e-4),
            flux_type=spatial_config.get("flux_type", "roe"),
            limiter_type=spatial_config.get("limiter_type", "minmod"),
            gradient_over_relaxation=spatial_config.get("gradient_over_relaxation", 1.0),
            output_format=output_config.get("format", "vtk"),
            output_interval=output_config.get("interval", 100),
            output_filename_prefix=output_config.get("filename_prefix", "solution"),
        )

        # Manually set non-solver-specific properties
        instance.mesh_file = input_config.get("mesh_file", "data/euler_mesh.msh")
        instance.t_end = sim_config.get("t_end", 0.25)
        instance.equation = sim_config.get("equation", "euler")
        instance.case = sim_config.get("case", "riemann")
        
        instance.gamma = physics_config.get("euler", {}).get("gamma", 1.4)
        instance.g = physics_config.get("shallow_water", {}).get("g", 9.806)

        return instance

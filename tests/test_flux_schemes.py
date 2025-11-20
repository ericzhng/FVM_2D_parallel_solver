import unittest
import numpy as np
import sys
import os

# Disable Numba JIT compilation for testing
os.environ["NUMBA_DISABLE_JIT"] = "1"

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.flux_schemes import RiemannSolverFlux, CentralDifferenceFlux
from src.equation_euler import EulerEquations


class TestFluxSchemes(unittest.TestCase):
    """
    Test suite for the new FluxScheme implementations.
    """

    def setUp(self):
        """
        Set up common data for the tests.
        """
        self.q_left = np.array([1.0, 0.1, 0.0, 2.5]).astype(np.float64)
        self.q_right = np.array([0.125, 0.0, 0.0, 0.25]).astype(np.float64)
        self.gamma = 1.40
        self.normal = np.array([0.0, 1.0])
        self.euler_equation = EulerEquations(self.gamma)
        self.nvars = self.q_left.shape[0]

    def test_riemann_solver_flux_roe(self):
        """
        Test RiemannSolverFlux with Roe scheme.
        """
        roe_flux_scheme = RiemannSolverFlux("roe")
        flux = roe_flux_scheme.calculate_numerical_flux(
            self.q_left, self.q_right, self.normal, self.euler_equation
        )
        self.assertIsInstance(flux, np.ndarray)
        self.assertEqual(flux.shape, (self.nvars,))
        # Further assertions could compare against known good values if available

    def test_riemann_solver_flux_hllc(self):
        """
        Test RiemannSolverFlux with HLLC scheme.
        """
        hllc_flux_scheme = RiemannSolverFlux("hllc")
        flux = hllc_flux_scheme.calculate_numerical_flux(
            self.q_left, self.q_right, self.normal, self.euler_equation
        )
        self.assertIsInstance(flux, np.ndarray)
        self.assertEqual(flux.shape, (self.nvars,))
        # Further assertions could compare against known good values if available

    def test_central_difference_flux(self):
        """
        Test CentralDifferenceFlux scheme.
        """
        cd_flux_scheme = CentralDifferenceFlux()
        flux = cd_flux_scheme.calculate_numerical_flux(
            self.q_left, self.q_right, self.normal, self.euler_equation
        )
        self.assertIsInstance(flux, np.ndarray)
        self.assertEqual(flux.shape, (self.nvars,))
        # Further assertions could compare against known good values if available

    def test_unsupported_flux_type(self):
        """
        Test that RiemannSolverFlux raises ValueError for unsupported types.
        """
        with self.assertRaises(ValueError):
            RiemannSolverFlux("unsupported_type")


if __name__ == "__main__":
    unittest.main()

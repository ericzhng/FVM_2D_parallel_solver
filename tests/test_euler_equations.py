import unittest
import numpy as np
import sys
import os

# Disable Numba JIT compilation for testing
os.environ["NUMBA_DISABLE_JIT"] = "1"

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.euler_equations import EulerEquations


class TestEulerEquations(unittest.TestCase):
    """
    Test suite for the euler_equations module.
    """

    def setUp(self):
        """
        Set up common data for the tests.
        """
        # Example data for a single cell/interface
        self.q_left = np.array([1.0, 0.1, 0.0, 2.5]).astype(np.float64)
        self.q_right = np.array([0.125, 0.0, 0.0, 0.25]).astype(np.float64)
        self.gamma = 1.40
        self.normal = np.array([0.0, 1.0])
        self.euler_equation = EulerEquations(self.gamma)

    def test_roe_flux(self):
        """
        Test the Roe flux function.
        """
        # This is a placeholder for the actual test.
        # You would need to know the expected output for the given inputs.
        # For now, we'll just check if the function runs without errors.
        flux = self.euler_equation.roe_flux(self.q_left, self.q_right, self.normal)
        print("reference ROE flux:")
        print(flux)

        flux = self.euler_equation.roe_flux(self.q_left, self.q_right, self.normal)
        print("changed ROE flux:")
        print(flux)

    def test_hllc_flux(self):
        """
        Test the HLLC flux function.
        """
        flux = self.euler_equation.hllc_flux(self.q_left, self.q_right, self.normal)
        print("reference HLLC flux:")
        print(flux)

        flux = self.euler_equation.hllc_flux(self.q_left, self.q_right, self.normal)
        print("changed HLLC flux:")
        print(flux)


if __name__ == "__main__":
    unittest.main()

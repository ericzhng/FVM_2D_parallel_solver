import numpy as np
from numba import float64
from numba.experimental import jitclass

from src.physics_model import PhysicsModel


@jitclass([("gamma", float64)])
class EulerEquations(PhysicsModel):
    """
    Represents the 2D Euler equations for compressible fluid flow.

    This class provides the specific implementation for the Euler equations,
    including the conversion between conservative and primitive variables,
    flux calculation (Roe and HLLC), and wave speed estimation.

    Attributes:
        gamma (float): The ratio of specific heats (adiabatic index).
    """

    def __init__(self, gamma: float = 1.4):
        """
        Initializes the EulerEquations object.

        Args:
            gamma (float, optional): The ratio of specific heats. Defaults to 1.4.
        """
        self.gamma = gamma

    def _cons_to_prim(self, U):
        """
        Converts a single conservative state vector to primitive variables.

        Args:
            U (np.ndarray): Conservative state vector [rho, rho*u, rho*v, E].

        Returns:
            np.ndarray: Primitive state vector [rho, u, v, p].
        """
        rho, rho_u, rho_v, E = U
        # Floor density to prevent division by zero or negative pressure
        rho = np.maximum(rho, 1e-9)
        u = rho_u / rho
        v = rho_v / rho
        # Calculate pressure from total energy
        p = (self.gamma - 1) * (E - 0.5 * rho * (u**2 + v**2))
        return np.array([rho, u, v, p])

    def _prim_to_cons(self, P):
        """
        Converts a single primitive state vector to conservative variables.

        Args:
            P (np.ndarray): Primitive state vector [rho, u, v, p].

        Returns:
            np.ndarray: Conservative state vector [rho, rho*u, rho*v, E].
        """
        rho, u, v, p = P
        rho_u = rho * u
        rho_v = rho * v
        # Total energy E is the sum of internal and kinetic energy
        E = p / (self.gamma - 1) + 0.5 * rho * (u**2 + v**2)
        return np.array([rho, rho_u, rho_v, E])

    def _compute_flux(self, U, normal):
        """
        Calculates the physical flux across a face with a given normal.

        Args:
            U (np.ndarray): Conservative state vector [rho, rho*u, rho*v, E].
            normal (np.ndarray): Normal vector of the face.

        Returns:
            np.ndarray: The flux vector normal to the face.
        """
        rho, u, v, p = self._cons_to_prim(U)
        # Normal velocity
        vn = u * normal[0] + v * normal[1]
        # Total enthalpy H = (E + p) / rho
        H = (U[3] + p) / rho

        # Normal flux components
        F = np.array(
            [
                rho * vn,
                rho * vn * u + p * normal[0],
                rho * vn * v + p * normal[1],
                rho * vn * H,
            ]
        )
        return F

    def max_eigenvalue(self, U):
        """
        Calculates the maximum wave speed (eigenvalue) for a cell.
        This is used for determining the stable time step (CFL condition).
        """
        rho, u, v, p = self._cons_to_prim(U)
        # Speed of sound 'a'
        a = np.sqrt(self.gamma * p / rho)
        # Max eigenvalue = |v| + a
        return np.sqrt(u**2 + v**2) + a

    def apply_transmissive_bc(self, U_inside: np.ndarray) -> np.ndarray:
        """
        Applies a transmissive (zero-gradient) boundary condition.
        """
        return U_inside

    def apply_inlet_bc(self, bc_value: np.ndarray) -> np.ndarray:
        """
        Applies an inlet boundary condition with specified primitive values.
        """
        rho_bc, u_bc, v_bc, p_bc = bc_value
        P_ghost = np.array([rho_bc, u_bc, v_bc, p_bc])
        return self._prim_to_cons(P_ghost)

    def apply_outlet_bc(self, U_inside: np.ndarray, bc_value: np.ndarray) -> np.ndarray:
        """
        Applies an outlet boundary condition, typically with a specified pressure.
        """
        P_inside = self._cons_to_prim(U_inside)
        rho_in, u_in, v_in, _ = P_inside
        p_ghost = bc_value[3]  # Pressure is stored in the 4th element
        P_ghost = np.array([rho_in, u_in, v_in, p_ghost])
        return self._prim_to_cons(P_ghost)

    def apply_wall_bc(
        self, U_inside: np.ndarray, normal: np.ndarray, bc_value: np.ndarray
    ) -> np.ndarray:
        """
        Applies a wall boundary condition (slip or no-slip).
        """
        P_inside = self._cons_to_prim(U_inside)
        rho_in, u_in, v_in, p_in = P_inside

        rho_ghost = rho_in
        p_ghost = p_in

        vn_in = u_in * normal[0] + v_in * normal[1]
        vt_in = u_in * -normal[1] + v_in * normal[0]

        vn_ghost = -vn_in
        is_slip = bc_value[0] == 1.0
        if is_slip:
            vt_ghost = vt_in
        else:
            vt_ghost = 0.0

        u_ghost = vn_ghost * normal[0] - vt_ghost * normal[1]
        v_ghost = vn_ghost * normal[1] + vt_ghost * normal[0]

        P_ghost = np.array([rho_ghost, u_ghost, v_ghost, p_ghost])
        return self._prim_to_cons(P_ghost)

    def hllc_flux(self, U_L, U_R, normal) -> np.ndarray:
        """
        Computes the numerical flux using the HLLC (Harten-Lax-van Leer-Contact) Riemann solver.

        The HLLC solver is a modification of the HLL solver that restores the
        contact wave and shear waves, providing better resolution for these phenomena.

        Args:
            U_L (np.ndarray): Conservative state vector of the left cell.
            U_R (np.ndarray): Conservative state vector of the right cell.
            normal (np.ndarray): Normal vector of the face.

        Returns:
            np.ndarray: The HLLC numerical flux across the face.
        """
        nx, ny = normal
        tx, ty = -ny, nx  # Tangent vector

        # --- Left State ---
        rL, uL, vL, pL = self._cons_to_prim(U_L)
        vnL = uL * nx + vL * ny  # Normal velocity
        vtL = uL * tx + vL * ty  # Tangential velocity
        aL = np.sqrt(self.gamma * pL / rL)  # Speed of sound
        FL = self._compute_flux(U_L, normal)

        # --- Right State ---
        rR, uR, vR, pR = self._cons_to_prim(U_R)
        vnR = uR * nx + vR * ny  # Normal velocity
        vtR = uR * tx + vR * ty  # Tangential velocity
        aR = np.sqrt(self.gamma * pR / rR)  # Speed of sound
        FR = self._compute_flux(U_R, normal)

        # --- Wave Speed Estimates (Roe Averages) ---
        # Roe-averaged density
        r_roe = np.sqrt(rL * rR)
        # Roe-averaged velocities
        u_roe = (np.sqrt(rL) * uL + np.sqrt(rR) * uR) / (np.sqrt(rL) + np.sqrt(rR))
        v_roe = (np.sqrt(rL) * vL + np.sqrt(rR) * vR) / (np.sqrt(rL) + np.sqrt(rR))
        # Roe-averaged normal velocity
        vn_roe = u_roe * nx + v_roe * ny
        # Roe-averaged speed of sound
        hL = (U_L[3] + pL) / rL  # Total enthalpy left
        hR = (U_R[3] + pR) / rR  # Total enthalpy right
        h_roe = (np.sqrt(rL) * hL + np.sqrt(rR) * hR) / (np.sqrt(rL) + np.sqrt(rR))
        a_roe = np.sqrt((self.gamma - 1) * (h_roe - 0.5 * (u_roe**2 + v_roe**2)))

        # --- Davis-Einfeldt Wave Speed Estimates ---
        SL = np.minimum(vnL - aL, vn_roe - a_roe)
        SR = np.maximum(vnR + aR, vn_roe + a_roe)

        # --- Middle Wave Speed (Contact Wave) ---
        SM = (pR - pL + rL * vnL * (SL - vnL) - rR * vnR * (SR - vnR)) / (
            rL * (SL - vnL) - rR * (SR - vnR)
        )

        # --- HLLC Flux Calculation ---
        if 0.0 <= SL:
            # All waves move to the right
            return FL
        elif SL < 0.0 <= SM:
            # Left-going shock/rarefaction, contact wave to the right
            # U_star_L = rho_L * (SL - vnL) / (SL - SM) * [1, SM*nx - vtL*ny, SM*ny + vtL*nx, E_star_L]
            U_star_L = (
                rL
                * (SL - vnL)
                / (SL - SM)
                * np.array(
                    [
                        1.0,
                        SM * nx - vtL * ny,  # Corrected velocity
                        SM * ny + vtL * nx,  # Corrected velocity
                        (U_L[3] / rL) + (SM - vnL) * (SM + pL / (rL * (SL - vnL))),
                    ]
                )
            )
            return FL + SL * (U_star_L - U_L)
        elif SM < 0.0 < SR:
            # Contact wave to the left, right-going shock/rarefaction
            # U_star_R = rho_R * (SR - vnR) / (SR - SM) * [1, SM*nx - vtR*ny, SM*ny + vtR*nx, E_star_R]
            U_star_R = (
                rR
                * (SR - vnR)
                / (SR - SM)
                * np.array(
                    [
                        1.0,
                        SM * nx - vtR * ny,  # Corrected velocity
                        SM * ny + vtR * nx,  # Corrected velocity
                        (U_R[3] / rR) + (SM - vnR) * (SM + pR / (rR * (SR - vnR))),
                    ]
                )
            )
            return FR + SR * (U_star_R - U_R)
        else:  # SR <= 0
            # All waves move to the left
            return FR

    def roe_flux(self, U_L, U_R, normal) -> np.ndarray:
        """
        Computes the numerical flux using the Roe approximate Riemann solver.

        The Roe solver is known for its high resolution of contact discontinuities
        but can be susceptible to expansion shocks without an entropy fix.

        Args:
            U_L (np.ndarray): Conservative state vector of the left cell.
            U_R (np.ndarray): Conservative state vector of the right cell.
            normal (np.ndarray): Normal vector of the face.

        Returns:
            np.ndarray: The Roe numerical flux across the face.
        """
        nx, ny = normal
        tx, ty = -ny, nx  # Tangent vector

        # --- Left and Right States ---
        rL, uL, vL, pL = self._cons_to_prim(U_L)
        vnL = uL * nx + vL * ny
        vtL = uL * tx + vL * ty
        HL = (U_L[3] + pL) / rL
        FL = self._compute_flux(U_L, normal)

        rR, uR, vR, pR = self._cons_to_prim(U_R)
        vnR = uR * nx + vR * ny
        vtR = uR * tx + vR * ty
        HR = (U_R[3] + pR) / rR
        FR = self._compute_flux(U_R, normal)

        # --- Roe Averages ---
        sqrt_rL = np.sqrt(rL)
        sqrt_rR = np.sqrt(rR)
        r = sqrt_rL * sqrt_rR

        u = (sqrt_rL * uL + sqrt_rR * uR) / (sqrt_rL + sqrt_rR)
        v = (sqrt_rL * vL + sqrt_rR * vR) / (sqrt_rL + sqrt_rR)
        H = (sqrt_rL * HL + sqrt_rR * HR) / (sqrt_rL + sqrt_rR)
        a_sq = (self.gamma - 1) * (H - 0.5 * (u**2 + v**2))
        a = np.sqrt(np.maximum(a_sq, 1e-6))  # Ensure non-negativity
        vn = u * nx + v * ny

        # --- Wave Strengths (Jump in characteristics) ---
        dr = rR - rL
        dp = pR - pL
        dvn = vnR - vnL
        dvt = vtR - vtL

        # dV = [l1*dV, l2*dV, l3*dV, l4*dV]
        dV = np.array(
            [
                (dp - r * a * dvn) / (2 * a**2),
                r * dvt,
                dr - dp / a**2,
                (dp + r * a * dvn) / (2 * a**2),
            ]
        )

        # --- Wave Speeds (Eigenvalues) ---
        ws = np.array([abs(vn - a), abs(vn), abs(vn), abs(vn + a)])

        # --- Harten's Entropy Fix ---
        # This fix is applied to prevent non-physical expansion shocks
        # by adding dissipation in the case of vanishing pressure jumps.
        delta = 0.1 * a  # Entropy fix parameter
        for i in range(4):
            if i == 0 or i == 3:
                if ws[i] < delta:
                    ws[i] = (ws[i] ** 2 + delta**2) / (2 * delta)
                else:
                    ws[i] = ws[i]

        # --- Right Eigenvectors Matrix ---
        # The columns of this matrix are the right eigenvectors of the Roe matrix.
        Rv = np.array(
            [
                [1.0, 0.0, 1.0, 1.0],
                [u - a * nx, -a * ny, u, u + a * nx],
                [v - a * ny, a * nx, v, v + a * ny],
                [H - vn * a, -(u * ny - v * nx) * a, 0.5 * (u**2 + v**2), H + vn * a],
            ]
        )

        # --- Roe Flux ---
        # F_roe = 0.5 * (F_L + F_R) - 0.5 * sum(ws_i * dV_i * R_i)
        dissipation = Rv @ (ws * dV)
        flux = 0.5 * (FL + FR - dissipation)

        return flux

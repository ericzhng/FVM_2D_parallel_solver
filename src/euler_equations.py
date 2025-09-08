import numpy as np
from numba import float64
from numba.experimental import jitclass

spec = [
    ("gamma", float64),
]


@jitclass(spec)
class EulerEquations:
    """
    Represents the 2D Euler equations for compressible fluid flow.

    This class provides the specific implementation for the Euler equations,
    including the conversion between conservative and primitive variables,
    flux calculation (Roe and HLLC), and wave speed estimation.

    Attributes:
        gamma (float): The ratio of specific heats (adiabatic index).
    """

    def __init__(self, gamma=1.4):
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
        rho = max(rho, 1e-9)
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

    def _apply_wall_bc(self, U_inside, normal):
        """
        Applies a solid wall (reflective) boundary condition.

        This condition reflects the velocity normal to the wall while keeping the
        tangential velocity and thermodynamic properties (pressure, density) the same.

        Args:
            U_inside (np.ndarray): State vector of the interior cell.
            normal (np.ndarray): Normal vector of the boundary face.

        Returns:
            np.ndarray: The state vector of the ghost cell.
        """
        rho, u, v, p = self._cons_to_prim(U_inside)

        # Decompose velocity into normal and tangential components
        vn = u * normal[0] + v * normal[1]
        vt = u * -normal[1] + v * normal[0]

        # Reflect the normal velocity, keep tangential velocity
        vn_ghost = -vn
        vt_ghost = vt

        # Recompose the ghost velocity vector from the new normal and tangential components
        u_ghost = vn_ghost * normal[0] - vt_ghost * normal[1]
        v_ghost = vn_ghost * normal[1] + vt_ghost * normal[0]

        # Create the primitive state for the ghost cell
        P_ghost = np.array([rho, u_ghost, v_ghost, p])

        # Convert back to conservative variables
        return self._prim_to_cons(P_ghost)

    def apply_boundary_condition(self, U_inside, normal, bc_type, bc_value):
        # "inlet": 1
        # "outlet": 2
        # "wall": 3
        # "transmissive": 4
        if bc_type == 1:
            U_inside[1] = bc_value[0]
            U_inside[2] = bc_value[1]
            return U_inside
        elif bc_type == 2:
            U_inside[0] = bc_value[0]
            return U_inside
        elif bc_type == 3:
            return self._apply_wall_bc(U_inside, normal)
        elif bc_type == 4:
            return U_inside
        else:
            raise ValueError("Invalid boundary condition type")

    def hllc_flux(self, U_L, U_R, normal):
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
        SL = min(vnL - aL, vn_roe - a_roe)
        SR = max(vnR + aR, vn_roe + a_roe)

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

    def roe_flux(self, U_L, U_R, normal):
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
        a = np.sqrt(max(a_sq, 1e-6))  # Ensure non-negativity
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
        ws[0] = (ws[0] * ws[0] / delta + delta) / 2 if ws[0] < delta else ws[0]
        ws[3] = (ws[3] * ws[3] / delta + delta) / 2 if ws[3] < delta else ws[3]

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

    def hllc_flux_change(self, U_L, U_R, normal):
        """
        NOT WORKING YET
        Computes the numerical flux using the HLLC (Harten-Lax-van Leer-Contact) Riemann solver.
        Based on "Riemann Solvers and Numerical Methods for Fluid Dynamics" by Eleuterio F. Toro.

        Args:
            U_L (np.ndarray): Conservative state vector of the left cell.
            U_R (np.ndarray): Conservative state vector of the right cell.
            normal (np.ndarray): Normal vector of the face.

        Returns:
            np.ndarray: The HLLC numerical flux across the face.
        """
        nx, ny = normal
        # Tangential vector (rotated 90 degrees clockwise from normal)
        tx, ty = ny, -nx

        # --- Left State ---
        rL, uL, vL, pL = self._cons_to_prim(U_L)
        vnL = uL * nx + vL * ny  # Normal velocity
        vtL = uL * tx + vL * ty  # Tangential velocity
        aL = np.sqrt(self.gamma * pL / rL)  # Speed of sound
        EL = U_L[3]
        FL = self._compute_flux(U_L, normal)

        # --- Right State ---
        rR, uR, vR, pR = self._cons_to_prim(U_R)
        vnR = uR * nx + vR * ny  # Normal velocity
        vtR = uR * tx + vR * ty  # Tangential velocity
        aR = np.sqrt(self.gamma * pR / rR)  # Speed of sound
        ER = U_R[3]
        FR = self._compute_flux(U_R, normal)

        # --- Roe Averages (for wave speed estimates) ---
        sqrt_rL = np.sqrt(rL)
        sqrt_rR = np.sqrt(rR)
        r_roe = sqrt_rL * sqrt_rR
        u_roe = (sqrt_rL * uL + sqrt_rR * uR) / (sqrt_rL + sqrt_rR)
        v_roe = (sqrt_rL * vL + sqrt_rR * vR) / (sqrt_rL + sqrt_rR)
        vn_roe = u_roe * nx + v_roe * ny
        HL = (EL + pL) / rL
        HR = (ER + pR) / rR
        H_roe = (sqrt_rL * HL + sqrt_rR * HR) / (sqrt_rL + sqrt_rR)
        a_roe = np.sqrt((self.gamma - 1) * (H_roe - 0.5 * (u_roe**2 + v_roe**2)))

        # Compute guess pressure from PVRS Riemann solver
        PPV = max(
            0, 0.5 * (pL + pR) + 0.5 * (vnL - vnR) * (0.25 * (rL + rR) * (aL + aR))
        )
        pmin = min(pL, pR)
        pmax = max(pL, pR)
        Qmax = pmax / pmin
        Quser = 2.0  # <--- parameter manually set (I don't like this!)

        if (Qmax <= Quser) and (pmin <= PPV) and (PPV <= pmax):
            # Select PRVS Riemann solver
            pM = PPV
        else:
            if PPV < pmin:
                # Select Two-Rarefaction Riemann solver
                PQ = (pL / pR) ** ((self.gamma - 1.0) / (2.0 * self.gamma))
                uM = (PQ * vnL / aL + vnR / aR + 2 / (self.gamma - 1) * (PQ - 1.0)) / (
                    PQ / aL + 1.0 / aR
                )
                PTL = 1 + (self.gamma - 1) / 2.0 * (vnL - uM) / aL
                PTR = 1 + (self.gamma - 1) / 2.0 * (uM - vnR) / aR
                pM = 0.5 * (
                    pL * PTL ** (2 * self.gamma / (self.gamma - 1))
                    + pR * PTR ** (2 * self.gamma / (self.gamma - 1))
                )
            else:
                # Use Two-Shock Riemann solver with PVRS as estimate
                GEL = np.sqrt(
                    (2 / (self.gamma + 1) / rL)
                    / ((self.gamma - 1) / (self.gamma + 1) * pL + PPV)
                )
                GER = np.sqrt(
                    (2 / (self.gamma + 1) / rR)
                    / ((self.gamma - 1) / (self.gamma + 1) * pR + PPV)
                )
                pM = (GEL * pL + GER * pR - (vnR - vnL)) / (GEL + GER)

        # Estimate wave speeds: SL, SR and SM (Toro, 1994)
        if pM > pL:
            zL = np.sqrt(1 + (self.gamma + 1) / (2 * self.gamma) * (pM / pL - 1))
        else:
            zL = 1

        if pM > pR:
            zR = np.sqrt(1 + (self.gamma + 1) / (2 * self.gamma) * (pM / pR - 1))
        else:
            zR = 1

        # Wave speeds
        SL = vnL - aL * zL
        SR = vnR + aR * zR
        # Contact wave speed
        SM = (pL - pR + rR * vnR * (SR - vnR) - rL * vnL * (SL - vnL)) / (
            rR * (SR - vnR) - rL * (SL - vnL)
        )

        # --- HLLC Flux Calculation ---
        if 0 <= SL:
            # All waves move to the right
            return FL
        elif SL < 0 <= SM:
            # Left-going shock/rarefaction, contact wave to the right
            # U_star_L state
            r_star_L = rL * (SL - vnL) / (SL - SM)
            u_star_L = SM * nx + vtL * tx
            v_star_L = SM * ny + vtL * ty
            E_star_L = r_star_L * (
                (EL / rL) + (SM - vnL) * (SM + pL / (rL * (SL - vnL)))
            )
            U_star_L = np.array(
                [r_star_L, r_star_L * u_star_L, r_star_L * v_star_L, E_star_L]
            )
            return FL + SL * (U_star_L - U_L)
        elif SM < 0 < SR:
            # Contact wave to the left, right-going shock/rarefaction
            # U_star_R state
            r_star_R = rR * (SR - vnR) / (SR - SM)
            u_star_R = SM * nx + vtR * tx
            v_star_R = SM * ny + vtR * ty
            E_star_R = r_star_R * (
                (ER / rR) + (SM - vnR) * (SM + pR / (rR * (SR - vnR)))
            )
            U_star_R = np.array(
                [r_star_R, r_star_R * u_star_R, r_star_R * v_star_R, E_star_R]
            )
            return FR + SR * (U_star_R - U_R)
        else:  # SR <= 0
            # All waves move to the left
            return FR

    def roe_flux_change(self, U_L, U_R, normal):
        """
        WORKING
        Computes the numerical flux using the Roe approximate Riemann solver.
        Based on "Riemann Solvers and Numerical Methods for Fluid Dynamics" by Eleuterio F. Toro.

        Args:
            U_L (np.ndarray): Conservative state vector of the left cell.
            U_R (np.ndarray): Conservative state vector of the right cell.
            normal (np.ndarray): Normal vector of the face.

        Returns:
            np.ndarray: The Roe numerical flux across the face.
        """
        nx, ny = normal

        # --- Left and Right States ---
        rL, uL, vL, pL = self._cons_to_prim(U_L)
        EL = U_L[3]
        FL = self._compute_flux(U_L, normal)

        rR, uR, vR, pR = self._cons_to_prim(U_R)
        ER = U_R[3]
        FR = self._compute_flux(U_R, normal)

        # --- Roe Averages ---
        sqrt_rL = np.sqrt(rL)
        sqrt_rR = np.sqrt(rR)
        r_avg = sqrt_rL * sqrt_rR
        u_avg = (sqrt_rL * uL + sqrt_rR * uR) / (sqrt_rL + sqrt_rR)
        v_avg = (sqrt_rL * vL + sqrt_rR * vR) / (sqrt_rL + sqrt_rR)
        H_avg = (sqrt_rL * (EL + pL) / rL + sqrt_rR * (ER + pR) / rR) / (
            sqrt_rL + sqrt_rR
        )
        # a_avg = np.sqrt((self.gamma - 1) * (H_avg - 0.5 * (u_avg**2 + v_avg**2)))
        a_avg = np.sqrt(
            max(1e-9, (self.gamma - 1) * (H_avg - 0.5 * (u_avg**2 + v_avg**2)))
        )
        vn_avg = u_avg * nx + v_avg * ny

        # --- Eigenvalues (Wave Speeds) ---
        lambda1 = vn_avg - a_avg
        lambda2 = vn_avg
        lambda3 = vn_avg
        lambda4 = vn_avg + a_avg
        ws = np.array([lambda1, lambda2, lambda3, lambda4])

        # --- Entropy Fix (Harten's entropy fix) ---
        delta = 0.1 * a_avg
        for i in [0, 3]:
            if abs(ws[i]) < delta:
                ws[i] = (ws[i] ** 2 + delta**2) / (2 * delta)
        ws = np.abs(ws)  # Use absolute values for dissipation

        # --- Jump in Conservative Variables ---
        dU = U_R - U_L

        # --- Right Eigenvectors (Roe matrix for 2D Euler) ---
        # These are the columns of the matrix.
        # Based on Toro, Chapter 10, Section 10.2.2
        # R1: (1, u-a*nx, v-a*ny, H-vn*a)
        R1 = np.array(
            [1.0, u_avg - a_avg * nx, v_avg - a_avg * ny, H_avg - vn_avg * a_avg]
        )

        # R2: (1, u, v, 0.5*(u^2+v^2)) - (1, u+a*nx, v+a*ny, H+vn*a)
        # This is for the contact discontinuity, related to tangential velocity.
        # The original code's R2 and R3 were for 1D.
        # For 2D, the second and third waves are shear waves.
        # R2 and R3 are related to the tangential components of velocity.
        # R2: (0, -ny, nx, -u*ny + v*nx)
        R2 = a_avg * np.array([0.0, -ny, nx, -u_avg * ny + v_avg * nx])

        # R3: (0, nx, ny, u*nx + v*ny) - This is not standard for the third wave.
        # The third wave is also a shear wave, orthogonal to the second.
        # R3: (1, u, v, 0.5*(u^2+v^2)) - (1, u, v, H)
        # A common choice for the third eigenvector is related to the pressure/density jump.
        # Let's use a simpler form for the third eigenvector, related to the density jump.
        # R3: (1, u, v, 0.5*(u^2+v^2))
        R3 = np.array([1.0, u_avg, v_avg, 0.5 * (u_avg**2 + v_avg**2)])

        # R4: (1, u+a*nx, v+a*ny, H+vn*a)
        R4 = np.array(
            [1.0, u_avg + a_avg * nx, v_avg + a_avg * ny, H_avg + vn_avg * a_avg]
        )

        # Assemble the right eigenvector matrix
        Rv = np.column_stack((R1, R2, R3, R4))

        # --- Wave Strengths (alpha_k = L_k . dU) ---
        # L_k are the left eigenvectors. Instead of explicitly computing L_k,
        # we can solve Rv * alpha = dU for alpha.
        # alpha = np.linalg.solve(Rv, dU)
        # However, the original code used a direct calculation for dV (wave strengths).
        # Let's try to adapt the original dV calculation to 2D, or use the inverse of Rv.

        # For 2D Euler, the wave strengths are more complex.
        # A common approach is to use the characteristic variables directly.
        # dU = alpha_1 * R1 + alpha_2 * R2 + alpha_3 * R3 + alpha_4 * R4
        # We need to find alpha_k.
        # This requires inverting Rv or using the left eigenvectors.

        # Let's use the direct calculation of alpha_k from Toro, Section 10.2.2
        # d_rho = rR - rL
        # d_rho_u = U_R[1] - U_L[1]
        # d_rho_v = U_R[2] - U_L[2]
        # d_E = U_R[3] - U_L[3]

        # alpha_1 = (dp - r_avg * a_avg * (d_rho_u * nx + d_rho_v * ny) / r_avg) / (2 * a_avg**2)
        # alpha_2 = r_avg * (d_rho_v * nx - d_rho_u * ny) / r_avg
        # alpha_3 = d_rho - dp / a_avg**2
        # alpha_4 = (dp + r_avg * a_avg * (d_rho_u * nx + d_rho_v * ny) / r_avg) / (2 * a_avg**2)

        # The original dV was:
        # dV = np.array([
        #     (dp - r * a * dvn) / (2 * a**2),
        #     r * dvt,
        #     dr - dp / a**2,
        #     (dp + r * a * dvn) / (2 * a**2),
        # ])
        # This is for 1D. For 2D, the characteristic variables are:
        # alpha_1 = 0.5 * (dp / a_avg**2 - d_rho) + 0.5 * r_avg / a_avg * (dU[1]*nx + dU[2]*ny - vn_avg*d_rho)
        # alpha_2 = r_avg * (dU[2]*nx - dU[1]*ny) # Tangential momentum
        # alpha_3 = d_rho - dp / a_avg**2
        # alpha_4 = 0.5 * (dp / a_avg**2 - d_rho) - 0.5 * r_avg / a_avg * (dU[1]*nx + dU[2]*ny - vn_avg*d_rho)

        # Let's use the simpler approach of alpha = inv(Rv) * dU
        # This is numerically more stable than the explicit formulas for alpha_k
        # if Rv is well-conditioned.
        try:
            alpha = np.linalg.solve(Rv, dU)
        except Exception:
            # Fallback to a simpler method or raise error if matrix is singular
            # For now, return HLL flux as a fallback
            return (
                0.5 * (FL + FR) - 0.5 * np.abs(vn_avg) * dU
            )  # Simple HLL-like fallback

        # --- Roe Flux ---
        # F_roe = 0.5 * (F_L + F_R) - 0.5 * sum(ws_i * alpha_i * R_i)
        dissipation = np.zeros_like(dU)
        for i in range(len(ws)):
            dissipation += ws[i] * alpha[i] * Rv[:, i]

        # dissipation = Rv @ (ws * alpha)
        roe_flux = 0.5 * (FL + FR - dissipation)

        return roe_flux

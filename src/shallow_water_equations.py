import numpy as np
from numba import float64
from numba.experimental import jitclass


@jitclass([("gamma", float64)])
class ShallowWaterEquations:
    """
    Represents the 2D Shallow Water Equations for incompressible fluid flow with a free surface.

    This class provides the specific implementation for the Shallow Water Equations,
    including the conversion between conservative and primitive variables,
    flux calculation (HLLC and Roe), and wave speed estimation.

    Attributes:
        g (float): The acceleration due to gravity.
    """

    def __init__(self, g=9.81):
        """
        Initializes the ShallowWaterEquations object.

        Args:
            g (float, optional): The acceleration due to gravity. Defaults to 9.81 m/s^2.
        """
        self.g = g

    def _cons_to_prim(self, U):
        """
        Converts a single conservative state vector to primitive variables.

        Args:
            U (np.ndarray): Conservative state vector [h, hu, hv].
                            h: water depth
                            hu: momentum in x-direction
                            hv: momentum in y-direction

        Returns:
            np.ndarray: Primitive state vector [h, u, v].
                        h: water depth
                        u: velocity in x-direction
                        v: velocity in y-direction
        """
        h, hu, hv = U
        # Ensure h is not too small to prevent division by zero or very large velocities
        h = max(h, 1e-8)
        u = hu / h
        v = hv / h
        return np.array([h, u, v])

    def _prim_to_cons(self, P):
        """
        Converts a single primitive state vector to conservative variables.

        Args:
            P (np.ndarray): Primitive state vector [h, u, v].

        Returns:
            np.ndarray: Conservative state vector [h, hu, hv].
        """
        h, u, v = P
        hu = h * u
        hv = h * v
        return np.array([h, hu, hv])

    def _compute_flux(self, U, normal):
        """
        Calculates the physical flux across a face with a given normal.

        Args:
            U (np.ndarray): Conservative state vector [h, hu, hv].
            normal (np.ndarray): Normal vector of the face.

        Returns:
            np.ndarray: The flux vector normal to the face.
        """
        h, hu, hv = U
        h = max(h, 1e-8)
        u, v = hu / h, hv / h  # Primitive velocities
        un = u * normal[0] + v * normal[1]  # Normal velocity component

        # Flux components for Shallow Water Equations
        F = np.array(
            [
                h * un,
                hu * un + 0.5 * self.g * h**2 * normal[0],
                hv * un + 0.5 * self.g * h**2 * normal[1],
            ]
        )
        return F

    def max_eigenvalue(self, U):
        """
        Calculates the maximum wave speed (eigenvalue) for a cell.
        This is used for determining the stable time step (CFL condition).

        Args:
            U (np.ndarray): Conservative state vector [h, hu, hv].

        Returns:
            float: The maximum absolute eigenvalue.
        """
        h, u, v = self._cons_to_prim(U)
        c = np.sqrt(self.g * h)
        # Max eigenvalue = |velocity| + c
        return np.sqrt(u**2 + v**2) + c

    def apply_boundary_condition(self, U_inside, normal, bc_type, bc_value):
        """
        Computes the state of a ghost cell based on a boundary condition.

        Args:
            U_inside (np.ndarray): Conservative state vector of the interior cell.
            normal (np.ndarray): The outward-pointing normal vector of the boundary face.
            bc_type (int): The type of the boundary condition (0=transmissive, 1=inlet, 2=outlet, 3=wall).
            bc_value (np.ndarray): The value(s) associated with the boundary condition.

        Returns:
            np.ndarray: The conservative state vector of the ghost cell.
        """
        # Convert interior state to primitive variables for easier manipulation
        h_in, u_in, v_in = self._cons_to_prim(U_inside)
        c = np.sqrt(self.g * h_in)

        if bc_type == 0:  # transmissive
            # Ghost cell state is the same as the interior cell state.
            # This is a zero-gradient condition, used for supersonic outlets
            # or far-field boundaries.
            return U_inside

        elif bc_type == 1:  # inlet
            # All primitive variables are specified at the inlet.
            # This is typical for a supersonic inlet (Dirichlet condition).
            h_bc, u_bc, v_bc = bc_value
            P_ghost = np.array([h_bc, u_bc, v_bc])
            return self._prim_to_cons(P_ghost)

        elif bc_type == 2:  # outlet
            # Pressure is specified at the outlet, other variables are extrapolated
            # from the interior. This is typical for a subsonic outlet.
            U_outside = U_inside.copy()
            U_outside[0] = bc_value[0]
            return self._prim_to_cons(U_outside)

        elif bc_type == 3:  # wall
            h_in, u_in, v_in = self._cons_to_prim(U_inside)

            # Decompose velocity into normal and tangential components
            vn_in = u_in * normal[0] + v_in * normal[1]
            vt_in = u_in * -normal[1] + v_in * normal[0]

            # Reflect the normal velocity, keep tangential velocity
            vn_ghost = -vn_in
            vt_ghost = vt_in

            is_slip = bc_value[0] == 1.0
            if is_slip:
                # For a slip wall (inviscid flow), tangential velocity is conserved.
                vt_ghost = vt_in
            else:
                # For a no-slip wall (viscous flow), tangential velocity is zero.
                vt_ghost = 0.0

            # Recompose the ghost velocity vector from the new normal and tangential components
            # u = vn*nx - vt*ny
            # v = vn*ny + vt*nx
            u_ghost = vn_ghost * normal[0] - vt_ghost * normal[1]
            v_ghost = vn_ghost * normal[1] + vt_ghost * normal[0]

            # Create the primitive state for the ghost cell
            P_ghost = np.array([h_in, u_ghost, v_ghost])

            # Convert back to conservative variables
            return self._prim_to_cons(P_ghost)

        else:
            # Default to transmissive for any unknown boundary condition types.
            return U_inside

    def hllc_flux(self, U_L, U_R, normal):
        """
        Computes the numerical flux using the HLLC (Harten-Lax-van Leer-Contact) Riemann solver
        for the 2D Shallow Water Equations.

        Based on "Riemann Solvers and Numerical Methods for Fluid Dynamics" by Eleuterio F. Toro,
        adapted for Shallow Water Equations.

        Args:
            U_L (np.ndarray): Conservative state vector of the left cell [h, hu, hv].
            U_R (np.ndarray): Conservative state vector of the right cell [h, hu, hv].
            normal (np.ndarray): Normal vector of the face (nx, ny).

        Returns:
            np.ndarray: The HLLC numerical flux across the face.
        """
        nx, ny = normal
        # Tangent vector (rotated 90 degrees clockwise from normal)
        tx, ty = -ny, nx

        # --- Left State ---
        hL, uL, vL = self._cons_to_prim(U_L)
        vnL = uL * nx + vL * ny  # Normal velocity
        vtL = uL * tx + vL * ty  # Tangential velocity
        cL = np.sqrt(self.g * max(1e-6, hL))  # Wave speed (celerity)
        FL = self._compute_flux(U_L, normal)

        # --- Right State ---
        hR, uR, vR = self._cons_to_prim(U_R)
        vnR = uR * nx + vR * ny  # Normal velocity
        vtR = uR * tx + vR * ty  # Tangential velocity
        cR = np.sqrt(self.g * max(1e-6, hR))  # Wave speed (celerity)
        FR = self._compute_flux(U_R, normal)

        # --- Roe Averages (for wave speed estimates) ---
        # Roe-averaged density
        h_roe = np.sqrt(hL * hR)
        sqrt_hL = np.sqrt(hL)
        sqrt_hR = np.sqrt(hR)
        # Roe-averaged velocities
        u_roe = (sqrt_hL * uL + sqrt_hR * uR) / (sqrt_hL + sqrt_hR)
        v_roe = (sqrt_hL * vL + sqrt_hR * vR) / (sqrt_hL + sqrt_hR)
        # Roe-averaged normal velocity
        vn_roe = u_roe * nx + v_roe * ny
        # Roe-averaged speed of sound
        a_roe = np.sqrt(self.g * (hL + hR) / 2)  # Simpler Roe average for celerity

        # Wave speeds
        SL = min(vnL - cL, vn_roe - a_roe)
        SR = max(vnR + cR, vn_roe + a_roe)
        SM = (vnL * sqrt_hL + vnR * sqrt_hR + 2 * (cL - cR)) / (
            sqrt_hL + sqrt_hR
        )  # Contact wave speed is the star region velocity

        # --- HLLC Flux Calculation ---
        if 0 <= SL:
            # All waves move to the right
            return FL
        elif SL < 0 <= SM:
            # Left-going shock/rarefaction, contact wave to the right
            # U_star_L state
            h_star_L = hL * (SL - vnL) / (SL - SM)
            hu_star_L = h_star_L * (SM * nx + vtL * tx)
            hv_star_L = h_star_L * (SM * ny + vtL * ty)
            U_star_L = np.array([h_star_L, hu_star_L, hv_star_L])

            return FL + SL * (U_star_L - U_L)
        elif SM < 0 < SR:
            # Contact wave to the left, right-going shock/rarefaction
            # U_star_R state
            h_star_R = hR * (SR - vnR) / (SR - SM)
            hu_star_R = h_star_R * (SM * nx + vtR * tx)
            hv_star_R = h_star_R * (SM * ny + vtR * ty)
            U_star_R = np.array([h_star_R, hu_star_R, hv_star_R])

            return FR + SR * (U_star_R - U_R)
        else:  # SR <= 0
            # All waves move to the left
            return FR

    def roe_flux(self, U_L, U_R, normal):
        """
        Computes the numerical flux using the Roe approximate Riemann solver
        for the 2D Shallow Water Equations.

        Based on "Riemann Solvers and Numerical Methods for Fluid Dynamics" by Eleuterio F. Toro,
        adapted for Shallow Water Equations.

        Args:
            U_L (np.ndarray): Conservative state vector of the left cell [h, hu, hv].
            U_R (np.ndarray): Conservative state vector of the right cell [h, hu, hv].
            normal (np.ndarray): Normal vector of the face (nx, ny).

        Returns:
            np.ndarray: The Roe numerical flux across the face.
        """
        nx, ny = normal
        tx, ty = -ny, nx  # Tangent vector

        # --- Left and Right States ---
        hL, uL, vL = self._cons_to_prim(U_L)
        vnL = uL * nx + vL * ny
        vtL = uL * tx + vL * ty
        FL = self._compute_flux(U_L, normal)

        hR, uR, vR = self._cons_to_prim(U_R)
        vnR = uR * nx + vR * ny
        vtR = uR * tx + vR * ty
        FR = self._compute_flux(U_R, normal)

        # --- Roe Averages ---
        sqrt_hL = np.sqrt(hL)
        sqrt_hR = np.sqrt(hR)
        h_avg = sqrt_hL * sqrt_hR

        u_avg = (sqrt_hL * uL + sqrt_hR * uR) / (sqrt_hL + sqrt_hR)
        v_avg = (sqrt_hL * vL + sqrt_hR * vR) / (sqrt_hL + sqrt_hR)
        c_avg = np.sqrt(self.g * h_avg)
        vn_avg = u_avg * nx + v_avg * ny

        # --- Eigenvalues (Wave Speeds) ---
        lambda1 = vn_avg - c_avg
        lambda2 = vn_avg
        lambda3 = vn_avg + c_avg
        ws = np.array([lambda1, lambda2, lambda3])  # SWE has 3 waves

        # --- Entropy Fix (Harten's entropy fix) ---
        delta = 0.1 * c_avg
        for i in [0]:
            if abs(ws[i]) < delta:
                ws[i] = (ws[i] ** 2 + delta**2) / (2 * delta)
        ws = np.abs(ws)  # Use absolute values for dissipation

        # --- Jump in Conservative Variables ---
        dU = U_R - U_L

        # --- Right Eigenvectors (Roe matrix for 2D Shallow Water) ---
        # Based on Toro, Chapter 13, Section 13.3.2 (for 1D, extended to 2D)
        # R1: (1, u-c, v) - for 1D, need to project to normal/tangential
        R1 = np.array([1.0, u_avg - c_avg * nx, v_avg - c_avg * ny])

        # R2: (0, -ny, nx) - tangential wave
        R2 = c_avg * np.array([0, -ny, nx])

        # R3: (1, u+c, v) - for 1D, need to project to normal/tangential
        R3 = np.array([1, u_avg + c_avg * nx, v_avg + c_avg * ny])

        # Assemble the right eigenvector matrix
        Rv = np.column_stack((R1, R2, R3))

        # --- Wave Strengths (alpha_k = L_k . dU) ---
        try:
            alpha = np.linalg.solve(Rv, dU)
        except Exception:
            # Fallback to HLL flux if matrix is singular
            return 0.5 * (FL + FR) - 0.5 * np.abs(vn_avg) * dU

        # --- Roe Flux ---
        # F_roe = 0.5 * (F_L + F_R) - 0.5 * sum(ws_i * alpha_i * R_i)
        dissipation = np.zeros_like(dU)
        for i in range(len(ws)):
            dissipation += ws[i] * alpha[i] * Rv[:, i]

        # dissipation = Rv @ (ws * alpha)
        roe_flux = 0.5 * (FL + FR - dissipation)

        return roe_flux

import numpy as np
from src.mesh import Mesh
from src.euler_equations import EulerEquations  # Import the jitclass
from src.boundary import create_numba_bcs

import numba
from numba import prange
from line_profiler import profile


# --- Limiter Functions (JIT-compiled for performance) ---
@numba.njit
def barth_jespersen_limiter(r):
    return min(1.0, r)


@numba.njit
def minmod_limiter(r):
    return max(0.0, min(1.0, r))


@numba.njit
def superbee_limiter(r):
    return max(0.0, max(min(2.0 * r, 1.0), min(r, 2.0)))


LIMITERS = {
    "barth_jespersen": barth_jespersen_limiter,
    "minmod": minmod_limiter,
    "superbee": superbee_limiter,
}

# --- Parallelized Core Functions ---


@profile
@numba.njit(parallel=True)
def compute_gradients_gaussian(
    nelem,
    nvars,
    cell_neighbors,
    face_normals,
    face_areas,
    cell_centroids,
    cell_volumes,
    face_to_cell_distances,
    U,
    over_relaxation,
):
    """
    Computes gradients at cell centroids using the Gaussian method.

    This method iterates over each face of a cell, calculates the value of the
    variable at the face, and then uses the divergence theorem to approximate
    the gradient.

    Args:
        mesh (Mesh): The mesh object.
        U (np.ndarray): The conservative state vector for all cells.
        over_relaxation (float, optional): Over-relaxation factor for non-orthogonal
                                         correction. Defaults to 1.2.

    Returns:
        np.ndarray: The gradients of the state variables at each cell centroid.
    """
    gradients = np.zeros((nelem, nvars, 2))
    for i in prange(nelem):
        grad_sum = np.zeros((nvars, 2))
        for j, neighbor_idx in enumerate(cell_neighbors[i]):
            face_normal = face_normals[i, j]
            face_area = face_areas[i, j]

            if neighbor_idx != -1:
                d_i, d_j = face_to_cell_distances[i, j]

                if d_i + d_j > 1e-9:
                    w_i = d_j / (d_i + d_j)
                    w_j = d_i / (d_i + d_j)
                    U_face = w_i * U[i] + w_j * U[neighbor_idx]
                else:
                    # Fallback to simple average if distances are zero
                    U_face = 0.5 * (U[i] + U[neighbor_idx])

                # Non-orthogonal correction for unstructured meshes
                d = cell_centroids[neighbor_idx] - cell_centroids[i]
                if np.linalg.norm(d) > 1e-9:
                    e = d / np.linalg.norm(d)
                    k = face_normal / np.linalg.norm(face_normal)
                    if abs(np.dot(d, k)) > 1e-9:
                        # The correction term is a scalar, but was calculated as a vector.
                        # The correction vector is dotted with the cell-to-cell vector 'd'
                        # to get a scalar correction value.
                        correction_vector = (e - k * np.dot(e, k)) / np.dot(d, k)
                        for m in range(nvars):
                            non_orth_correction = (
                                U[neighbor_idx, m] - U[i, m]
                            ) * np.dot(correction_vector, d)
                            U_face[m] += over_relaxation * non_orth_correction
            else:
                # Boundary face: use the interior cell value
                U_face = U[i]

            grad_sum[:, 0] += U_face * face_normal[0] * face_area
            grad_sum[:, 1] += U_face * face_normal[1] * face_area

        if cell_volumes[i] > 1e-9:
            gradients[i] = grad_sum / cell_volumes[i]

    return gradients


@profile
@numba.njit(parallel=True)
def compute_limiters(
    nelem,
    nvars,
    cell_neighbors,
    face_midpoints,
    cell_centroids,
    U,
    gradients,
    limiter_func,
):
    """
    Computes the slope limiter for each cell to ensure monotonicity.

    This function prevents spurious oscillations (Gibbs phenomenon) near
    discontinuities by limiting the gradient of the solution.

    Args:
        mesh (Mesh): The mesh object.
        U (np.ndarray): The conservative state vector for all cells.
        gradients (np.ndarray): The gradients of the state variables.
        limiter_type (str, optional): The type of limiter to use.
                                    Defaults to "barth_jespersen".

    Returns:
        np.ndarray: The limiter values (phi) for each cell and variable.
    """
    limiters = np.ones((nelem, nvars))
    limiter_func = barth_jespersen_limiter
    for i in prange(nelem):
        neighbors = cell_neighbors[i]
        nfaces = neighbors.shape[0]
        U_i = U[i]
        grad_i = gradients[i]

        # Determine the max and min values among the cell and its neighbors
        U_max = U_i.copy()
        U_min = U_i.copy()
        for neighbor_idx in neighbors:
            if neighbor_idx != -1:
                U_neighbor = U[neighbor_idx]
                U_max = np.maximum(U_max, U_neighbor)
                U_min = np.minimum(U_min, U_neighbor)

        # Check against extrapolated values at face midpoints
        for j in range(nfaces):
            face_midpoint = face_midpoints[i, j]
            r_if = face_midpoint - cell_centroids[i]

            # Extrapolate value to the face midpoint
            U_face_extrap = U_i + np.array(
                [np.dot(grad_i[k], r_if[:2]) for k in range(nvars)]
            )

            # Calculate the limiter ratio (r)
            for k in range(nvars):
                diff = U_face_extrap[k] - U_i[k]
                if abs(diff) > 1e-9:
                    if diff > 0:
                        r = (U_max[k] - U_i[k]) / diff
                    else:
                        r = (U_min[k] - U_i[k]) / diff
                    limiters[i, k] = min(limiters[i, k], limiter_func(r))

    return limiters


@profile
@numba.njit(parallel=True)
def compute_residual_flux_loop(
    nelem,
    nvars,
    cell_neighbors,
    face_normals,
    face_areas,
    face_midpoints,
    cell_centroids,
    cell_volumes,
    U,
    gradients,
    limiters,
    equation,
    flux_type,
    bcs_array,
    elem_faces,
    boundary_faces_nodes,
    boundary_faces_tags,
):
    """
    Computes the residual for the finite volume discretization.

    The residual represents the rate of change of the conservative variables
    in each cell, and is calculated as the sum of fluxes across all faces
    of the cell, divided by the cell volume.

    This function implements a second-order MUSCL-Hancock scheme for spatial
    reconstruction, which involves:
    1. Gradient computation at cell centroids.
    2. Slope limiting to prevent spurious oscillations.
    3. Reconstruction of cell-face values from cell-centroid values.
    4. Numerical flux calculation at each face using a Riemann solver (Roe or HLLC).
    5. Summation of fluxes to compute the cell residual.

    Args:
        mesh (Mesh): The mesh object.
        U (np.ndarray): The array of conservative state vectors for all cells.
        equation: The equation object (e.g., EulerEquations).
        boundary_conditions (dict): A dictionary defining the boundary conditions.
        limiter_type (str): The type of slope limiter to use.
        flux_type (str): The type of numerical flux (Riemann solver) to use.
        over_relaxation (float, optional): Over-relaxation factor for gradient
                                         computation. Defaults to 1.2.

    Returns:
        np.ndarray: The residual array for all cells.
    """
    residual = np.zeros((nelem, nvars))
    for i in prange(nelem):
        flux_sum = np.zeros(nvars)
        for j, neighbor_idx in enumerate(cell_neighbors[i]):
            face_normal = face_normals[i, j, 0:2]
            face_area = face_areas[i, j]
            face_midpoint = face_midpoints[i, j]

            # --- MUSCL Reconstruction ---
            # Reconstruct the state variables at the "left" side of the face (inside the current cell).
            r_i = face_midpoint - cell_centroids[i]
            delta_U_i = np.zeros(nvars)
            for k in range(nvars):
                delta_U_i[k] = np.dot(gradients[i, k], r_i[:2])
            U_L = U[i] + limiters[i] * delta_U_i

            if neighbor_idx != -1:
                # --- Interior Face ---
                # Reconstruct the state variables at the "right" side of the face (inside the neighbor cell).
                r_j = face_midpoint - cell_centroids[neighbor_idx]
                delta_U_j = np.zeros(nvars)
                for k in range(nvars):
                    delta_U_j[k] = np.dot(gradients[neighbor_idx, k], r_j[:2])
                U_R = U[neighbor_idx] + limiters[neighbor_idx] * delta_U_j
            else:
                # --- Boundary Face ---
                # For boundary faces, the "right" state is determined by the boundary condition.
                # This ensures a consistent second-order treatment at the boundaries.

                bcs_array,

                bc_face = elem_faces[i][j]
                for m, face_nodes in enumerate(boundary_faces_nodes):
                    if np.all(face_nodes == bc_face):
                        break

                group_id_to_find = boundary_faces_tags[m]

                bc_type = 3  # Default to wall
                bc_value = np.array([0.0, 0.0, 0.0], dtype=np.float64)
                for n in range(len(bcs_array)):
                    if bcs_array[n][0] == group_id_to_find:
                        bc_type = bcs_array[n][1]
                        bc_value = bcs_array[n][2].copy()
                        break

                U_R = equation.apply_boundary_condition(
                    U_L, face_normal, bc_type, bc_value
                )

            # --- Numerical Flux Calculation ---
            # The numerical flux is computed using the specified Riemann solver.
            if flux_type == "roe":  # Roe
                flux = equation.roe_flux(U_L, U_R, face_normal)
            elif flux_type == "hllc":  # HLLC
                flux = equation.hllc_flux(U_L, U_R, face_normal)

            flux_sum += flux * face_area

        # --- Residual Calculation ---
        # The residual is the sum of fluxes divided by the cell volume.
        # R(U_i) = (1/V_i) * sum(F_j * A_j)
        if cell_volumes[i] > 1e-12:
            residual[i] = flux_sum / cell_volumes[i]

    return residual


@profile
def compute_residual(
    mesh: Mesh,
    U: np.ndarray,
    equation,
    bcs_array,
    limiter_type: str,
    flux_type: str,
    over_relaxation: float = 1.2,
) -> np.ndarray:
    nvars = U.shape[1]

    # --- 1. Gradient Computation ---
    # Gradients are computed at cell centroids and used for second-order reconstruction.
    gradients = compute_gradients_gaussian(
        mesh.nelem,
        nvars,
        mesh.cell_neighbors,
        mesh.face_normals,
        mesh.face_areas,
        mesh.cell_centroids,
        mesh.cell_volumes,
        mesh.face_to_cell_distances,
        U,
        over_relaxation,
    )

    # --- 2. Slope Limiting ---
    # Limiters are applied to the gradients to ensure monotonicity and prevent oscillations.
    limiter_func = LIMITERS.get(limiter_type, barth_jespersen_limiter)
    limiters = compute_limiters(
        mesh.nelem,
        nvars,
        mesh.cell_neighbors,
        mesh.face_midpoints,
        mesh.cell_centroids,
        U,
        gradients,
        limiter_func,
    )

    # --- 3. Flux Integration Loop ---
    # This loop iterates through each cell, calculates the fluxes on its faces,
    # and aggregates them to compute the residual for that cell.
    residual = compute_residual_flux_loop(
        mesh.nelem,
        nvars,
        mesh.cell_neighbors,
        mesh.face_normals,
        mesh.face_areas,
        mesh.face_midpoints,
        mesh.cell_centroids,
        mesh.cell_volumes,
        U,
        gradients,
        limiters,
        equation,
        flux_type,
        bcs_array,
        mesh.elem_faces,
        mesh.boundary_faces_nodes,
        mesh.boundary_faces_tags,
    )

    return residual

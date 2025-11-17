import numba
from numba import prange
import numpy as np

from src.euler_equations import EulerEquations


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


@numba.njit(parallel=True)
def compute_gradients_gaussian(
    num_cells,
    nvars,
    cell_neighbors,
    face_normals,
    face_areas,
    cell_centroids,
    cell_volumes,
    face_to_cell_distances,
    face_midpoints,
    U,
    over_relaxation,
):
    """
    Computes gradients at cell centroids using the Gaussian method.
    """
    gradients = np.zeros((num_cells, nvars, 2))
    for i in prange(num_cells):
        grad_sum = np.zeros((nvars, 2))
        for j, neighbor_idx in enumerate(cell_neighbors[i]):
            face_normal = face_normals[i, j]
            face_area = face_areas[i, j]

            if neighbor_idx != -1:
                d_i = face_to_cell_distances[i, j]
                face_midpoint = face_midpoints[i, j]
                d_j = np.linalg.norm(face_midpoint - cell_centroids[neighbor_idx])

                if d_i + d_j > 1e-9:
                    w_i = d_j / (d_i + d_j)
                    w_j = d_i / (d_i + d_j)
                    U_face = w_i * U[i] + w_j * U[neighbor_idx]
                else:
                    U_face = 0.5 * (U[i] + U[neighbor_idx])

                d = cell_centroids[neighbor_idx] - cell_centroids[i]
                if np.linalg.norm(d) > 1e-9:
                    e = d / np.linalg.norm(d)
                    k = face_normal / np.linalg.norm(face_normal)
                    if abs(np.dot(d, k)) > 1e-9:
                        correction_vector = (e - k * np.dot(e, k)) / np.dot(d, k)
                        for m in range(nvars):
                            non_orth_correction = (
                                U[neighbor_idx, m] - U[i, m]
                            ) * np.dot(correction_vector, d)
                            U_face[m] += over_relaxation * non_orth_correction
            else:
                U_face = U[i]

            grad_sum[:, 0] += U_face * face_normal[0] * face_area
            grad_sum[:, 1] += U_face * face_normal[1] * face_area

        if cell_volumes[i] > 1e-9:
            gradients[i] = grad_sum / cell_volumes[i]

    return gradients


@numba.njit(parallel=True)
def compute_limiters(
    num_cells,
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
    This is a necessary step before the fused flux computation.
    """
    limiters = np.ones((num_cells, nvars))
    for i in prange(num_cells):
        neighbors = cell_neighbors[i]
        U_i = U[i]
        grad_i = gradients[i]

        U_max = U_i.copy()
        U_min = U_i.copy()
        for neighbor_idx in neighbors:
            if neighbor_idx != -1:
                U_neighbor = U[neighbor_idx]
                U_max = np.maximum(U_max, U_neighbor)
                U_min = np.minimum(U_min, U_neighbor)

        for j in range(len(neighbors)):
            face_midpoint = face_midpoints[i, j]
            r_if = face_midpoint - cell_centroids[i]
            U_face_extrap = U_i + np.array(
                [np.dot(grad_i[k], r_if[:2]) for k in range(nvars)]
            )

            for k in range(nvars):
                diff = U_face_extrap[k] - U_i[k]
                if abs(diff) > 1e-9:
                    r = (U_max[k] - U_i[k]) / diff if diff > 0 else (U_min[k] - U_i[k]) / diff
                    limiters[i, k] = min(limiters[i, k], limiter_func(r))
    return limiters


@numba.njit
def _compute_cell_flux(
    i,
    nvars,
    cell_neighbors_i,
    face_midpoints_i,
    cell_centroids,
    U,
    gradients,
    limiters,
    face_normals_i,
    face_areas_i,
    cell_face_tags_i,
    bcs_lookup,
    equation,
    flux_type,
):
    """
    Computes the sum of fluxes for a single cell using pre-computed limiters.
    """
    flux_sum = np.zeros(nvars)
    for j, neighbor_idx in enumerate(cell_neighbors_i):
        face_normal = face_normals_i[j, 0:2]
        face_area = face_areas_i[j]
        face_midpoint = face_midpoints_i[j]

        # Reconstruct left state
        r_i = face_midpoint - cell_centroids[i]
        delta_U_i = np.zeros(nvars)
        for k in range(nvars):
            delta_U_i[k] = np.dot(gradients[i, k], r_i[:2])
        U_L = U[i] + limiters[i] * delta_U_i

        if neighbor_idx != -1:
            # Reconstruct right state (interior face)
            r_j = face_midpoint - cell_centroids[neighbor_idx]
            delta_U_j = np.zeros(nvars)
            for k in range(nvars):
                delta_U_j[k] = np.dot(gradients[neighbor_idx, k], r_j[:2])
            U_R = U[neighbor_idx] + limiters[neighbor_idx] * delta_U_j
        else:
            # Determine right state from boundary condition
            face_tag = cell_face_tags_i[j]
            bc_data = bcs_lookup[face_tag]
            bc_type = bc_data["type"]
            bc_value = bc_data["values"]
            U_R = equation.apply_boundary_condition(U_L, face_normal, bc_type, bc_value)

        # Compute numerical flux
        if flux_type == "roe":
            flux = equation.roe_flux(U_L, U_R, face_normal)
        else:  # HLLC
            flux = equation.hllc_flux(U_L, U_R, face_normal)

        flux_sum += flux * face_area

    return flux_sum


@numba.njit(parallel=True)
def compute_fused_residual(
    num_owned_cells,
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
    equation: EulerEquations,
    bcs_lookup,
    cell_face_tags,
    flux_type,
):
    """
    Computes the residual using a fused loop for fluxes, using pre-computed gradients and limiters.
    """
    residual = np.zeros((num_owned_cells, nvars))
    for i in prange(num_owned_cells):
        flux_sum = _compute_cell_flux(
            i,
            nvars,
            cell_neighbors[i],
            face_midpoints[i],
            cell_centroids,
            U,
            gradients,
            limiters,
            face_normals[i],
            face_areas[i],
            cell_face_tags[i],
            bcs_lookup,
            equation,
            flux_type,
        )

        if cell_volumes[i] > 1e-12:
            residual[i] = flux_sum / cell_volumes[i]

    return residual

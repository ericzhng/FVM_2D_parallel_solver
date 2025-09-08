import numpy as np
import unittest


class TestEulerEquations(unittest.TestCase):
    """
    Test suite for the euler_equations module.
    """

    def setUp(self):
        self.node_tags = np.load("tests/node_tags.npy")
        self.elem_conn = np.load("tests/elem_conn.npy")
        self.node_coords = np.load("tests/node_coords.npy")
        self.nnode = len(self.node_tags)
        self.nelem = self.elem_conn.shape[0]

        # Create a mapping from node tags to their 0-based index.
        max_tag = np.max(self.node_tags)
        self.node_tag_map = np.full(max_tag + 1, -1, dtype=np.int32)
        self.node_tag_map[self.node_tags] = np.arange(self.nnode, dtype=np.int32)

    def _compute_cell_centroids_old(self):
        """Computes the centroid of each element."""
        cell_centroids = np.zeros((self.nelem, 3))
        for i, elem_nodes_tags in enumerate(self.elem_conn):
            node_indices = [
                np.where(self.node_tags == tag)[0][0] for tag in elem_nodes_tags
            ]
            nodes = self.node_coords[np.array(node_indices)]
            cell_centroids[i] = np.mean(nodes, axis=0)

    def _compute_cell_centroids(self):
        """Computes the centroid of each element."""
        cell_centroids = np.zeros((self.nelem, 3))

        for i, elem_nodes_tags in enumerate(self.elem_conn):
            node_indices = [self.node_tag_map[tag] for tag in elem_nodes_tags]
            nodes = self.node_coords[node_indices]
            cell_centroids[i] = np.mean(nodes, axis=0)

    def _compute_cell_centroids_fastest(self):
        """Computes the centroid of each element using vectorized operations."""
        # Use the map to convert element connectivity from tags to indices.
        elem_node_indices = self.node_tag_map[self.elem_conn]

        # Gather all node coordinates for all elements.
        elem_nodes_coords = self.node_coords[elem_node_indices]

        # Compute the mean over the nodes for each element to get the centroids.
        cell_centroids = np.mean(elem_nodes_coords, axis=1)

    def test_compare(self):
        self._compute_cell_centroids_fastest()
        self._compute_cell_centroids()
        self._compute_cell_centroids_old()


if __name__ == "__main__":
    unittest.main()

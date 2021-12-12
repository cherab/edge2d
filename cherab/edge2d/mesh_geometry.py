
# Copyright 2016-2021 Euratom
# Copyright 2016-2021 United Kingdom Atomic Energy Authority
# Copyright 2016-2021 Centro de Investigaciones Energéticas, Medioambientales y Tecnológicas
#
# Licensed under the EUPL, Version 1.1 or – as soon they will be approved by the
# European Commission - subsequent versions of the EUPL (the "Licence");
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at:
#
# https://joinup.ec.europa.eu/software/page/eupl5
#
# Unless required by applicable law or agreed to in writing, software distributed
# under the Licence is distributed on an "AS IS" basis, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied.
#
# See the Licence for the specific language governing permissions and limitations
# under the Licence.

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection


class Edge2DMesh:
    """
    Edge2DMesh geometry object.

    The Edge2D mesh is quadrilateral. Each mesh cell is denoted by four vertices with one centre
    point. Vertices may be shared with neighbouring cells. The centre points should be unique.

    Raysect's mesh interpolator uses a different mesh scheme. Mesh cells are triangles and data
    values are stored at the triangle vertices.
    Therefore, each Edge2D rectangular cell is split into two triangular cells.

    :param ndarray r: Array of cell vertex r coordinates, must be 2 dimensional.
                      Example shape is (4 x 3000).
    :param ndarray z: Array of cell vertex z coordinates, must be 2 dimensional.
                      Example shape is (4 x 3000).
    :param ndarray vol: Array of cell volumes in m-3. Default is None (calculated from r, z).
    """

    def __init__(self, r, z, vol=None):

        if r.shape != z.shape:
            raise ValueError('Shape of r array: {0} mismatch the shape of z array: {1}.'.format(r.shape, z.shape))

        if vol is not None and vol.size != r.shape[1]:
            raise ValueError('Size of vol array: {0} mismatch the number of grid cells: {1}.'.format(vol.size, r.shape[1]))

        self._r = r
        self._z = z

        if vol is None:
            cell_area = 0.5 * np.abs((r[0] - r[1]) * (z[0] + z[1]) + (r[1] - r[2]) * (z[1] + z[2]) +
                                     (r[2] - r[3]) * (z[2] + z[3]) + (r[3] - r[0]) * (z[3] + z[0]))
            self._vol = 0.5 * np.pi * r.sum(0) * cell_area
        else:
            self._vol = vol

        self._initial_setup()

    def _initial_setup(self):

        self._cr = self._r.sum(0) / 4.
        self._cz = self._z.sum(0) / 4.

        self._n = self._r.shape[1]

        # Calculating poloidal basis vector
        self._poloidal_basis_vector = np.zeros((2, self._n))
        vec_r = self._r[3] - self._r[0]
        vec_z = self._z[3] - self._z[0]
        vec_magn = np.sqrt(vec_r**2 + vec_z**2)
        self._poloidal_basis_vector[0] = np.divide(vec_r, vec_magn, out=np.zeros_like(vec_magn), where=(vec_magn > 0))
        self._poloidal_basis_vector[1] = np.divide(vec_z, vec_magn, out=np.zeros_like(vec_magn), where=(vec_magn > 0))

        # Calculating radial contact areas
        self._radial_area = np.pi * (self._r[3] + self._r[0]) * vec_magn

        # Calculating radial basis vector
        self._radial_basis_vector = np.zeros((2, self._n))
        vec_r = self._r[1] - self._r[0]
        vec_z = self._z[1] - self._z[0]
        vec_magn = np.sqrt(vec_r**2 + vec_z**2)
        self._radial_basis_vector[0] = np.divide(vec_r, vec_magn, out=np.zeros_like(vec_magn), where=(vec_magn > 0))
        self._radial_basis_vector[1] = np.divide(vec_z, vec_magn, out=np.zeros_like(vec_magn), where=(vec_magn > 0))

        # Calculating poloidal contact areas
        self._poloidal_area = np.pi * (self._r[1] + self._r[0]) * vec_magn

        # For convertion from Cartesian to poloidal
        # TODO Make it work with triangle cells
        self._inv_det = 1. / (self._poloidal_basis_vector[0] * self._radial_basis_vector[1] -
                              self._poloidal_basis_vector[1] * self._radial_basis_vector[0])

        # Finding unique vertices
        vertices = np.array([self._r.flatten(), self._z.flatten()]).T
        self._vertex_coords, unique_vertices = np.unique(vertices, axis=0, return_inverse=True)
        self._num_vertices = self._vertex_coords.shape[0]

        # Work out the extent of the mesh.
        self._mesh_extent = {"minr": self._r.min(), "maxr": self._r.max(), "minz": self._z.min(), "maxz": self._z.max()}

        # add quadrangle Edge2D grid
        self._quadrangles = np.zeros((self._n, 4), dtype=np.int32)
        self._quadrangles[:, 0] = unique_vertices[0:self._n]
        self._quadrangles[:, 1] = unique_vertices[self._n: 2 * self._n]
        self._quadrangles[:, 2] = unique_vertices[2 * self._n: 3 * self._n]
        self._quadrangles[:, 3] = unique_vertices[3 * self._n: 4 * self._n]

        # Number of triangles must be equal to number of rectangle centre points times 2.
        self._num_tris = self._n * 2
        self._triangles = np.zeros((self._num_tris, 3), dtype=np.int32)
        self._triangles[0::2, 0] = self._quadrangles[:, 0]
        self._triangles[0::2, 1] = self._quadrangles[:, 1]
        self._triangles[0::2, 2] = self._quadrangles[:, 2]
        # Split the quad cell into two triangular cells.
        self._triangles[1::2, 0] = self._quadrangles[:, 2]
        self._triangles[1::2, 1] = self._quadrangles[:, 3]
        self._triangles[1::2, 2] = self._quadrangles[:, 0]

        # Each triangle cell is mapped to the tuple ID (ix, iy) of its parent mesh cell.
        self._triangle_to_grid_map = np.zeros(self._num_tris, dtype=np.int32)
        self._triangle_to_grid_map[::2] = np.arange(self._n, dtype=np.int32)
        self._triangle_to_grid_map[1::2] = self._triangle_to_grid_map[::2]

    @property
    def n(self):
        """Number of grid cells."""
        return self._n

    @property
    def cr(self):
        """R-coordinate of the cell centres."""
        return self._cr

    @property
    def cz(self):
        """Z-coordinate of the cell centres."""
        return self._cz

    @property
    def r(self):
        """R-coordinates of the cell vertices."""
        return self._r

    @property
    def z(self):
        """Z-coordinate of the cell vertices."""
        return self._z

    @property
    def vol(self):
        """Volume of each grid cell in m-3."""
        return self._vol

    @property
    def radial_area(self):
        """Radial contact area in m-3."""
        return self._radial_area

    @property
    def poloidal_area(self):
        """Poloidal contact area in m-2."""
        return self._poloidal_area

    @property
    def vertex_coordinates(self):
        """RZ-coordinates of unique vertices."""
        return self._vertex_coords

    @property
    def num_vertices(self):
        """Total number of unique vertices."""
        return self._num_vertices

    @property
    def mesh_extent(self):
        """Extent of the mesh. A dictionary with minr, maxr, minz and maxz keys."""
        return self._mesh_extent

    @property
    def num_triangles(self):
        """Total number of triangles (the number of cells doubled)."""
        return self._num_tris

    @property
    def triangles(self):
        """Array of triangle vertex indices with (num_thiangles, 3) shape."""
        return self._triangles

    @property
    def quadrangles(self):
        """Array of quadrangle vertex indices with (num_thiangles, 3) shape."""
        return self._quadrangles

    @property
    def poloidal_basis_vector(self):
        """
        Array of 2D poloidal basis vectors for grid cells.

        For each cell there is a poloidal and radial basis vector.

        Any vector on the poloidal grid can be converted to cartesian with the following transformation.
        bx = (p_x  r_x) ( b_p )
        by   (p_y  r_y) ( b_r )

        :return: ndarray with shape (2, n).
        """
        return self._poloidal_basis_vector

    @property
    def radial_basis_vector(self):
        """
        Array of 2D radial basis vectors for grid cells.

        For each cell there is a poloidal and radial basis vector.

        Any vector on the poloidal grid can be converted to cartesian with the following transformation.
        bx = (p_x  r_x) ( b_p )
        by   (p_y  r_y) ( b_r )

        :return: ndarray with shape (2, n).
        """
        return self._radial_basis_vector

    @property
    def triangle_to_grid_map(self):
        """
        Array mapping every triangle index to a tuple grid cell ID.

        :return: ndarray with size n * 2
        """
        return self._triangle_to_grid_map

    def __getstate__(self):
        state = {
            'r': self._r,
            'z': self._z,
            'vol': self._vol
        }
        return state

    def __setstate__(self, state):
        self._r = state['r']
        self._z = state['z']
        self._vol = state['vol']
        self._initial_setup()

    def to_cartesian(self, vec_pol):
        """
        Converts the 2D vector defined on mesh from poloidal to cartesian coordinates.
        :param ndarray vec_pol: Array of 2D vector with with shape (2, n).
            [0, :] - poloidal component, [1, :] - radial component

        :return: ndarray with shape (2, n)
        """
        vec_cart = np.zeros((2, self._n))
        vec_cart[0] = self._poloidal_basis_vector[0] * vec_pol[0] + self._radial_basis_vector[0] * vec_pol[1]
        vec_cart[1] = self._poloidal_basis_vector[1] * vec_pol[0] + self._radial_basis_vector[1] * vec_pol[1]

        return vec_cart

    def to_poloidal(self, vec_cart):
        """
        Converts the 2D vector defined on mesh from cartesian to poloidal coordinates.
        :param ndarray vector_on_mesh: Array of 2D vector with with shape (2, n).
            [0, :] - R component, [1, :] - Z component

        :return: ndarray with shape (2, n)
        """
        vec_pol = np.zeros((2, self._n))
        vec_pol[0] = self._inv_det * (self._radial_basis_vector[1] * vec_cart[0] - self._radial_basis_vector[0] * vec_cart[1])
        vec_pol[1] = self._inv_det * (self._poloidal_basis_vector[0] * vec_cart[1] - self._poloidal_basis_vector[1] * vec_cart[0])

        return vec_pol

    def plot_triangle_mesh(self, edge2d_data=None, ax=None):
        """
        Plot the triangle mesh grid geometry to a matplotlib figure.

        :param edge2d_data: Data array defined on the EDGE2D mesh
        """
        if ax is None:
            _, ax = plt.subplots(constrained_layout=True)

        verts = self.vertex_coordinates[self.triangles]
        if edge2d_data is None:
            collection_mesh = PolyCollection(verts, facecolor="none", edgecolor='b', linewidth=0.5)
        else:
            collection_mesh = PolyCollection(verts)
            collection_mesh.set_array(edge2d_data[self.triangle_to_grid_map])
        ax.add_collection(collection_mesh)
        ax.set_aspect(1)
        ax.set_xlim(self.mesh_extent["minr"], self.mesh_extent["maxr"])
        ax.set_ylim(self.mesh_extent["minz"], self.mesh_extent["maxz"])
        ax.set_xlabel("R [m]")
        ax.set_ylabel("Z [m]")

        return ax

    def plot_quadrangle_mesh(self, edge2d_data=None, ax=None):
        """
        Plot the quadrangle mesh grid geometry to a matplotlib figure.

        :param edge2d_data: Data array defined on the EDGE2D mesh
        """

        if ax is None:
            _, ax = plt.subplots(constrained_layout=True)

        verts = self.vertex_coordinates[self.quadrangles]
        if edge2d_data is None:
            collection_mesh = PolyCollection(verts, facecolor="none", edgecolor='b', linewidth=0.5)
        else:
            collection_mesh = PolyCollection(verts)
            collection_mesh.set_array(edge2d_data)
        ax.add_collection(collection_mesh)
        ax.set_aspect(1)
        ax.set_xlim(self.mesh_extent["minr"], self.mesh_extent["maxr"])
        ax.set_ylim(self.mesh_extent["minz"], self.mesh_extent["maxz"])
        ax.set_xlabel("R [m]")
        ax.set_ylabel("Z [m]")

        return ax

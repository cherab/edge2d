
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

import matplotlib.pyplot as plt

from cherab.edge2d import load_edge2d_from_tranfile

# Load the simulation.
tranfile = '/home/pheliste/cmg/catalog/edge2d/jet/81472/jul1816/seq#2/tran'
sim = load_edge2d_from_tranfile(tranfile)

# plot the quadrangle EDGE2D mesh
ax = sim.mesh.plot_quadrangle_mesh()
ax.set_title("Quadrangle EDGE2D Mesh")
ax.get_figure().set_size_inches((5.8, 12))

# plot the quadrangle EDGE2D mesh with EDGE2D ion temperature values
ax = sim.mesh.plot_quadrangle_mesh(edge2d_data=sim.ion_temperature)
ax.get_figure().colorbar(ax.collections[0], aspect=40)
ax.get_figure().set_size_inches((5.8, 12))
ax.set_title("EDGE2D Ion Temperature [eV]")

# axes can also be passed as an argument
fig_pass, ax = plt.subplots(figsize=(5.8, 12))
ax = sim.mesh.plot_triangle_mesh(edge2d_data=sim.ion_temperature, ax=ax)
ax.get_figure().colorbar(ax.collections[0], aspect=40)
ax.set_title("Cherab Triangle Mesh with Ion Temperature [eV]")

# plot poloidal and toroidal magnetic field
# in EDGE2D poloidal projection is positive in counterclockwise direction
# toroidal projection is positive in clockwise direction when looking at the torus from above
ax = sim.mesh.plot_quadrangle_mesh(edge2d_data=sim.b_field[0])
ax.get_figure().colorbar(ax.collections[0], aspect=40)
ax.get_figure().set_size_inches((5.8, 12))
ax.set_title("EDGE2D Poloidal Magnetic Field [T]")

ax = sim.mesh.plot_quadrangle_mesh(edge2d_data=sim.b_field[2])
ax.get_figure().colorbar(ax.collections[0], aspect=40)
ax.get_figure().set_size_inches((5.8, 12))
ax.set_title("EDGE2D Toroidal Magnetic Field [T]")

# plot poloidal and toroidal ion velocities
ax = sim.mesh.plot_quadrangle_mesh(edge2d_data=sim.velocities[1, 0])
ax.get_figure().colorbar(ax.collections[0], aspect=40)
ax.get_figure().set_size_inches((5.8, 12))
ax.set_title("EDGE2D D+ Poloidal Velocity")
ax.collections[0].set_cmap('seismic')
ax.collections[0].set_clim(-5000, 5000)

ax = sim.mesh.plot_quadrangle_mesh(edge2d_data=sim.velocities[1, 2])
ax.get_figure().colorbar(ax.collections[0], aspect=40)
ax.get_figure().set_size_inches((5.8, 12))
ax.set_title("EDGE2D D+ Toroidal Velocity")
ax.collections[0].set_cmap('seismic')
ax.collections[0].set_clim(-5000, 5000)

plt.show()

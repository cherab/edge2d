
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
from matplotlib.colors import SymLogNorm
import numpy as np
from scipy import linalg

from cherab.core.atomic import deuterium, beryllium
from cherab.core.math import sample2d, sample2d_grid, sample3d, sample3d_grid, samplevector3d_grid
from cherab.edge2d import load_edge2d_from_tranfile


###############################################################################
# Load the simulation.
###############################################################################
tranfile = '/home/vsolokha/cmg/catalog/edge2d/jet/81472/aug1820/seq#1/tran'
print('Loading simulation...')
sim = load_edge2d_from_tranfile(tranfile)

###############################################################################
# Sample some poloidal profiles from the simulation.
###############################################################################
mesh = sim.mesh
me = mesh.mesh_extent
xl, xu = (me['minr'], me['maxr'])
zl, zu = (me['minz'], me['maxz'])
nx = 500
nz = 500

print('Sampling profiles...')
xsamp, zsamp, ne_samples = sample2d(sim.electron_density_f2d, (xl, xu, nx), (zl, zu, nz))
te_samples = sample2d_grid(sim.electron_temperature_f2d, xsamp, zsamp)
d0_samples = sample2d_grid(sim.species_density_f2d[('deuterium', 0)], xsamp, zsamp)
d1_samples = sample2d_grid(sim.species_density_f2d[('deuterium', 1)], xsamp, zsamp)
be0_samples = sample2d_grid(sim.species_density_f2d[('beryllium', 0)], xsamp, zsamp)
be1_samples = sample2d_grid(sim.species_density_f2d[('beryllium', 1)], xsamp, zsamp)
be2_samples = sample2d_grid(sim.species_density_f2d[('beryllium', 2)], xsamp, zsamp)
be3_samples = sample2d_grid(sim.species_density_f2d[('beryllium', 3)], xsamp, zsamp)
be4_samples = sample2d_grid(sim.species_density_f2d[('beryllium', 4)], xsamp, zsamp)

# Cartesian velocity is a 3D profile.
d1_velocity = samplevector3d_grid(sim.velocities_cartesian[('deuterium', 1)],
                                  xsamp, [0], zsamp).squeeze()
d1_speed = linalg.norm(d1_velocity, axis=-1)
# Mask determining whether a point is inside or outside the simulation mesh in
# the poloidal plane. See sim.inside_volume_mesh for the 3D equivalent.
inside_samples = sample2d_grid(sim.inside_mesh, xsamp, zsamp)

###############################################################################
# Create a Cherab plasma from the simulation and sample quantities.
###############################################################################
print('Creating plasma...')
plasma = sim.create_plasma()

# Extract information about the species in the plasma. For brevity we'll only
# use the main ion and a single impurity charge state in this demo.
d0 = plasma.composition.get(deuterium, 0)
d1 = plasma.composition.get(deuterium, 1)
be0 = plasma.composition.get(beryllium, 0)

# The distributions are 3D, so we perform a sample in 3D space with only a
# single y value to get a poloidal profile.
xsamp, _, zsamp, ne_plasma = sample3d(plasma.electron_distribution.density,
                                      (xl, xu, nx), (0, 0, 1), (zl, zu, nz))
ne_plasma = ne_plasma.squeeze()
te_plasma = sample3d_grid(plasma.electron_distribution.effective_temperature,
                          xsamp, [0], zsamp).squeeze()
d0_plasma = sample3d_grid(d0.distribution.density, xsamp, [0], zsamp).squeeze()
d1_plasma = sample3d_grid(d1.distribution.density, xsamp, [0], zsamp).squeeze()
be0_plasma = sample3d_grid(be0.distribution.density, xsamp, [0], zsamp).squeeze()
d1_plasma_velocity = samplevector3d_grid(d1.distribution.bulk_velocity, xsamp, [0], zsamp).squeeze()
d1_plasma_speed = linalg.norm(d1_plasma_velocity, axis=-1)

# Compare sampled quantities from the plasma with those from the simulation object.
print('Comparing plasma and simulation sampled quantities...')
np.testing.assert_equal(ne_plasma, ne_samples)
np.testing.assert_equal(te_plasma, te_samples)
np.testing.assert_equal(d0_plasma, d0_samples)
np.testing.assert_equal(d1_plasma, d1_samples)
np.testing.assert_equal(be0_plasma, be0_samples)
np.testing.assert_equal(d1_plasma_speed, d1_speed)
print('Plasma and simulation sampled quantities are identical.')

###############################################################################
# Plot the sampled vales.
###############################################################################
mesh.plot_triangle_mesh()
plt.title('Mesh geometry')


def plot_quantity(quantity, title, logscale):
    """
    Make a 2D plot of quantity, with a title, optionally on a log scale.
    """
    fig, ax = plt.subplots()
    if logscale:
        # Plot lowest values (mainly 0's) on linear map, as log(0) = -inf.
        linthresh = np.percentile(np.unique(quantity), 1)
        norm = SymLogNorm(linthresh=linthresh)
    else:
        norm = None
    # Sampled data is indexed as quantity(x, y), but matplotlib's imshow
    # expects quantity(y, x).
    image = ax.imshow(quantity.T, extent=[xl, xu, zl, zu], origin='lower', norm=norm)
    fig.colorbar(image)
    ax.set_xlim(xl, xu)
    ax.set_ylim(zl, zu)
    ax.set_title(title)


plot_quantity(ne_samples, 'Electron density [m-3]', True)
plot_quantity(te_samples, 'Electron temperature [eV]', False)
plot_quantity(d0_samples, 'D0 density [m-3]', True)
plot_quantity(d1_samples, 'D+ density [m-3]', True)
plot_quantity(be0_samples, 'BeI density [m-3]', True)
plot_quantity(be1_samples, 'BeII density [m-3]', True)
plot_quantity(be2_samples, 'BeIII density [m-3]', True)
plot_quantity(be3_samples, 'BeIV density [m-3]', True)
plot_quantity(be4_samples, 'BeV density [m-3]', True)
plot_quantity(inside_samples, 'Inside/outside test', False)

plt.show()

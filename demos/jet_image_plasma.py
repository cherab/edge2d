
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

from raysect.core.math import Point3D, Vector3D, translate, rotate_basis, rotate_z
from raysect.optical import World, Spectrum
from raysect.optical.observer import PinholeCamera, RadiancePipeline2D

from cherab.core.atomic import Line, deuterium
from cherab.core.model import ExcitationLine, RecombinationLine
from cherab.openadas import OpenADAS

from cherab.jet.machine import import_jet_mesh
from cherab.edge2d.models import make_edge2d_emitter
from cherab.edge2d import load_edge2d_from_tranfile


###############################################################################
# Load the simulation and create a plasma object from it.
###############################################################################
tranfile = '/home/pheliste/cmg/catalog/edge2d/jet/81472/jul1816/seq#2/tran'
print('Loading simulation...')
sim = load_edge2d_from_tranfile(tranfile)

print('Creating plasma...')
plasma = sim.create_plasma()

###############################################################################
# Image the plasma with a camera.
###############################################################################
print('Imaging plasma...')
world = World()

# Load the generomak first wall
import_jet_mesh(world)


# Adding D-alpha emission model
plasma.atomic_data = OpenADAS(permit_extrapolation=True)
d_alpha = Line(deuterium, 0, (3, 2))
plasma.models = [ExcitationLine(d_alpha), RecombinationLine(d_alpha)]

# Caching D-alpha emissivity for faster calculation
direction = Vector3D(0, 0, 1)
min_wavelength = 655.1
max_wavelength = 657.1
dw = max_wavelength - min_wavelength
dalpha_emissivity = np.zeros(sim.mesh.n)
for model in plasma.models:
    for i in range(sim.mesh.n):
        point = Point3D(sim.mesh.cr[i], 0, sim.mesh.cz[i])
        spectrum = Spectrum(min_wavelength, max_wavelength, 1)
        dalpha_emissivity[i] += 4. * np.pi * model.emission(point, direction, spectrum).samples[0] * dw
sim.halpha_radiation = dalpha_emissivity

# Making the emitter
emitter = make_edge2d_emitter(sim.mesh, sim.halpha_radiation_f2d, parent=world)

# A wide-angle pinhole camera looking horizontally
camera = PinholeCamera((256, 256))
camera.parent = world
camera.pixel_samples = 200
camera.transform = (rotate_z(22.5)
                    * translate(1.02 * sim.mesh.mesh_extent['maxr'], 0, 0)
                    * rotate_basis(Vector3D(-1, 0, 0), Vector3D(0, 0, 1)))
camera.fov = 90
# The emitter returned by make_edge2_emitter is not spectrally resolved. So use a monochromatic
# pipeline to image the plasma.
camera.pipelines = [RadiancePipeline2D()]
camera.spectral_bins = 1

plt.ion()
camera.observe()

plt.ioff()
plt.show()

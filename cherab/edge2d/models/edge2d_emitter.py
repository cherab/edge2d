# cython: language_level=3

# Copyright 2016-2018 Euratom
# Copyright 2016-2018 United Kingdom Atomic Energy Authority
# Copyright 2016-2018 Centro de Investigaciones Energéticas, Medioambientales y Tecnológicas
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

from raysect.core import translate
from raysect.optical.material import VolumeTransform
from raysect.primitive import Cylinder, Subtract

from cherab.core.math.mappers import AxisymmetricMapper
from cherab.tools.emitters import RadiationFunction

from cherab.edge2d import Edge2DMesh, Edge2DFunction


def make_edge2d_emitter(edge2d_mesh, radiation_function, parent=None, step=0.01):
    """
    Non-spectral emitter with the emissivity defined as Edge2DFunction.

    :param Edge2DMesh edge2d_mesh: EDGE2D simulation mesh.
    :param Edge2DFunction radiation_function: Emissivity in W m-3.
    :param Node parent: parent node in the scenegraph, e.g. a World object.
    :param float step: Volume integration step in meters.

    :rtype: Primitive
    """

    if not isinstance(edge2d_mesh, Edge2DMesh):
        raise TypeError('Argument edge2d_mesh must be a Edge2DMesh instance.')
    if not isinstance(radiation_function, Edge2DFunction):
        raise TypeError('Argument radiation_function must be a Edge2DFunction instance.')

    radiation_function_3d = AxisymmetricMapper(radiation_function)
    outer_radius = edge2d_mesh.mesh_extent['maxr']
    inner_radius = edge2d_mesh.mesh_extent['minr']
    height = edge2d_mesh.mesh_extent['maxz'] - edge2d_mesh.mesh_extent['minz']
    lower_z = edge2d_mesh.mesh_extent['minz']
    emitter = RadiationFunction(radiation_function_3d, step=step)
    material = VolumeTransform(emitter, transform=translate(0, 0, -lower_z))

    plasma_volume = Subtract(Cylinder(outer_radius, height), Cylinder(inner_radius, height),
                             material=material, parent=parent, transform=translate(0, 0, lower_z))

    return plasma_volume

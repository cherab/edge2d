
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

import os
import numpy as np

from eproc import dataread

from cherab.core.utility import PhotonToJ
from cherab.core.atomic.elements import lookup_isotope, lookup_element
from cherab.openadas import OpenADAS

from cherab.edge2d.mesh_geometry import Edge2DMesh
from cherab.edge2d.edge2d_plasma import Edge2DSimulation, prefer_element


# TODO: find out if EDGE2D supports non-hydrogen (e.g. helium as a main ion) plasmas
#       and how to process such cases
# TODO: find out how to process multi-isotope cases, there is no data for impurity's
#       atomic mass number, only for the nuclear charge
# TODO: find out how to obtain neutral atom velocities for impurity species
# TODO: read total radiation
def load_edge2d_from_tranfile(tranfile):
    """
    Load an EDGE2D simulation from EDGE2D tran file.

    :param str tranfile: String path to a simulation tran file.
    :rtype: Edge2DSimulation
    """

    if not os.path.isfile(tranfile):
        raise RuntimeError("File {} not found.".format(tranfile))

    mesh = create_mesh_from_tranfile(tranfile)
    n = mesh.n

    korpg = dataread(tranfile, 'KORPG')
    korpg.data = np.frombuffer(korpg.data, dtype=np.int32)  # float32 to int32
    indx_cell, = np.where(korpg.data[:korpg.nPts] > 0)

    # plasma composition
    am = dataread(tranfile, 'HMASS')  # main ion mass number
    am = int(am.data[0])
    zn = dataread(tranfile, 'HCH')  # main ion nuclear charge
    zn = np.frombuffer(zn.data[:zn.nPts], np.int32)[0]  # float32 to int32
    main_isotope = lookup_isotope(zn, number=am)
    main_species = prefer_element(main_isotope)  # main plasma species
    species_list = [(main_species.name, charge) for charge in range(zn + 1)]

    nz_imp = dataread(tranfile, 'NZ')  # impurity charge states
    nz_imp = np.frombuffer(nz_imp.data[:nz_imp.nPts], np.int32)  # float32 to int32
    n_imp = nz_imp.size if nz_imp.sum() else 0  # number of impurities
    if n_imp:
        zn_imp = np.frombuffer(dataread(tranfile, 'ZCH').data[:n_imp], np.int32)  # impurity nuclear charge
        # obtaining impurity charges from charge state distribution
        # Note: Cherab does not support fractional charge states, so using a workaround
        i_impz = 0
        impurity_indx_dict = {}
        for i, zn_i in enumerate(zn_imp):
            element = lookup_element(zn_i)
            species_list.append((element.name, 0))
            for ich in range(nz_imp[i]):
                charge_distrib = np.round(dataread(tranfile, 'ZI{:02d}'.format(i_impz + 1)).data[indx_cell]).astype(np.int32)
                charges = np.unique(charge_distrib)
                for charge in charges:
                    species = (element.name, charge)
                    if species in species_list:
                        impurity_indx_dict[species].append((i_impz, np.where(charge_distrib == charge)[0]))
                    else:
                        species_list.append(species)
                        impurity_indx_dict[species] = [(i_impz, np.where(charge_distrib == charge)[0])]
                i_impz += 1
    neutral_index = [i for i, spec in enumerate(species_list) if spec[1] == 0]

    sim = Edge2DSimulation(mesh, species_list)

    # magnetic field
    b_field = np.zeros((3, n))
    b_field[2] = dataread(tranfile, 'BFI').data[indx_cell]  # toroidal component
    b_pol_to_b_tot = dataread(tranfile, 'SH').data[indx_cell]
    b_tot = b_field[2] / np.sqrt(1. - b_pol_to_b_tot * b_pol_to_b_tot)
    b_field[0] = b_tot * b_pol_to_b_tot  # poloidal component
    b_rz = mesh.poloidal_basis_vector * b_field[0]
    b_field_cyl = np.array([b_rz[0], -b_field[2], b_rz[1]])  # magnetic field in cylindrical coordinates

    parallel_vector = b_field_cyl / b_tot
    perpendicular_vector = np.array([b_rz[1], np.zeros(n), -b_rz[0]]) / np.sqrt((b_rz * b_rz).sum(0))

    # electron distibution data
    ne = dataread(tranfile, 'DENEL').data[indx_cell]  # electron density
    te = dataread(tranfile, 'TEVE').data[indx_cell]  # electron temperature
    ve_par = dataread(tranfile, 'VPE').data[indx_cell]  # electron parallel velocity
    ve_perp = dataread(tranfile, 'VROE').data[indx_cell]  # electron perpendicular velocity
    ve_cyl = ve_par * parallel_vector + ve_perp * perpendicular_vector  # electron velocity in cylindrical coordinates

    species_density = np.zeros((len(species_list), n))
    neutral_temperature = np.zeros((len(neutral_index), n))
    velocities_cyl = np.zeros((len(species_list), 3, n))

    # main neutral atom distibution data
    species_density[0] = dataread(tranfile, 'DA').data[indx_cell]  # main atom density
    neutral_temperature[0] = dataread(tranfile, 'ENEUTA').data[indx_cell]  # main atom temperature
    velocities_cyl[0, 0] = dataread(tranfile, 'VA0R').data[indx_cell]  # radial component of main atom's velocity
    velocities_cyl[0, 1] = -dataread(tranfile, 'VA0T').data[indx_cell]  # toroidal component of main atom's velocity
    velocities_cyl[0, 2] = dataread(tranfile, 'VA0Z').data[indx_cell]  # z component of main atom's velocity

    # main ion distibution data
    species_density[1] = dataread(tranfile, 'DEN').data[indx_cell]  # main ion density
    ti = dataread(tranfile, 'TEV').data[indx_cell]  # main ion temperature
    vi_par = dataread(tranfile, 'VPI').data[indx_cell]  # main ion parallel velocity
    vi_perp = dataread(tranfile, 'VRO').data[indx_cell]  # main ion parallel velocity
    velocities_cyl[1] = vi_par * parallel_vector + vi_perp * perpendicular_vector  # ion velocity in cylindrical coordinates

    if n_imp:
        # read data for neutral impurities
        for i, i_spec in enumerate(neutral_index[1:]):
            if n_imp == 1:
                species_density[i_spec] = dataread(tranfile, 'DZ').data[indx_cell]  # impurity neutral atom density
                neutral_temperature[i + 1] = dataread(tranfile, 'ENEUTZ').data[indx_cell]  # impurity neutral atom temperature
            else:
                species_density[i_spec] = dataread(tranfile, 'DZ_{}'.format(i + 1)).data[indx_cell]  # impurity neutral atom density
                neutral_temperature[i + 1] = dataread(tranfile, 'ENEUTZ_{}'.format(i + 1)).data[indx_cell]  # impurity neutral atom temperature

        # read data for ionized impurities
        nz_total = nz_imp.sum()
        nz_density = np.zeros((nz_total, n))
        nz_velocities_cyl = np.zeros((nz_total, 3, n))
        for i_impz in range(nz_total):
            nz_density[i_impz] = dataread(tranfile, 'DENZ{:02d}'.format(i_impz + 1)).data[indx_cell]  # impurity iz+ ion density
            vi_imp_par = dataread(tranfile, 'VPZ{:02d}'.format(i_impz + 1)).data[indx_cell]  # impurity iz+ ion parallel velocity
            vi_imp_perp = dataread(tranfile, 'VROZ{:02d}'.format(i_impz + 1)).data[indx_cell]  # impurity iz+ ion parallel velocity
            nz_velocities_cyl[i_impz] = vi_imp_par * parallel_vector + vi_imp_perp * perpendicular_vector

        for i_spec, species in enumerate(species_list[zn + 1:]):
            if species[1]:  # impurity ions
                for (i_impz, indx_imp) in impurity_indx_dict[species]:
                    species_density[i_spec + zn + 1, indx_imp] += nz_density[i_impz, indx_imp]
                    for k in range(3):
                        velocities_cyl[i_spec + zn + 1, k, indx_imp] = nz_velocities_cyl[i_impz, k, indx_imp]

    # H-alpha radiation density
    h_alpha = dataread(tranfile, 'DHA').data[indx_cell]

    sim.b_field = b_field  # set magnetic field
    sim.electron_density = ne  # set electron density
    sim.electron_temperature = te  # set electron temperature
    sim.electron_velocities_cylindrical = ve_cyl  # set electron velocity
    sim.ion_temperature = ti  # set ion temperature
    sim.species_density = species_density  # set species density
    sim.neutral_temperature = neutral_temperature  # set neutral atom temperature
    sim.velocities_cylindrical = velocities_cyl  # set species velocities

    if np.any(h_alpha):
        openadas = OpenADAS()
        wavelength = openadas.wavelength(lookup_isotope(species_list[0][0]), 0, (3, 2))
        sim.h_alpha_radiation = PhotonToJ.to(h_alpha, wavelength)  # photon/s -> W

    return sim


def create_mesh_from_tranfile(tranfile):

    r_vert = dataread(tranfile, 'RVERTP').data  # R-coordinates of cell vertices
    z_vert = dataread(tranfile, 'ZVERTP').data  # Z-coordinates of cell vertices

    z_vert *= -1.0  # Z data is upside down

    korpg = dataread(tranfile, 'KORPG')  # maps k-mesh to plasma cells
    korpg.data = np.frombuffer(korpg.data, dtype=np.int32)  # float32 to int32
    indx_cell = korpg.data[:korpg.nPts]
    indx_cell_data, = np.where(indx_cell > 0)
    ncell = indx_cell_data.size

    r = np.zeros((4, ncell))
    z = np.zeros((4, ncell))

    indx_cell = 5 * (indx_cell - 1)
    indx_cell = indx_cell[indx_cell >= 0]

    for iv in range(4):
        r[iv] = r_vert[indx_cell + iv]
        z[iv] = z_vert[indx_cell + iv]

    vol = dataread(tranfile, 'DV').data[indx_cell_data]

    mesh = Edge2DMesh(r, z, vol=vol)

    return mesh

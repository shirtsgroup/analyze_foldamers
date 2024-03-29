import os

import matplotlib.pyplot as pyplot
import numpy as np
from cg_openmm.build.cg_build import build_topology
from cg_openmm.cg_model.cgmodel import CGModel
from cg_openmm.ensembles.ens_build import get_native_structure
from cg_openmm.parameters.reweight import (get_free_energy_differences,
                                           get_mbar_expectation,
                                           get_temperature_list)
from cg_openmm.parameters.secondary_structure import get_helical_parameters
from cg_openmm.simulation.rep_exch import *
from cg_openmm.thermo.calc import calculate_heat_capacity
from openmm import unit

grid_size = 11

# Job settings
top_directory = "output"
if not os.path.exists(top_directory):
    os.mkdir(top_directory)

# OpenMM simulation settings
print_frequency = 20  # Number of steps to skip when printing output
total_simulation_time = 1.0 * unit.nanosecond  # Units = picoseconds
simulation_time_step = 5.0 * unit.femtosecond
total_steps = round(total_simulation_time.__div__(simulation_time_step))

# Yank (replica exchange) simulation settings
output_data = str(str(top_directory) + "/output.nc")
number_replicas = 50
min_temp = 1.0 * unit.kelvin
max_temp = 200.0 * unit.kelvin
temperature_list = get_temperature_list(min_temp, max_temp, number_replicas)
if total_steps > 10000:
    exchange_attempts = round(total_steps / 1000)
else:
    exchange_attempts = 10

###
#
# Coarse grained model settings
#
###

# Global definitions
bond_length = 7.5 * unit.angstrom
bond_lengths = {
    "bb_bb_bond_length": bond_length,
    "bb_sc_bond_length": bond_length,
    "sc_sc_bond_length": bond_length,
}
bond_force_constant = 1250 * unit.kilojoule_per_mole / unit.nanometer / unit.nanometer

pitch_list = []
radius_list = []
monomers_per_turn_list = []
residual_list = []
data = open("helical_data.dat", "w")
data.write(
    "Sigma (nm) Epsilon (kcal/mol) Pitch (Angstroms) Radius (Angstroms) Monomers-per-turn\n"
)
data.close()

sigma_list = [(1.5 + i * 0.1) * bond_length for i in range(grid_size)]
epsilon_list = [
    unit.Quantity((0.5 + i * 0.25), unit.kilocalorie_per_mole) for i in range(grid_size)
]

for sigma in sigma_list:
    for epsilon in epsilon_list:
        print("Sigma = " + str(sigma))
        print("Epsilon = " + str(epsilon))
        output_data = str(
            str(top_directory)
            + "/eps_"
            + str(round(epsilon._value, 1))
            + "_sig_"
            + str(round(sigma._value, 1))
            + ".nc"
        )
        sigmas = {"bb_bb_sigma": sigma, "bb_sc_sigma": sigma, "sc_sc_sigma": sigma}
        epsilons = {"bb_bb_eps": epsilon, "bb_sc_eps": epsilon, "sc_sc_eps": epsilon}
        cgmodel = CGModel(sigmas=sigmas, epsilons=epsilons, bond_lengths=bond_lengths)
        if not os.path.exists(output_data):
            success = False
            while not success:
                try:
                    replica_energies, replica_positions, replica_states = run_replica_exchange(
                        cgmodel.topology,
                        cgmodel.system,
                        cgmodel.positions,
                        temperature_list=temperature_list,
                        simulation_time_step=simulation_time_step,
                        total_simulation_time=total_simulation_time,
                        print_frequency=print_frequency,
                        output_data=output_data,
                    )
                    success = True
                except:
                    os.remove(output_data)
        else:
            replica_energies, replica_positions, replica_states = read_replica_exchange_data(
                system=cgmodel.system,
                topology=cgmodel.topology,
                temperature_list=temperature_list,
                output_data=output_data,
                print_frequency=print_frequency,
            )

        steps_per_stage = round(total_steps / exchange_attempts)

        minimum_energy_structure = get_native_structure(
            replica_positions, replica_energies, temperature_list
        )

        cgmodel.positions = minimum_energy_structure

        pitch, radius, monomers_per_turn, residual = get_helical_parameters(cgmodel)

        pitch_list.append(pitch)
        radius_list.append(radius)
        monomers_per_turn_list.append(monomers_per_turn)
        residual_list.append(residual)

        data = open("helical_data.dat", "a")
        data.write(
            str(round(sigma._value, 3))
            + " "
            + str(round(epsilon._value, 3))
            + " "
            + str(round(float(pitch), 3))
            + " "
            + str(round(float(radius), 3))
            + " "
            + str(round(float(monomers_per_turn), 3))
            + " "
            + str(round(float(residual), 3))
            + "\n"
        )
        data.close()

file_name = str(str(top_directory) + "/pitch.png")
figure = pyplot.figure(1)
x = np.unique([sigma._value for sigma in sigma_list])
y = np.unique([epsilon._value for epsilon in epsilon_list])
X, Y = np.meshgrid(x, y)
Z = np.array([float(pitch) for pitch in pitch_list]).reshape(len(y), len(x))
pyplot.xlabel("$\sigma$ ( nm )")
pyplot.ylabel("$\epsilon$ ( kcal/mol )")
pyplot.title("Helical pitch (Angstroms)")
pyplot.pcolormesh(X, Y, Z)
pyplot.colorbar()
pyplot.savefig(file_name)
pyplot.show()
pyplot.close()

file_name = str(str(top_directory) + "/radius.png")
figure = pyplot.figure(1)
x = np.unique([sigma._value for sigma in sigma_list])
y = np.unique([epsilon._value for epsilon in epsilon_list])
X, Y = np.meshgrid(x, y)
Z = np.array([float(radius) for radius in radius_list]).reshape(len(y), len(x))
pyplot.xlabel("$\sigma$ ( nm )")
pyplot.ylabel("$\epsilon$ ( kcal/mol )")
pyplot.title("Helical radius (Angstroms)")
pyplot.pcolormesh(X, Y, Z)
pyplot.colorbar()
pyplot.savefig(file_name)
pyplot.show()
pyplot.close()

file_name = str(str(top_directory) + "/monomers-per-turn.png")
figure = pyplot.figure(1)
x = np.unique([sigma._value for sigma in sigma_list])
y = np.unique([epsilon._value for epsilon in epsilon_list])
X, Y = np.meshgrid(x, y)
Z = np.array([float(mpt) for mpt in monomers_per_turn_list]).reshape(len(y), len(x))
pyplot.xlabel("$\sigma$ ( nm )")
pyplot.ylabel("$\epsilon$ ( kcal/mol )")
pyplot.title("Monomers-per-turn")
pyplot.pcolormesh(X, Y, Z)
pyplot.colorbar()
pyplot.savefig(file_name)
pyplot.show()
pyplot.close()

file_name = str(str(top_directory) + "/residual.png")
figure = pyplot.figure(1)
x = np.unique([sigma._value for sigma in sigma_list])
y = np.unique([epsilon._value for epsilon in epsilon_list])
X, Y = np.meshgrid(x, y)
Z = np.array([float(pitch) for pitch in pitch_list]).reshape(len(y), len(x))
pyplot.xlabel("$\sigma$ ( nm )")
pyplot.ylabel("$\epsilon$ ( kcal/mol )")
pyplot.title("Residual of helical fit (Angstroms)")
pyplot.pcolormesh(X, Y, Z)
pyplot.colorbar()
pyplot.savefig(file_name)
pyplot.show()
pyplot.close()

exit()

import os

import matplotlib.pyplot as pyplot
import numpy as np
from cg_openmm.build.cg_build import build_topology
from cg_openmm.cg_model.cgmodel import CGModel
from cg_openmm.parameters.reweight import (get_free_energy_differences,
                                           get_mbar_expectation,
                                           get_temperature_list)
from cg_openmm.simulation.rep_exch import *
from cg_openmm.thermo.calc import calculate_heat_capacity
from openmm import unit

# Job settings
top_directory = "output"
if not os.path.exists(top_directory):
    os.mkdir(top_directory)

# OpenMM simulation settings
print_frequency = 5  # Number of steps to skip when printing output
total_simulation_time = 1.0 * unit.nanosecond  # Units = picoseconds
simulation_time_step = 5.0 * unit.femtosecond
total_steps = round(total_simulation_time.__div__(simulation_time_step))

# Yank (replica exchange) simulation settings
number_replicas = 50
min_temp = 1.0 * unit.kelvin
max_temp = 300.0 * unit.kelvin
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
polymer_length = 8
backbone_lengths = [1]
sidechain_lengths = [1]
sidechain_positions = [0]
include_bond_forces = False
include_bond_angle_forces = True
include_nonbonded_forces = True
include_torsion_forces = True
constrain_bonds = True

# Particle properties
mass = 100.0 * unit.amu
masses = {"backbone_bead_masses": mass, "sidechain_bead_masses": mass}

# Bonded interaction properties
bond_length = 7.5 * unit.angstrom
bond_lengths = {
    "bb_bb_bond_length": bond_length,
    "bb_sc_bond_length": bond_length,
    "sc_sc_bond_length": bond_length,
}
bond_force_constant = 1250 * unit.kilojoule_per_mole / unit.nanometer / unit.nanometer
bond_force_constants = {
    "bb_bb_bond_k": bond_force_constant,
    "bb_sc_bond_k": bond_force_constant,
    "sc_sc_bond_k": bond_force_constant,
}

sigma_range = range(round(bond_length._value * 1.5), round(bond_length._value * 2.5))
epsilon = 0.5 * unit.kilocalorie_per_mole
epsilons = {"bb_bb_eps": epsilon, "bb_sc_eps": epsilon, "sc_sc_eps": 0.5 * epsilon}

# Bond angle properties
bond_angle_force_constant = 200 * unit.kilojoule_per_mole / unit.radian / unit.radian
bond_angle_force_constants = {
    "bb_bb_bb_angle_k": bond_angle_force_constant,
    "bb_bb_sc_angle_k": bond_angle_force_constant,
    "bb_sc_sc_angle_k": bond_angle_force_constant,
    "sc_sc_sc_angle_k": bond_angle_force_constant,
    "sc_bb_sc_angle_k": bond_angle_force_constant,
    "sc_sc_bb_angle_k": bond_angle_force_constant,
}
equil_bond_angle = 120
equil_bond_angles = {
    "bb_bb_bb_angle_0": equil_bond_angle,
    "bb_bb_sc_angle_0": equil_bond_angle,
    "bb_sc_sc_angle_0": equil_bond_angle,
    "sc_sc_sc_angle_0": equil_bond_angle,
    "sc_bb_sc_angle_0": equil_bond_angle,
    "sc_sc_bb_angle_0": equil_bond_angle,
}

# Torsion properties
torsion_force_constant = 200
torsion_force_constants = {
    "bb_bb_bb_bb_torsion_k": torsion_force_constant,
    "bb_bb_bb_sc_torsion_k": torsion_force_constant,
    "bb_bb_sc_sc_torsion_k": torsion_force_constant,
    "bb_sc_sc_sc_torsion_k": torsion_force_constant,
    "sc_bb_bb_sc_torsion_k": torsion_force_constant,
    "bb_sc_sc_bb_torsion_k": torsion_force_constant,
    "sc_sc_sc_sc_torsion_k": torsion_force_constant,
    "sc_bb_bb_bb_torsion_k": torsion_force_constant,
}
equil_torsion_angle = 0
equil_torsion_angles = {
    "bb_bb_bb_bb_torsion_0": equil_torsion_angle,
    "bb_bb_bb_sc_torsion_0": equil_torsion_angle,
    "bb_bb_sc_sc_torsion_0": equil_torsion_angle,
    "bb_sc_sc_sc_torsion_0": equil_torsion_angle,
    "sc_bb_bb_sc_torsion_0": equil_torsion_angle,
    "bb_sc_sc_bb_torsion_0": equil_torsion_angle,
    "sc_sc_sc_sc_torsion_0": equil_torsion_angle,
    "sc_bb_bb_bb_torsion_0": equil_torsion_angle,
}

C_v_list = []
dC_v_list = []

sigma_list = [
    sigma * bond_length.unit
    for sigma in range(round(bond_length._value * 1.5), round(bond_length._value * 2.5))
]
for sigma in sigma_list:
    print("Performing simulations and heat capacity analysis for a coarse grained model")
    print("with sigma values of " + str(sigma))
    sigmas = {"bb_bb_sigma": sigma, "bb_sc_sigma": sigma, "sc_sc_sigma": sigma}
    cgmodel = CGModel(
        polymer_length=polymer_length,
        backbone_lengths=backbone_lengths,
        sidechain_lengths=sidechain_lengths,
        sidechain_positions=sidechain_positions,
        masses=masses,
        sigmas=sigmas,
        epsilons=epsilons,
        bond_lengths=bond_lengths,
        bond_force_constants=bond_force_constants,
        bond_angle_force_constants=bond_angle_force_constants,
        torsion_force_constants=torsion_force_constants,
        equil_bond_angles=equil_bond_angles,
        equil_torsion_angles=equil_torsion_angles,
        include_nonbonded_forces=include_nonbonded_forces,
        include_bond_forces=include_bond_forces,
        include_bond_angle_forces=include_bond_angle_forces,
        include_torsion_forces=include_torsion_forces,
        constrain_bonds=constrain_bonds,
    )

    # Run a replica exchange simulation with this cgmodel
    output_data = str(str(top_directory) + "/sig_" + str(sigma._value) + ".nc")
    if not os.path.exists(output_data):
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
        steps_per_stage = round(total_steps / exchange_attempts)
        plot_replica_exchange_energies(
            replica_energies,
            temperature_list,
            simulation_time_step,
            steps_per_stage=steps_per_stage,
        )
        plot_replica_exchange_summary(
            replica_states, temperature_list, simulation_time_step, steps_per_stage=steps_per_stage
        )
    else:
        replica_energies, replica_positions, replica_states = read_replica_exchange_data(
            system=cgmodel.system,
            topology=cgmodel.topology,
            temperature_list=temperature_list,
            output_data=output_data,
            print_frequency=print_frequency,
        )

    steps_per_stage = round(total_steps / exchange_attempts)
    plot_replica_exchange_energies(
        replica_energies,
        temperature_list,
        simulation_time_step,
        steps_per_stage=steps_per_stage,
        legend=False,
    )
    plot_replica_exchange_summary(
        replica_states,
        temperature_list,
        simulation_time_step,
        steps_per_stage=steps_per_stage,
        legend=False,
    )
    num_intermediate_states = 1
    mbar, E_kn, E_expect, dE_expect, new_temp_list = get_mbar_expectation(
        replica_energies, temperature_list, num_intermediate_states
    )

    mbar, E_kn, DeltaE_expect, dDeltaE_expect, new_temp_list = get_mbar_expectation(
        E_kn, temperature_list, num_intermediate_states, mbar=mbar, output="differences"
    )

    mbar, E_kn, E2_expect, dE2_expect, new_temp_list = get_mbar_expectation(
        E_kn ** 2, temperature_list, num_intermediate_states, mbar=mbar
    )

    df_ij, ddf_ij = get_free_energy_differences(mbar)

    C_v, dC_v = calculate_heat_capacity(
        E_expect,
        E2_expect,
        dE_expect,
        DeltaE_expect,
        dDeltaE_expect,
        df_ij,
        ddf_ij,
        new_temp_list,
        len(temperature_list),
        num_intermediate_states,
    )
    C_v_list.append(C_v)
    dC_v_list.append(dC_v)

file_name = str(str(top_directory) + "/heat_capacity.png")
figure = pyplot.figure(1)
original_temperature_list = np.array([temperature._value for temperature in temperature_list])
try:
    temperatures = np.array([temperature._value for temperature in new_temp_list])
except:
    temperatures = np.array([temperature for temperature in new_temp_list])
legend_labels = [
    str("$\sigma / r_{bond}$= " + str(round(i / bond_length._value, 2))) for i in sigma_range
]

for C_v, dC_v in zip(C_v_list, dC_v_list):
    C_v = np.array([C_v[i][0] for i in range(len(C_v))])
    dC_v = np.array([dC_v[i][0] for i in range(len(dC_v))])
    pyplot.errorbar(temperatures, C_v, yerr=dC_v, figure=figure)

pyplot.xlabel("Temperature ( Kelvin )")
pyplot.ylabel("C$_v$ ( kcal/mol * Kelvin )")
pyplot.title("Heat capacity for variable $\sigma / r_{bond}$")
pyplot.legend(legend_labels)
pyplot.savefig(file_name)
pyplot.show()
pyplot.close()

figure = pyplot.figure(2)
file_name = str(str(top_directory) + "/heat_capacity_low_T.png")
for C_v, dC_v in zip(C_v_list, dC_v_list):
    C_v = np.array([C_v[i][0] for i in range(len(C_v))])
    dC_v = np.array([dC_v[i][0] for i in range(len(dC_v))])
    pyplot.errorbar(temperatures, C_v, yerr=dC_v, figure=figure)
pyplot.xlabel("Temperature ( Kelvin )")
pyplot.ylabel("C$_v$ ( kcal/mol * Kelvin )")
pyplot.title("Heat capacity for variable $\sigma / r_{bond}$")
pyplot.legend(legend_labels)
pyplot.xlim(10.0, 25.0)
pyplot.savefig(file_name)
pyplot.show()
pyplot.close()

exit()

import os
import pickle

import matplotlib.pyplot as pyplot
import mdtraj as md
import numpy as np
from analyze_foldamers.parameters.angle_distributions import *
from cg_openmm.cg_model.cgmodel import CGModel
from openmm import unit

# This script calculates and plots angle distributions from a CGModel object and pdb trajectory

# Load in a trajectory pdb file:
traj_file = "simulation.pdb"

# Load in a CGModel:
cgmodel = pickle.load(open("stored_cgmodel.pkl","rb"))

# With a mapping dictionary, we can rename particle type names or 
# combine multiple particle types into a common bonded type:

particle_mapping = {
    'bb': 'b',
    'sc': 's',
    }

angle_hist_data = calc_bond_angle_distribution(
    cgmodel,
    traj_file,
    particle_mapping=particle_mapping,
    )
    
torsion_hist_data = calc_torsion_distribution(
    cgmodel,
    traj_file,
    particle_mapping=particle_mapping,
    )


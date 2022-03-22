import os
import pickle

import matplotlib.pyplot as pyplot
import mdtraj as md
import numpy as np
from analyze_foldamers.parameters.bond_distributions import *
from cg_openmm.cg_model.cgmodel import CGModel
from openmm import unit

# This script creates bond length distribution plots for a given trajectory and cgmodel:

# Load in a trajectory pdb file:
traj_file = "output/state_1.pdb"

# Load in a CGModel:
cgmodel = pickle.load(open("stored_cgmodel.pkl","rb"))

bond_hist_data = calc_bond_length_distribution(cgmodel, traj_file)
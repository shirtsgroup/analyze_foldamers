import os
import numpy as np
import matplotlib.pyplot as pyplot
import mdtraj as md
from simtk import unit
from cg_openmm.cg_model.cgmodel import CGModel
import pickle
from analyze_foldamers.parameters.bond_distributions import *

# This script creates bond length distribution plots for a given trajectory and cgmodel:

# Load in a trajectory pdb file:
traj_file = "output/state_1.pdb"

# Load in a CGModel:
cgmodel = pickle.load(open("stored_cgmodel.pkl","rb"))

bond_hist_data = calc_bond_length_distribution(cgmodel, traj_file)
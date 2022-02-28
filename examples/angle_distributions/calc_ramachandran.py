import os
import pickle

import matplotlib.pyplot as pyplot
import mdtraj as md
import numpy as np
from analyze_foldamers.parameters.angle_distributions import *
from cg_openmm.cg_model.cgmodel import CGModel
from openmm import unit

# This script creates theta-alpha ramachandran plots from a CGModel object and pdb trajectory

# Load in a trajectory pdb file:
traj_file_list = []
nreplica=1

for rep in range(nreplica):
    traj_file_list.append(f"output/state_{rep+1}.pdb")

# Load in a CGModel:
cgmodel = pickle.load(open("stored_cgmodel.pkl","rb"))

rama_hist, xedges, yedges = calc_ramachandran(
    cgmodel,
    traj_file_list,
    plotfile="ramachandran.pdf"
)

# Fit ramachandran data to 2d radially symmetric Gaussian function
param_opt, param_cov = fit_ramachandran_data(rama_hist, xedges, yedges)

np.savetxt("hist_data1.txt", rama_hist["output/state_1"])

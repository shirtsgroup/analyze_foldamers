import os
import numpy as np
from simtk import unit
from cg_openmm.cg_model.cgmodel import CGModel
import pickle
from analyze_foldamers.parameters.bond_distributions import *
from analyze_foldamers.parameters.angle_distributions import *

# This script calculates and plots 2d histograms for any combination of bonded parameters,
# given a CGModel object and list of trajectory files (pdb or dcd).

# Load in a trajectory pdb file:

file_list = []
file_list.append("output/state_1.pdb")

# Load in a CGModel:
cgmodel = pickle.load(open("stored_cgmodel.pkl", "rb"))

hist_data, xedges, yedges = calc_2d_distribution(
    cgmodel,
    file_list,
    nbin_xvar=180,
    nbin_yvar=180,
    frame_start=10000,
    frame_stride=1,
    frame_end=-1,
    plotfile="2d_hist.pdf",
    xvar_name="bb_bb_bb_sc",
    yvar_name="sc_bb_bb_sc",
    colormap="Spectral",
)

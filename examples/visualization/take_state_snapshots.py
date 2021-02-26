import os
import numpy as np
import matplotlib.pyplot as pyplot
import mdtraj as md
from simtk import unit
from cg_openmm.cg_model.cgmodel import CGModel
from analyze_foldamers.ensembles.cluster import *
from analyze_foldamers.utilities.snapshot import *
import pickle

# This example determines representative structures of a series of state trajectories,
# and renders each structure using VMD.
# ***Note: VMD must first be installed before using the function

# Load in a trajectory pdb file:
traj_file_list = []
nreplica = 12

for i in range(nreplica):
    traj_file_list.append(f"output/state_{i+1}.dcd")

# Load in a CGModel:
cgmodel = pickle.load(open("stored_cgmodel.pkl", "rb"))

# Determine representative structures:
file_list_out = get_representative_structures(
    traj_file_list, cgmodel, output_dir="output", frame_start=1000, frame_stride=100
)

# Create snapshots using VMD TachyonInternal renderer
take_snapshot(file_list_out)

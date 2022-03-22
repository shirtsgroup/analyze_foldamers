import os
import pickle

import matplotlib.pyplot as plt
import mdtraj as md
import numpy as np
from analyze_foldamers.ensembles.cluster import *
from cg_openmm.cg_model.cgmodel import CGModel
from openmm import unit

# Load in cgmodel
cgmodel = pickle.load(open( "stored_cgmodel.pkl", "rb" ))

# Create list of trajectory files for clustering analysis
number_replicas = 24
pdb_file_list = []
for i in range(number_replicas):
    pdb_file_list.append("output/replica_%s.pdb" %(i+1))

# Set clustering parameters
min_samples=5
frame_start=0
frame_stride=10
frame_end=-1

# Run OPTICS density-based clustering
    medoid_positions, cluster_sizes, cluster_rmsd, n_noise, silhouette_avg = get_cluster_medoid_positions_OPTICS(
        dcd_file_list,
        cgmodel,
        min_samples=min_samples,
        frame_start=frame_start,
        frame_stride=frame_stride,
        frame_end=-1,
        output_format="dcd",
        output_dir="output",
        plot_silhouette=True,
    )

print(f'Cluster sizes: {cluster_sizes}')
print(f'Cluster rmsd: {cluster_rmsd}')
print(f"Fraction noise: {n_noise/(np.sum(cluster_sizes)+n_noise)}")
print(f"Average silhouette score: {silhouette_avg}")

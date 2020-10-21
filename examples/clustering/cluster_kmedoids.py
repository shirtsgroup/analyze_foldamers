import os
from simtk import unit
import mdtraj as md
import numpy as np
import matplotlib.pyplot as plt
from cg_openmm.cg_model.cgmodel import CGModel
from analyze_foldamers.ensembles.cluster import *
import pickle

# Load in cgmodel
cgmodel = pickle.load(open( "stored_cgmodel.pkl", "rb" ))

# Create list of trajectory files for clustering analysis
number_replicas = 24
pdb_file_list = []
for i in range(number_replicas):
    pdb_file_list.append("output/replica_%s.pdb" %(i+1))

# Set clustering parameters
n_clusters=4
frame_start=0
frame_stride=10
frame_end=-1

# Run KMeans clustering
medoid_positions, cluster_size, cluster_rmsd, silhouette_avg = \
    get_cluster_medoid_positions_KMedoids(
        pdb_file_list,
        cgmodel,
        n_clusters=n_clusters,
        frame_start=frame_start,
        frame_stride=frame_stride,
        frame_end=frame_end,
        output_dir=output_directory,
        plot_silhouette=True,
        plot_rmsd_hist=True,
        filter=True,
        filter_ratio=0.20,
    )

print(f'Cluster sizes: {cluster_size}')
print(f'Cluster rmsd: {cluster_rmsd}')

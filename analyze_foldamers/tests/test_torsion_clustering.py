"""
Unit and regression test for the analyze_foldamers package.
"""

import os
import pickle
import sys

# Import package, test suite, and other packages as needed
import analyze_foldamers
import pytest
from analyze_foldamers.ensembles.cluster_torsion import *
from cg_openmm.cg_model.cgmodel import CGModel

current_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_path, 'test_data')

def test_cluster_torsions_kmedoids_pdb(tmpdir):
    """Test Kmeans clustering"""
    
    output_directory = tmpdir.mkdir("output")
    
    # Load in cgmodel
    cgmodel_path = os.path.join(data_path, "stored_cgmodel.pkl")
    cgmodel = pickle.load(open(cgmodel_path, "rb"))
    
    # Create list of trajectory files for clustering analysis
    number_replicas = 12
    pdb_file_list = []
    for i in range(number_replicas):
        pdb_file_list.append(f"{data_path}/replica_%s.pdb" %(i+1))

    # Set clustering parameters
    n_clusters=2
    frame_start=10
    frame_stride=2
    frame_end=-1

    # Run KMeans clustering
    medoid_positions, medoid_torsions, cluster_size, cluster_rmsd, silhouette_avg = \
        cluster_torsions_KMedoids(
            pdb_file_list,
            cgmodel,
            n_clusters=n_clusters,
            frame_start=frame_start,
            frame_stride=frame_stride,
            frame_end=-1,
            output_dir=output_directory,
            plot_silhouette=True,
            plot_distance_hist=True,
            filter=True,
            filter_ratio=0.20,
        )
    
    assert len(cluster_rmsd) == n_clusters
    assert os.path.isfile(f"{output_directory}/medoid_1.pdb")
    assert os.path.isfile(f"{output_directory}/silhouette_kmedoids_ncluster_{n_clusters}.pdf") 
    assert os.path.isfile(f"{output_directory}/torsion_distances_hist.pdf")
    
    
def test_cluster_torsions_kmedoids_dcd(tmpdir):
    """Test KMedoids clustering"""
    
    output_directory = tmpdir.mkdir("output")
    
    # Load in cgmodel
    cgmodel_path = os.path.join(data_path, "stored_cgmodel.pkl")
    cgmodel = pickle.load(open(cgmodel_path, "rb"))
    
    # Create list of trajectory files for clustering analysis
    number_replicas = 12
    dcd_file_list = []
    for i in range(number_replicas):
        dcd_file_list.append(f"{data_path}/replica_%s.dcd" %(i+1))

    # Set clustering parameters
    n_clusters=2
    frame_start=10
    frame_stride=2
    frame_end=-1

    # Run KMeans clustering
    medoid_positions, medoid_torsions, cluster_size, cluster_rmsd, silhouette_avg = \
        cluster_torsions_KMedoids(
            dcd_file_list,
            cgmodel,
            n_clusters=n_clusters,
            frame_start=frame_start,
            frame_stride=frame_stride,
            frame_end=-1,
            output_format="dcd",
            output_dir=output_directory,
            plot_silhouette=True,
            plot_distance_hist=True,
            filter=True,
            filter_ratio=0.20,
        )
    
    assert len(cluster_rmsd) == n_clusters
    assert os.path.isfile(f"{output_directory}/medoid_1.dcd")
    assert os.path.isfile(f"{output_directory}/silhouette_kmedoids_ncluster_{n_clusters}.pdf")
    assert os.path.isfile(f"{output_directory}/torsion_distances_hist.pdf")    
    
    
def test_cluster_torsions_dbscan_pdb(tmpdir):
    """Test DBSCAN clustering"""
    
    output_directory = tmpdir.mkdir("output")
    
    # Load in cgmodel
    cgmodel_path = os.path.join(data_path, "stored_cgmodel.pkl")
    cgmodel = pickle.load(open(cgmodel_path, "rb"))
    
    # Create list of trajectory files for clustering analysis
    number_replicas = 12
    pdb_file_list = []
    for i in range(number_replicas):
        pdb_file_list.append(f"{data_path}/replica_%s.pdb" %(i+1))

    # Set clustering parameters
    min_samples=3
    eps=50
    frame_start=10
    frame_stride=2
    frame_end=-1

    # Run OPTICS density-based clustering
    medoid_positions, medoid_torsions, cluster_sizes, cluster_rmsd, n_noise, silhouette_avg = \
        cluster_torsions_DBSCAN(
            pdb_file_list,
            cgmodel,
            min_samples=min_samples,
            eps=eps,
            frame_start=frame_start,
            frame_stride=frame_stride,
            frame_end=-1,
            output_dir=output_directory,
            plot_silhouette=True,
            plot_distance_hist=True,
            filter=True,
            filter_ratio=0.20,
            core_points_only=False,
        )
    
    assert os.path.isfile(f"{output_directory}/medoid_0.pdb")
    assert os.path.isfile(f"{output_directory}/torsion_distances_hist.pdf")

    
def test_cluster_torsions_dbscan_dcd(tmpdir):
    """Test DBSCAN clustering"""
    
    output_directory = tmpdir.mkdir("output")
    
    # Load in cgmodel
    cgmodel_path = os.path.join(data_path, "stored_cgmodel.pkl")
    cgmodel = pickle.load(open(cgmodel_path, "rb"))
    
    # Create list of trajectory files for clustering analysis
    number_replicas = 12
    dcd_file_list = []
    for i in range(number_replicas):
        dcd_file_list.append(f"{data_path}/replica_%s.dcd" %(i+1))

    # Set clustering parameters
    min_samples=3
    eps=50
    frame_start=10
    frame_stride=2
    frame_end=-1

    # Run OPTICS density-based clustering
    medoid_positions, medoid_torsions, cluster_sizes, cluster_rmsd, n_noise, silhouette_avg = \
        cluster_torsions_DBSCAN(
            dcd_file_list,
            cgmodel,
            min_samples=min_samples,
            eps=eps,
            frame_start=frame_start,
            frame_stride=frame_stride,
            frame_end=-1,
            output_format="dcd",
            output_dir=output_directory,
            plot_silhouette=True,
            plot_distance_hist=True,
            filter=True,
            filter_ratio=0.20,
            core_points_only=False,
        )
    
    assert os.path.isfile(f"{output_directory}/medoid_0.dcd")    
    assert os.path.isfile(f"{output_directory}/torsion_distances_hist.pdf")
    
    
def test_cluster_torsions_dbscan_dcd_core_medoids(tmpdir):
    """Test DBSCAN clustering"""
    
    output_directory = tmpdir.mkdir("output")
    
    # Load in cgmodel
    cgmodel_path = os.path.join(data_path, "stored_cgmodel.pkl")
    cgmodel = pickle.load(open(cgmodel_path, "rb"))
    
    # Create list of trajectory files for clustering analysis
    number_replicas = 12
    dcd_file_list = []
    for i in range(number_replicas):
        dcd_file_list.append(f"{data_path}/replica_%s.dcd" %(i+1))

    # Set clustering parameters
    min_samples=3
    eps=50
    frame_start=10
    frame_stride=2
    frame_end=-1

    # Run OPTICS density-based clustering
    medoid_positions, medoid_torsions, cluster_sizes, cluster_rmsd, n_noise, silhouette_avg = \
        cluster_torsions_DBSCAN(
            dcd_file_list,
            cgmodel,
            min_samples=min_samples,
            eps=eps,
            frame_start=frame_start,
            frame_stride=frame_stride,
            frame_end=-1,
            output_format="dcd",
            output_dir=output_directory,
            plot_silhouette=True,
            plot_distance_hist=True,
            filter=True,
            filter_ratio=0.20,
            core_points_only=True,
        )
    
    assert os.path.isfile(f"{output_directory}/medoid_0.dcd")    
    assert os.path.isfile(f"{output_directory}/torsion_distances_hist.pdf") 
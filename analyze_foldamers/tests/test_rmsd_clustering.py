"""
Unit and regression test for the analyze_foldamers package.
"""

import os
import pickle
import sys

# Import package, test suite, and other packages as needed
import analyze_foldamers
import pytest
from analyze_foldamers.ensembles.cluster import *
from cg_openmm.cg_model.cgmodel import CGModel

current_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_path, 'test_data')

def test_clustering_kmedoids_pdb(tmpdir):
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
    frame_stride=1
    frame_end=-1

    # Run KMeans clustering
    medoid_positions, cluster_size, cluster_rmsd, silhouette_avg, labels, original_indices = \
        get_cluster_medoid_positions_KMedoids(
            pdb_file_list,
            cgmodel,
            n_clusters=n_clusters,
            frame_start=frame_start,
            frame_stride=frame_stride,
            frame_end=-1,
            output_dir=output_directory,
            plot_silhouette=True,
            plot_rmsd_hist=True,
            filter=True,
            filter_ratio=0.20,
        )
    
    assert len(cluster_rmsd) == n_clusters
    assert len(labels) == len(original_indices)
    assert os.path.isfile(f"{output_directory}/medoid_1.pdb")
    assert os.path.isfile(f"{output_directory}/silhouette_kmedoids_ncluster_{n_clusters}.pdf") 
    assert os.path.isfile(f"{output_directory}/distances_rmsd_hist.pdf")
    

def test_clustering_kmedoids_pdb_no_cgmodel(tmpdir):
    """Test Kmeans clustering without a cgmodel"""
    
    output_directory = tmpdir.mkdir("output")
    
    # Create list of trajectory files for clustering analysis
    number_replicas = 12
    pdb_file_list = []
    for i in range(number_replicas):
        pdb_file_list.append(f"{data_path}/replica_%s.pdb" %(i+1))

    # Set clustering parameters
    n_clusters=2
    frame_start=10
    frame_stride=1
    frame_end=-1

    # Run KMeans clustering
    medoid_positions, cluster_size, cluster_rmsd, silhouette_avg, labels, original_indices = \
        get_cluster_medoid_positions_KMedoids(
            pdb_file_list,
            cgmodel=None,
            n_clusters=n_clusters,
            frame_start=frame_start,
            frame_stride=frame_stride,
            frame_end=-1,
            output_dir=output_directory,
            plot_silhouette=True,
            plot_rmsd_hist=True,
            filter=True,
            filter_ratio=0.20,
        )
    
    assert len(cluster_rmsd) == n_clusters
    assert len(labels) == len(original_indices)
    assert os.path.isfile(f"{output_directory}/medoid_1.pdb")
    assert os.path.isfile(f"{output_directory}/silhouette_kmedoids_ncluster_{n_clusters}.pdf") 
    assert os.path.isfile(f"{output_directory}/distances_rmsd_hist.pdf")
    
    
def test_clustering_kmedoids_dcd(tmpdir):
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
    frame_stride=1
    frame_end=-1

    # Run KMeans clustering
    medoid_positions, cluster_size, cluster_rmsd, silhouette_avg, labels, original_indices = \
        get_cluster_medoid_positions_KMedoids(
            dcd_file_list,
            cgmodel,
            n_clusters=n_clusters,
            frame_start=frame_start,
            frame_stride=frame_stride,
            frame_end=-1,
            output_format="dcd",
            output_dir=output_directory,
            plot_silhouette=True,
            plot_rmsd_hist=True,
            filter=True,
            filter_ratio=0.20,
        )
    
    assert len(cluster_rmsd) == n_clusters
    assert len(labels) == len(original_indices)
    assert os.path.isfile(f"{output_directory}/medoid_1.dcd")
    assert os.path.isfile(f"{output_directory}/silhouette_kmedoids_ncluster_{n_clusters}.pdf")
    assert os.path.isfile(f"{output_directory}/distances_rmsd_hist.pdf")    
    
    
def test_clustering_dbscan_pdb(tmpdir):
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
    eps=0.5
    frame_start=10
    frame_stride=1
    frame_end=-1

    # Run DBSCAN density-based clustering
    medoid_positions, cluster_sizes, cluster_rmsd, n_noise, silhouette_avg, labels, original_indices = \
        get_cluster_medoid_positions_DBSCAN(
            pdb_file_list,
            cgmodel,
            min_samples=min_samples,
            eps=eps,
            frame_start=frame_start,
            frame_stride=frame_stride,
            frame_end=-1,
            output_dir=output_directory,
            plot_silhouette=True,
            plot_rmsd_hist=True,
            filter=True,
            filter_ratio=0.20,
            core_points_only=False,
        )
    
    assert len(labels) == len(original_indices)
    assert os.path.isfile(f"{output_directory}/medoid_0.pdb")
    assert os.path.isfile(f"{output_directory}/distances_rmsd_hist.pdf")
    
def test_clustering_dbscan_pdb_core_medoids(tmpdir):
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
    eps=0.5
    frame_start=10
    frame_stride=1
    frame_end=-1

    # Run DBSCAN density-based clustering
    medoid_positions, cluster_sizes, cluster_rmsd, n_noise, silhouette_avg, labels, original_indices = \
        get_cluster_medoid_positions_DBSCAN(
            pdb_file_list,
            cgmodel,
            min_samples=min_samples,
            eps=eps,
            frame_start=frame_start,
            frame_stride=frame_stride,
            frame_end=-1,
            output_dir=output_directory,
            plot_silhouette=True,
            plot_rmsd_hist=True,
            filter=True,
            filter_ratio=0.20,
            core_points_only=True,
        )
    
    assert len(labels) == len(original_indices)
    assert os.path.isfile(f"{output_directory}/medoid_0.pdb")
    assert os.path.isfile(f"{output_directory}/distances_rmsd_hist.pdf")
    

def test_clustering_dbscan_pdb_no_cgmodel(tmpdir):
    """Test DBSCAN clustering without cgmodel object"""
    
    output_directory = tmpdir.mkdir("output")
    
    # Create list of trajectory files for clustering analysis
    number_replicas = 12
    pdb_file_list = []
    for i in range(number_replicas):
        pdb_file_list.append(f"{data_path}/replica_%s.pdb" %(i+1))

    # Set clustering parameters
    min_samples=3
    eps=0.5
    frame_start=10
    frame_stride=1
    frame_end=-1

    # Run DBSCAN density-based clustering
    medoid_positions, cluster_sizes, cluster_rmsd, n_noise, silhouette_avg, labels, original_indices = \
        get_cluster_medoid_positions_DBSCAN(
            pdb_file_list,
            cgmodel = None,
            min_samples=min_samples,
            eps=eps,
            frame_start=frame_start,
            frame_stride=frame_stride,
            frame_end=-1,
            output_dir=output_directory,
            plot_silhouette=True,
            plot_rmsd_hist=True,
            filter=True,
            filter_ratio=0.20,
            core_points_only=False,
        )
    assert len(labels) == len(original_indices)
    assert os.path.isfile(f"{output_directory}/medoid_0.pdb")
    assert os.path.isfile(f"{output_directory}/distances_rmsd_hist.pdf")

    
def test_clustering_dbscan_dcd(tmpdir):
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
    eps=0.5
    frame_start=10
    frame_stride=1
    frame_end=-1

    # Run DBSCAN density-based clustering
    medoid_positions, cluster_sizes, cluster_rmsd, n_noise, silhouette_avg, labels, original_indices = \
        get_cluster_medoid_positions_DBSCAN(
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
            plot_rmsd_hist=True,
            filter=True,
            filter_ratio=0.20,
            core_points_only=False,
        )
    assert len(labels) == len(original_indices)
    assert os.path.isfile(f"{output_directory}/medoid_0.dcd")    
    assert os.path.isfile(f"{output_directory}/distances_rmsd_hist.pdf")    
    
    
def test_clustering_dbscan_dcd_homopolymer_sym(tmpdir):
    """Test DBSCAN clustering, with end-to-end symmetry check for a 1-1 homopolymer model"""
    
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
    eps=0.5
    frame_start=10
    frame_stride=1
    frame_end=-1

    # Run DBSCAN density-based clustering
    medoid_positions, cluster_sizes, cluster_rmsd, n_noise, silhouette_avg, labels, original_indices = \
        get_cluster_medoid_positions_DBSCAN(
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
            plot_rmsd_hist=True,
            homopolymer_sym=True,
            filter=True,
            filter_ratio=0.20,
            core_points_only=False,
        )
    assert len(labels) == len(original_indices)
    assert os.path.isfile(f"{output_directory}/medoid_0.dcd")
    assert os.path.isfile(f"{output_directory}/distances_rmsd_hist.pdf")      
    
    
def test_clustering_optics_pdb(tmpdir):
    """Test OPTICS clustering"""
    
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
    min_samples=5
    frame_start=10
    frame_stride=1
    frame_end=-1

    # Run OPTICS density-based clustering
    medoid_positions, cluster_sizes, cluster_rmsd, n_noise, silhouette_avg, labels, original_indices = \
        get_cluster_medoid_positions_OPTICS(
            pdb_file_list,
            cgmodel,
            min_samples=min_samples,
            frame_start=frame_start,
            frame_stride=frame_stride,
            frame_end=-1,
            output_dir=output_directory,
            plot_silhouette=True,
            plot_rmsd_hist=True,
            filter=True,
            filter_ratio=0.20,
        )
    assert len(labels) == len(original_indices)
    assert os.path.isfile(f"{output_directory}/medoid_0.pdb")
    assert os.path.isfile(f"{output_directory}/distances_rmsd_hist.pdf") 


def test_clustering_optics_pdb_no_cgmodel(tmpdir):
    """Test OPTICS clustering without a cgmodel object"""
    
    output_directory = tmpdir.mkdir("output")
    
    # Create list of trajectory files for clustering analysis
    number_replicas = 12
    pdb_file_list = []
    for i in range(number_replicas):
        pdb_file_list.append(f"{data_path}/replica_%s.pdb" %(i+1))

    # Set clustering parameters
    min_samples=5
    frame_start=10
    frame_stride=1
    frame_end=-1

    # Run OPTICS density-based clustering
    medoid_positions, cluster_sizes, cluster_rmsd, n_noise, silhouette_avg, labels, original_indices = \
        get_cluster_medoid_positions_OPTICS(
            pdb_file_list,
            cgmodel = None,
            min_samples=min_samples,
            frame_start=frame_start,
            frame_stride=frame_stride,
            frame_end=-1,
            output_dir=output_directory,
            plot_silhouette=True,
            plot_rmsd_hist=True,
            filter=True,
            filter_ratio=0.20,
        )
    assert len(labels) == len(original_indices)
    assert os.path.isfile(f"{output_directory}/medoid_0.pdb")
    assert os.path.isfile(f"{output_directory}/distances_rmsd_hist.pdf") 

    
def test_clustering_optics_dcd(tmpdir):
    """Test OPTICS clustering"""
    
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
    min_samples=5
    frame_start=10
    frame_stride=1
    frame_end=-1

    # Run OPTICS density-based clustering
    medoid_positions, cluster_sizes, cluster_rmsd, n_noise, silhouette_avg, labels, original_indices = \
        get_cluster_medoid_positions_OPTICS(
            dcd_file_list,
            cgmodel,
            min_samples=min_samples,
            frame_start=frame_start,
            frame_stride=frame_stride,
            frame_end=-1,
            output_format="dcd",
            output_dir=output_directory,
            plot_silhouette=True,
            plot_rmsd_hist=True,
            filter=True,
            filter_ratio=0.20,
        )
    assert len(labels) == len(original_indices)
    assert os.path.isfile(f"{output_directory}/medoid_0.dcd")   
    assert os.path.isfile(f"{output_directory}/distances_rmsd_hist.pdf")


def test_clustering_dbscan_pdb_output_clusters(tmpdir):
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
    eps=0.5
    frame_start=10
    frame_stride=1
    frame_end=-1

    # Run DBSCAN density-based clustering
    medoid_positions, cluster_sizes, cluster_rmsd, n_noise, silhouette_avg, labels, original_indices = \
        get_cluster_medoid_positions_DBSCAN(
            pdb_file_list,
            cgmodel,
            min_samples=min_samples,
            eps=eps,
            frame_start=frame_start,
            frame_stride=frame_stride,
            frame_end=-1,
            output_dir=output_directory,
            output_cluster_traj=True,
            plot_silhouette=True,
            plot_rmsd_hist=True,
            filter=True,
            filter_ratio=0.20,
        )
    assert len(labels) == len(original_indices)
    assert os.path.isfile(f"{output_directory}/medoid_0.pdb")
    assert os.path.isfile(f"{output_directory}/cluster_0.pdb")

"""
Unit and regression test for the analyze_foldamers package.
"""

# Import package, test suite, and other packages as needed
import analyze_foldamers
import pytest
import sys
import os
import pickle
from cg_openmm.cg_model.cgmodel import CGModel
from analyze_foldamers.ensembles.cluster import *
from analyze_foldamers.parameters.angle_distributions import *

def test_analyze_foldamers_imported():
    """Sample test, will always pass so long as import statement worked"""
    assert "analyze_foldamers" in sys.modules
    
current_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_path, 'test_data')
  
    
def test_clustering_pdb(tmpdir):
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
    frame_start=0
    frame_stride=1
    frame_end=-1

    # Run KMeans clustering
    medoid_positions, cluster_size, cluster_rmsd = get_cluster_medoid_positions(
        pdb_file_list,
        cgmodel,
        n_clusters=n_clusters,
        frame_start=frame_start,
        frame_stride=frame_stride,
        frame_end=-1,
        output_dir=output_directory,
    )
    
    assert len(cluster_rmsd) == n_clusters
    assert os.path.isfile(f"{output_directory}/medoid_1.pdb")

    
def test_clustering_dcd(tmpdir):
    """Test Kmeans clustering"""
    
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
    frame_start=0
    frame_stride=1
    frame_end=-1

    # Run KMeans clustering
    medoid_positions, cluster_size, cluster_rmsd = get_cluster_medoid_positions(
        dcd_file_list,
        cgmodel,
        n_clusters=n_clusters,
        frame_start=frame_start,
        frame_stride=frame_stride,
        frame_end=-1,
        output_format="dcd",
        output_dir=output_directory,
    )
    
    assert len(cluster_rmsd) == n_clusters
    assert os.path.isfile(f"{output_directory}/medoid_1.dcd")
    
    
def test_angle_dist_calc_pdb(tmpdir):
    """Test angle/torsion distribution calculators"""
    
    output_directory = tmpdir.mkdir("output")
    
    # Load in a trajectory pdb file:
    traj_file = os.path.join(data_path, "replica_1.pdb")

    # Load in a CGModel:
    cgmodel_path = os.path.join(data_path, "stored_cgmodel.pkl")
    cgmodel = pickle.load(open(cgmodel_path, "rb"))

    angle_hist_data = calc_bond_angle_distribution(
        cgmodel,
        traj_file,
        plotfile=f"{output_directory}/angle_hist",
    )
    
def test_torsion_dist_calc_pdb(tmpdir):
    """Test angle/torsion distribution calculators"""
    
    output_directory = tmpdir.mkdir("output")
    
    # Load in a trajectory pdb file:
    traj_file = os.path.join(data_path, "replica_1.pdb")

    # Load in a CGModel:
    cgmodel_path = os.path.join(data_path, "stored_cgmodel.pkl")
    cgmodel = pickle.load(open(cgmodel_path, "rb"))
    
    torsion_hist_data = calc_torsion_distribution(
        cgmodel,
        traj_file,
        plotfile=f"{output_directory}/torsion_hist",
    )
    
    
def test_angle_dist_calc_dcd(tmpdir):
    """Test angle/torsion distribution calculators"""
    
    output_directory = tmpdir.mkdir("output")
    
    # Load in a trajectory pdb file:
    traj_file = os.path.join(data_path, "replica_1.dcd")

    # Load in a CGModel:
    cgmodel_path = os.path.join(data_path, "stored_cgmodel.pkl")
    cgmodel = pickle.load(open(cgmodel_path, "rb"))

    angle_hist_data = calc_bond_angle_distribution(
        cgmodel,
        traj_file,
        plotfile=f"{output_directory}/angle_hist",
    )
    
    
def test_torsion_dist_calc_dcd(tmpdir):
    """Test angle/torsion distribution calculators"""
    
    output_directory = tmpdir.mkdir("output")
    
    # Load in a trajectory pdb file:
    traj_file = os.path.join(data_path, "replica_1.dcd")

    # Load in a CGModel:
    cgmodel_path = os.path.join(data_path, "stored_cgmodel.pkl")
    cgmodel = pickle.load(open(cgmodel_path, "rb"))
    
    torsion_hist_data = calc_torsion_distribution(
        cgmodel,
        traj_file,
        plotfile=f"{output_directory}/torsion_hist",
    )
        
    
    
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
    frame_start=0
    frame_stride=1
    frame_end=-1

    # Run KMeans clustering
    medoid_positions, cluster_size, cluster_rmsd, silhouette_avg = \
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
    frame_start=0
    frame_stride=1
    frame_end=-1

    # Run KMeans clustering
    medoid_positions, cluster_size, cluster_rmsd, silhouette_avg = \
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
    frame_start=0
    frame_stride=1
    frame_end=-1

    # Run OPTICS density-based clustering
    medoid_positions, cluster_sizes, cluster_rmsd, n_noise, silhouette_avg = \
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
        )
    
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
    frame_start=0
    frame_stride=1
    frame_end=-1

    # Run OPTICS density-based clustering
    medoid_positions, cluster_sizes, cluster_rmsd, n_noise, silhouette_avg = \
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
        )
    
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
    frame_start=0
    frame_stride=1
    frame_end=-1

    # Run OPTICS density-based clustering
    medoid_positions, cluster_sizes, cluster_rmsd, n_noise, silhouette_avg = \
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
    frame_start=0
    frame_stride=1
    frame_end=-1

    # Run OPTICS density-based clustering
    medoid_positions, cluster_sizes, cluster_rmsd, n_noise, silhouette_avg = \
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
    
    assert os.path.isfile(f"{output_directory}/medoid_0.dcd")   
    assert os.path.isfile(f"{output_directory}/distances_rmsd_hist.pdf") 
    
    
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
        plotfile=f"{output_directory}/angle_hist_pdb.pdf",
    )
    
    assert os.path.isfile(f"{output_directory}/angle_hist_pdb.pdf") 
    
    
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
        plotfile=f"{output_directory}/torsion_hist_pdb.pdf",
    )
    
    assert os.path.isfile(f"{output_directory}/torsion_hist_pdb.pdf") 
    
    
def test_angle_dist_calc_dcd(tmpdir):
    """Test angle/torsion distribution calculators"""
    
    output_directory = tmpdir.mkdir("output")
    
    # Load in a trajectory dcd file:
    traj_file = os.path.join(data_path, "replica_1.dcd")

    # Load in a CGModel:
    cgmodel_path = os.path.join(data_path, "stored_cgmodel.pkl")
    cgmodel = pickle.load(open(cgmodel_path, "rb"))

    angle_hist_data = calc_bond_angle_distribution(
        cgmodel,
        traj_file,
        plotfile=f"{output_directory}/angle_hist_dcd.pdf",
    )
    
    assert os.path.isfile(f"{output_directory}/angle_hist_dcd.pdf") 
    
    
def test_torsion_dist_calc_dcd(tmpdir):
    """Test angle/torsion distribution calculators"""
    
    output_directory = tmpdir.mkdir("output")
    
    # Load in a trajectory dcd file:
    traj_file = os.path.join(data_path, "replica_1.dcd")

    # Load in a CGModel:
    cgmodel_path = os.path.join(data_path, "stored_cgmodel.pkl")
    cgmodel = pickle.load(open(cgmodel_path, "rb"))
    
    torsion_hist_data = calc_torsion_distribution(
        cgmodel,
        traj_file,
        plotfile=f"{output_directory}/torsion_hist_dcd.pdf",
    )
    
    assert os.path.isfile(f"{output_directory}/torsion_hist_dcd.pdf") 

    
def test_ramachandran_calc_pdb(tmpdir):
    """Test ramachandran calculation/plotting"""

    output_directory = tmpdir.mkdir("output")
    
    # Load in a trajectory pdb file:
    traj_file = os.path.join(data_path, "replica_1.pdb")
    
    # Load in a CGModel:
    cgmodel_path = os.path.join(data_path, "stored_cgmodel.pkl")
    cgmodel = pickle.load(open(cgmodel_path, "rb"))    
    
    rama_hist, xedges, yedges = calc_ramachandran(
        cgmodel,
        traj_file,
        plotfile=f"{output_directory}/ramachandran_pdb.pdf",
    )
     
    assert os.path.isfile(f"{output_directory}/ramachandran_pdb.pdf") 
     
    # Fit ramachandran data to 2d Gaussian:
    param_opt, param_cov = fit_ramachandran_data(rama_hist, xedges, yedges)
    
    
def test_ramachandran_calc_dcd(tmpdir):
    """Test ramachandran calculation/plotting"""

    output_directory = tmpdir.mkdir("output")
    
    # Load in a trajectory pdb file:
    traj_file = os.path.join(data_path, "replica_1.dcd")
    
    # Load in a CGModel:
    cgmodel_path = os.path.join(data_path, "stored_cgmodel.pkl")
    cgmodel = pickle.load(open(cgmodel_path, "rb"))    
    
    rama_hist, xedges, yedges = calc_ramachandran(
        cgmodel,
        traj_file,
        plotfile=f"{output_directory}/ramachandran_dcd.pdf",
    )
     
    assert os.path.isfile(f"{output_directory}/ramachandran_dcd.pdf") 
     
    # Fit ramachandran data to 2d Gaussian:
    param_opt, param_cov = fit_ramachandran_data(rama_hist, xedges, yedges)    
    
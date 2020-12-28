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
from analyze_foldamers.parameters.bond_distributions import *
from analyze_foldamers.parameters.angle_distributions import *

def test_analyze_foldamers_imported():
    """Sample test, will always pass so long as import statement worked"""
    assert "analyze_foldamers" in sys.modules
    
current_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_path, 'test_data')
  

def test_bond_dist_calc_pdb(tmpdir):
    """Test bond distribution calculator"""
    
    output_directory = tmpdir.mkdir("output")
    
    # Load in a trajectory pdb file:
    traj_file = os.path.join(data_path, "replica_1.pdb")

    # Load in a CGModel:
    cgmodel_path = os.path.join(data_path, "stored_cgmodel.pkl")
    cgmodel = pickle.load(open(cgmodel_path, "rb"))

    bond_hist_data = calc_bond_length_distribution(
        cgmodel,
        traj_file,
        plotfile=f"{output_directory}/bond_hist_pdb.pdf",
    )
    
    assert os.path.isfile(f"{output_directory}/bond_hist_pdb.pdf") 
    
def test_angle_dist_calc_pdb(tmpdir):
    """Test angle distribution calculator"""
    
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
    """Test torsion distribution calculator"""
    
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
    
    
def test_bond_dist_calc_dcd(tmpdir):
    """Test bond distribution calculator"""
    
    output_directory = tmpdir.mkdir("output")
    
    # Load in a trajectory dcd file:
    traj_file = os.path.join(data_path, "replica_1.dcd")

    # Load in a CGModel:
    cgmodel_path = os.path.join(data_path, "stored_cgmodel.pkl")
    cgmodel = pickle.load(open(cgmodel_path, "rb"))

    bond_hist_data = calc_bond_length_distribution(
        cgmodel,
        traj_file,
        plotfile=f"{output_directory}/bond_hist_dcd.pdf",
    )
    
    assert os.path.isfile(f"{output_directory}/bond_hist_dcd.pdf") 
    
    
def test_bond_dist_calc_dcd_multi(tmpdir):
    """Test bond distribution calculator"""
    
    output_directory = tmpdir.mkdir("output")
    
    # Load in trajectory dcd files:
    traj_file_list = []
    for i in range(3):
        traj_file_list.append(os.path.join(data_path, f"replica_{i+1}.dcd"))

    # Load in a CGModel:
    cgmodel_path = os.path.join(data_path, "stored_cgmodel.pkl")
    cgmodel = pickle.load(open(cgmodel_path, "rb"))

    bond_hist_data = calc_bond_length_distribution(
        cgmodel,
        traj_file_list,
        plotfile=f"{output_directory}/bond_hist_dcd.pdf",
    )
    
    assert os.path.isfile(f"{output_directory}/bond_hist_dcd.pdf")     
    
    
def test_angle_dist_calc_dcd(tmpdir):
    """Test angle distribution calculator"""
    
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
    
    
def test_angle_dist_calc_dcd_multi(tmpdir):
    """Test angle distribution calculator"""
    
    output_directory = tmpdir.mkdir("output")
    
    # Load in trajectory dcd files:
    traj_file_list = []
    for i in range(3):
        traj_file_list.append(os.path.join(data_path, f"replica_{i+1}.dcd"))

    # Load in a CGModel:
    cgmodel_path = os.path.join(data_path, "stored_cgmodel.pkl")
    cgmodel = pickle.load(open(cgmodel_path, "rb"))

    angle_hist_data = calc_bond_angle_distribution(
        cgmodel,
        traj_file_list,
        plotfile=f"{output_directory}/angle_hist_dcd.pdf",
    )
    
    assert os.path.isfile(f"{output_directory}/angle_hist_dcd.pdf")     
    
    
def test_torsion_dist_calc_dcd(tmpdir):
    """Test torsion distribution calculator"""
    
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


def test_torsion_dist_calc_dcd_multi(tmpdir):
    """Test torsion distribution calculator"""
    
    output_directory = tmpdir.mkdir("output")
    
    # Load in trajectory dcd files:
    traj_file_list = []
    for i in range(3):
        traj_file_list.append(os.path.join(data_path, f"replica_{i+1}.dcd"))

    # Load in a CGModel:
    cgmodel_path = os.path.join(data_path, "stored_cgmodel.pkl")
    cgmodel = pickle.load(open(cgmodel_path, "rb"))
    
    torsion_hist_data = calc_torsion_distribution(
        cgmodel,
        traj_file_list,
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
    
    
def test_2d_dist_bond_bond_dcd(tmpdir):
    """Test general 2d histogram - bond-bond correlation"""
    output_directory = tmpdir.mkdir("output")
    
    # Load in a trajectory pdb file:
    traj_file = os.path.join(data_path, "replica_1.dcd")
    
    # Load in a CGModel:
    cgmodel_path = os.path.join(data_path, "stored_cgmodel.pkl")
    cgmodel = pickle.load(open(cgmodel_path, "rb"))

    hist_out, xedges, yedges = calc_2d_distribution(
        cgmodel,
        traj_file,
        plotfile=f"{output_directory}/bond_bond_2d.pdf",
        xvar_name='bb_bb',
        yvar_name='bb_sc',
    )
     
    assert os.path.isfile(f"{output_directory}/bond_bond_2d.pdf")
    
    
def test_2d_dist_bond_angle_dcd(tmpdir):
    """Test general 2d histogram - bond-angle correlation"""
    output_directory = tmpdir.mkdir("output")
    
    # Load in a trajectory pdb file:
    traj_file = os.path.join(data_path, "replica_1.dcd")
    
    # Load in a CGModel:
    cgmodel_path = os.path.join(data_path, "stored_cgmodel.pkl")
    cgmodel = pickle.load(open(cgmodel_path, "rb"))

    hist_out, xedges, yedges = calc_2d_distribution(
        cgmodel,
        traj_file,
        plotfile=f"{output_directory}/bond_angle_2d.pdf",
        xvar_name='bb_bb',
        yvar_name='bb_bb_sc',
    )
     
    assert os.path.isfile(f"{output_directory}/bond_angle_2d.pdf")     
    
    
def test_2d_dist_bond_torsion_dcd(tmpdir):
    """Test general 2d histogram - bond-torsion correlation"""
    output_directory = tmpdir.mkdir("output")
    
    # Load in a trajectory pdb file:
    traj_file = os.path.join(data_path, "replica_1.dcd")
    
    # Load in a CGModel:
    cgmodel_path = os.path.join(data_path, "stored_cgmodel.pkl")
    cgmodel = pickle.load(open(cgmodel_path, "rb"))

    hist_out, xedges, yedges = calc_2d_distribution(
        cgmodel,
        traj_file,
        plotfile=f"{output_directory}/bond_torsion_2d.pdf",
        xvar_name='bb_bb',
        yvar_name='bb_bb_bb_sc',
    )
     
    assert os.path.isfile(f"{output_directory}/bond_torsion_2d.pdf")
    

def test_2d_dist_angle_angle_dcd(tmpdir):
    """Test general 2d histogram - angle-angle correlation"""
    output_directory = tmpdir.mkdir("output")
    
    # Load in a trajectory pdb file:
    traj_file = os.path.join(data_path, "replica_1.dcd")
    
    # Load in a CGModel:
    cgmodel_path = os.path.join(data_path, "stored_cgmodel.pkl")
    cgmodel = pickle.load(open(cgmodel_path, "rb"))

    hist_out, xedges, yedges = calc_2d_distribution(
        cgmodel,
        traj_file,
        plotfile=f"{output_directory}/angle_angle_2d.pdf",
        xvar_name='bb_bb_bb',
        yvar_name='sc_bb_bb',
    )
     
    assert os.path.isfile(f"{output_directory}/angle_angle_2d.pdf")
    
    
def test_2d_dist_angle_torsion_dcd(tmpdir):
    """Test general 2d histogram - angle-torsion correlation"""
    output_directory = tmpdir.mkdir("output")
    
    # Load in a trajectory pdb file:
    traj_file = os.path.join(data_path, "replica_1.dcd")
    
    # Load in a CGModel:
    cgmodel_path = os.path.join(data_path, "stored_cgmodel.pkl")
    cgmodel = pickle.load(open(cgmodel_path, "rb"))

    hist_out, xedges, yedges = calc_2d_distribution(
        cgmodel,
        traj_file,
        plotfile=f"{output_directory}/angle_torsion_2d.pdf",
        xvar_name='bb_bb_sc',
        yvar_name='bb_bb_bb_sc',
    )
     
    assert os.path.isfile(f"{output_directory}/angle_torsion_2d.pdf")   


def test_2d_dist_torsion_torsion_dcd(tmpdir):
    """Test general 2d histogram - torsion-torsion correlation"""
    output_directory = tmpdir.mkdir("output")
    
    # Load in a trajectory pdb file:
    traj_file = os.path.join(data_path, "replica_1.dcd")
    
    # Load in a CGModel:
    cgmodel_path = os.path.join(data_path, "stored_cgmodel.pkl")
    cgmodel = pickle.load(open(cgmodel_path, "rb"))

    hist_out, xedges, yedges = calc_2d_distribution(
        cgmodel,
        traj_file,
        plotfile=f"{output_directory}/torsion_torsion_2d.pdf",
        xvar_name='bb_bb_bb_sc',
        yvar_name='sc_bb_bb_sc',
    )
     
    assert os.path.isfile(f"{output_directory}/torsion_torsion_2d.pdf")       
    
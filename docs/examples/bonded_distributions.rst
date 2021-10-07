Bonded distributions
=========================

Distributions in bond-stretching distances, bond-bending angles, and torsion angles
for all unique particle sequences are histogrammed over one or more trajectory
files and plotted.

In this example, topology from a CGModel is used to plot distributions in bond distances,
bond-bending angles, and torsion angles as a function of temperature, using a set
of constant-temperature trajectories created from a replica exchange MD simulation.

.. code-block:: python

    import os
    import pickle
    from cg_openmm.cg_model.cgmodel import CGModel
    from analyze_foldamers.parameters.angle_distributions import *
    from analyze_foldamers.parameters.bond_distributions import *

    # Specify path to directory with trajectory files:
    output_dir = "output_directory"
    
    # Load in cgmodel created with cg_openmm:
    cgmodel = pickle.load(open("stored_cgmodel.pkl","rb"))
    
    # Create list of trajectory files to analyze:
    traj_file_list = []
    number_replicas = 12
    for i in range(number_replicas):
        traj_file_list.append(f"{output_directory}/state_{i+1}.dcd")
    
    # Set the starting frame (i.e., after the equilibration period)
    frame_start = 20000
    
    bond_hist_data = calc_bond_length_distribution(
        cgmodel,
        traj_file_list, 
        frame_start=frame_start,
        plotfile=f"{output_dir}/bonds_all_states.pdf"
    )
        
    angle_hist_data = calc_bond_angle_distribution(
        cgmodel,
        traj_file_list,
        frame_start=frame_start,
        plotfile=f"{output_dir}/angles_all_states.pdf"
    )
        
    torsion_hist_data = calc_torsion_distribution(
        cgmodel,
        traj_file_list,
        frame_start=frame_start,
        plotfile=f"{output_dir}/torsions_all_states.pdf"
    )

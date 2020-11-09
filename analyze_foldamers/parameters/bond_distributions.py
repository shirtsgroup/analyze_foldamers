import os
import numpy as np
import mdtraj as md
from simtk import unit
from cg_openmm.cg_model.cgmodel import CGModel
from analyze_foldamers.utilities.plot import plot_distribution
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages

def assign_bond_types(cgmodel, bond_list):
    """Internal function for assigning bond types"""
    
    bond_types = []
    
    bond_array = np.zeros((len(bond_list),2))
    
    # Relevant bond types are added to a dictionary as they are discovered 
    bond_dict = {}
    
    # Create an inverse dictionary for getting bond string name from integer type
    inv_bond_dict = {}
    
    # Counter for number of bond types found:
    i_bond_type = 0
    
    # Assign bond types:
    
    for i in range(len(bond_list)):
        bond_array[i,0] = bond_list[i][0]
        bond_array[i,1] = bond_list[i][1]
        
        particle_types = [
            CGModel.get_particle_type_name(cgmodel,bond_list[i][0]),
            CGModel.get_particle_type_name(cgmodel,bond_list[i][1])
        ]
        
        string_name = ""
        reverse_string_name = ""
        for particle in particle_types:
            string_name += f"{particle}_"
        string_name = string_name[:-1]
        for particle in reversed(particle_types):
            reverse_string_name += f"{particle}_"
        reverse_string_name = reverse_string_name[:-1]
            
        if (string_name in bond_dict.keys()) == False:
            # New bond type found, add to bond dictionary
            i_bond_type += 1
            bond_dict[string_name] = i_bond_type
            bond_dict[reverse_string_name] = i_bond_type
            # For inverse dict we will use only the forward name based on first encounter
            inv_bond_dict[str(i_bond_type)] = string_name
            print(f"adding new bond type {i_bond_type}: {string_name} to dictionary")
            print(f"adding reverse version {i_bond_type}: {reverse_string_name} to dictionary")
            
        bond_types.append(bond_dict[string_name])

    # Sort bonds by type into separate sub arrays for mdtraj compute_distances
    bond_sub_arrays = {}
    for i in range(i_bond_type):
        bond_sub_arrays[str(i+1)] = np.zeros((bond_types.count(i+1),2))
    
    # Counter vector for all bond types
    n_i = np.zeros((i_bond_type,1), dtype=int)
    
    for i in range(len(bond_list)):
        bond_sub_arrays[str(bond_types[i])][n_i[bond_types[i]-1],:] = bond_array[i,:]
        n_i[bond_types[i]-1] += 1
        
    return bond_types, bond_array, bond_sub_arrays, n_i, i_bond_type, bond_dict, inv_bond_dict        
    
    
def calc_bond_length_distribution(
    cgmodel, file_list, nbins=90, frame_start=0, frame_stride=1, frame_end=-1,
    plot_per_page=2, plotfile="bond_hist.pdf"
    ):
    """
    Calculate and plot all bond length distributions from a CGModel object and trajectory

    :param cgmodel: CGModel() object
    :type cgmodel: class
    
    :param file_list: path to pdb or dcd trajectory file(s)
    :type file_list: str or list(str)
    
    :param nbins: number of histogram bins
    :type nbins: int
    
    :param frame_start: First frame in trajectory file to use for analysis.
    :type frame_start: int

    :param frame_stride: Advance by this many frames when reading trajectories.
    :type frame_stride: int

    :param frame_end: Last frame in trajectory file to use for analysis.
    :type frame_end: int
    
    :param plot_per_page: number of subplots to display on each page (default=2)
    :type plot_per_page: int
    
    :param plotfile: filename for saving bond length distribution pdf plots
    :type plotfile: str
    
    """   
    
    # Convert file_list to list if a single string:
    if type(file_list) == str:
        # Single file
        file_list = file_list.split()    
    
    # Create dictionary for saving bond histogram data:
    bond_hist_data = {}

    # Get bond list
    bond_list = CGModel.get_bond_list(cgmodel)
    
    # Assign bond types:
    bond_types, bond_array, bond_sub_arrays, n_i, i_bond_type, bond_dict, inv_bond_dict = \
        assign_bond_types(cgmodel, bond_list)
    
    for file in file_list:
    
        # Load in a trajectory file:
        if file[-3:] == 'dcd':
            traj = md.load(file,top=md.Topology.from_openmm(cgmodel.topology))
        else:
            traj = md.load(file)
            
        # Select frames for analysis:    
        if frame_end == -1:
            frame_end = traj.n_frames

        traj = traj[frame_start:frame_end:frame_stride]   
        
        nframes = traj.n_frames
            
        # Created inner dictionary for current file:
        # ***TODO: make this more general to file names other than 'output/state_i.dcd' form
        bond_hist_data[file[7:-4]] = {}
                
        for i in range(i_bond_type):
            # Compute all bond distances in trajectory
            # This returns an [nframes x n_bonds] array
            bond_val_array = md.compute_distances(traj,bond_sub_arrays[str(i+1)])
            
            # Reshape arrays:  
            bond_val_array = np.reshape(bond_val_array, (nframes*n_i[i][0],1))
            
            # Histogram and plot results:
            n_out, bin_edges_out = np.histogram(
                bond_val_array, bins=nbins, density=True)
            
            bond_bin_centers = np.zeros((len(bin_edges_out)-1,1))
            for j in range(len(bin_edges_out)-1):
                bond_bin_centers[j] = (bin_edges_out[j]+bin_edges_out[j+1])/2   
            
            bond_hist_data[file[7:-4]][f"{inv_bond_dict[str(i+1)]}_density"]=n_out
            bond_hist_data[file[7:-4]][f"{inv_bond_dict[str(i+1)]}_bin_centers"]=bond_bin_centers
        
    plot_distribution(
        inv_bond_dict,
        bond_hist_data,
        xlabel="Bond length (nm)",
        ylabel="Probability density",
        figure_title="Bond distributions",
        file_name=f"{plotfile}",
        plot_per_page=plot_per_page,
    )
        
    return bond_hist_data
        

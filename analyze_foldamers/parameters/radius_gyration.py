import os
import numpy as np
import mdtraj as md
from simtk import unit
from cg_openmm.cg_model.cgmodel import CGModel
from analyze_foldamers.utilities.plot import plot_distribution

def calc_radius_gyration(
    cgmodel, file_list, nbins=90, frame_start=0, frame_stride=1, frame_end=-1,
    temperature_list=None, plotfile="rg_distributions.pdf"
    ):
    """
    Calculate radius of gyration for all specified frames in each trajectory file,
    and plot the distributions as a function of temperature.

    :param cgmodel: CGModel() object. If None, only pdb trajectory option supported.
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
    
    :param temperature_list: list of temperatures corresponding to file_list. If None, file names will be the plot labels.
    :type temperature_list: list(Quantity())
    
    :param plotfile: filename for saving radius of gyration distribution pdf plots
    :type plotfile: str
    
    :returns:
       - rg_data ( dict mapping file_name:1D numpy array )
       - rg_hist_data ( dict mapping file_name to an inner dict of histogram density and bin_centers 1D numpy arrays )
    """   
       
    # Convert file_list to list if a single string:
    if type(file_list) == str:
        # Single file
        file_list = file_list.split()
    
    # Create dictionary for saving full rg and rg histogram data:
    rg_data = {}
    rg_hist_data = {}

    # TODO: get masses directly from the cgmodel instead of assuming constant masses
    
    file_index = 0
    for file in file_list:
        # Load in a trajectory file:
        if file[-3:] == 'dcd':
            if cgmodel is not None:
                traj = md.load(file,top=md.Topology.from_openmm(cgmodel.topology))
            else:
                print(f'A cgmodel with openmm topology must be supplied to use dcd files')
                exit()
        else:
            traj = md.load(file)
            
        # Select frames for analysis:    
        if frame_end == -1:
            frame_end = traj.n_frames

        traj = traj[frame_start:frame_end:frame_stride]   
        
        nframes = traj.n_frames
            
        # Create inner dictionary for current file:
        if temperature_list is not None:
            file_key = f"{temperature_list[file_index].value_in_unit(unit.kelvin):.2f}" 
        else:
            file_key = file[:-4]
            
        rg_hist_data[file_key] = {}
                
        # Compute radius of gyration for each frame in the trajectory
        # This returns a 1D [nframes] array
        rg_val_array = md.compute_rg(traj,masses=None)
        rg_data[file_key] = rg_val_array
        
        # Histogram and plot results:
        n_out, bin_edges_out = np.histogram(
            rg_val_array, bins=nbins, density=True)
        
        rg_bin_centers = np.zeros((len(bin_edges_out)-1,1))
        for j in range(len(bin_edges_out)-1):
            rg_bin_centers[j] = (bin_edges_out[j]+bin_edges_out[j+1])/2
        
        rg_hist_data[file_key][f"Rg_density"]=n_out
        rg_hist_data[file_key][f"Rg_bin_centers"]=rg_bin_centers
        
        file_index += 1
        
    # Set a dummy title dict for the distribution plotting function:
    # This needs to match the inner dictionary key prefix
    title_dict = {}
    title_dict[0] = 'Rg'
    
    plot_distribution(
        title_dict,
        rg_hist_data,
        xlabel="Radius of gyration (nm)",
        ylabel="Probability density",
        figure_title=None,
        file_name=f"{plotfile}",
        plot_per_page=1,
    )
    
    return rg_data, rg_hist_data
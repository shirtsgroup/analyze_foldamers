import os
import numpy as np
import mdtraj as md
from simtk import unit
from cg_openmm.cg_model.cgmodel import CGModel
from analyze_foldamers.utilities.plot import plot_distribution
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.optimize import curve_fit

# These functions calculate and plot bond angle and torsion distributions from a CGModel object and pdb trajectory

def assign_angle_types(cgmodel, angle_list):
    """Internal function for assigning angle types"""
    
    ang_types = [] # List of angle types for each angle in angle_list
    
    ang_array = np.zeros((len(angle_list),3))
    
    # Relevant angle types are added to a dictionary as they are discovered 
    ang_dict = {}
    
    # Create an inverse dictionary for getting angle string name from integer type
    inv_ang_dict = {}
    
    # Counter for number of angle types found:
    i_angle_type = 0
    
    # Assign angle types:
    
    for i in range(len(angle_list)):
        ang_array[i,0] = angle_list[i][0]
        ang_array[i,1] = angle_list[i][1]
        ang_array[i,2] = angle_list[i][2]
        
        particle_types = [
            CGModel.get_particle_type_name(cgmodel,angle_list[i][0]),
            CGModel.get_particle_type_name(cgmodel,angle_list[i][1]),
            CGModel.get_particle_type_name(cgmodel,angle_list[i][2])
        ]
        
        string_name = ""
        reverse_string_name = ""
        for particle in particle_types:
            string_name += f"{particle}_"
        string_name = string_name[:-1]
        for particle in reversed(particle_types):
            reverse_string_name += f"{particle}_"
        reverse_string_name = reverse_string_name[:-1]
            
        if (string_name in ang_dict.keys()) == False:
            # New angle type found, add to angle dictionary
            i_angle_type += 1
            ang_dict[string_name] = i_angle_type
            ang_dict[reverse_string_name] = i_angle_type
            # For inverse dict we will use only the forward name based on first encounter
            inv_ang_dict[str(i_angle_type)] = string_name
            print(f"adding new angle type {i_angle_type}: {string_name} to dictionary")
            print(f"adding reverse version {i_angle_type}: {reverse_string_name} to dictionary")
            
        ang_types.append(ang_dict[string_name])
                    
    # Sort angles by type into separate sub arrays for mdtraj compute_angles
    ang_sub_arrays = {}
    for i in range(i_angle_type):
        ang_sub_arrays[str(i+1)] = np.zeros((ang_types.count(i+1),3))
    
    # Counter vector for all angle types
    n_i = np.zeros((i_angle_type,1), dtype=int)
    
    for i in range(len(angle_list)):
        ang_sub_arrays[str(ang_types[i])][n_i[ang_types[i]-1],:] = ang_array[i,:]
        n_i[ang_types[i]-1] += 1
        
    return ang_types, ang_array, ang_sub_arrays, n_i, i_angle_type, ang_dict, inv_ang_dict
    

def assign_torsion_types(cgmodel, torsion_list):
    """Internal function for assigning torsion types"""
    
    torsion_types = [] # List of torsion types for each torsion in torsion_list
    torsion_array = np.zeros((len(torsion_list),4))
    
    # Relevant torsion types are added to a dictionary as they are discovered 
    torsion_dict = {}
    
    # Create an inverse dictionary for getting torsion string name from integer type
    inv_torsion_dict = {}
    
    # Counter for number of torsion types found:
    i_torsion_type = 0  
    
    for i in range(len(torsion_list)):
        torsion_array[i,0] = torsion_list[i][0]
        torsion_array[i,1] = torsion_list[i][1]
        torsion_array[i,2] = torsion_list[i][2]
        torsion_array[i,3] = torsion_list[i][3]
        
        particle_types = [
            CGModel.get_particle_type_name(cgmodel,torsion_list[i][0]),
            CGModel.get_particle_type_name(cgmodel,torsion_list[i][1]),
            CGModel.get_particle_type_name(cgmodel,torsion_list[i][2]),
            CGModel.get_particle_type_name(cgmodel,torsion_list[i][3])
        ]
        
        string_name = ""
        reverse_string_name = ""
        for particle in particle_types:
            string_name += f"{particle}_"
        string_name = string_name[:-1]
        for particle in reversed(particle_types):
            reverse_string_name += f"{particle}_"
        reverse_string_name = reverse_string_name[:-1]
            
        if (string_name in torsion_dict.keys()) == False:
            # New torsion type found, add to torsion dictionary
            i_torsion_type += 1
            torsion_dict[string_name] = i_torsion_type
            torsion_dict[reverse_string_name] = i_torsion_type
            # For inverse dict we will use only the forward name based on first encounter
            inv_torsion_dict[str(i_torsion_type)] = string_name
            
            print(f"adding new torsion type {i_torsion_type}: {string_name} to dictionary")
            print(f"adding reverse version {i_torsion_type}: {reverse_string_name} to dictionary")
            
            
        torsion_types.append(torsion_dict[string_name])
                        
    # Sort torsions by type into separate sub arrays for mdtraj compute_dihedrals
    torsion_sub_arrays = {}
    for i in range(i_torsion_type):
        torsion_sub_arrays[str(i+1)] = np.zeros((torsion_types.count(i+1),4))
    
    # Counter vector for all angle types
    n_i = np.zeros((i_torsion_type,1), dtype=int) 
    
    for i in range(len(torsion_list)):
        torsion_sub_arrays[str(torsion_types[i])][n_i[torsion_types[i]-1],:] = torsion_array[i,:]
        n_i[torsion_types[i]-1] += 1
        
    return torsion_types, torsion_array, torsion_sub_arrays, n_i, i_torsion_type, torsion_dict, inv_torsion_dict


def calc_bond_angle_distribution(
    cgmodel, file, nbins=90, frame_start=0, frame_stride=1, frame_end=-1, plotfile="angle_hist.pdf"
    ):
    """
    Calculate and plot all bond angle distributions from a CGModel object and pdb trajectory

    :param cgmodel: CGModel() object
    :type cgmodel: class
    
    :param file: path to pdb or dcd trajectory file
    :type file: str
    
    :param nbins: number of bins spanning the range of 0 to 180 degrees, default = 90
    :type nbins: int
    
    :param frame_start: First frame in trajectory file to use for analysis.
    :type frame_start: int

    :param frame_stride: Advance by this many frames when reading trajectories.
    :type frame_stride: int

    :param frame_end: Last frame in trajectory file to use for analysis.
    :type frame_end: int
    
    :param plotfile: Base filename for saving bond angle distribution pdf plots
    :type plotfile: str
    
    """
    
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
    
    # Get angle list
    angle_list = CGModel.get_bond_angle_list(cgmodel)
    
    # Assign angle types:
    ang_types, ang_array, ang_sub_arrays, n_i, i_angle_type, ang_dict, inv_ang_dict = \
        assign_angle_types(cgmodel, angle_list)
         
    # Create dictionary for saving angle histogram data:
    angle_hist_data = {}
    
    # Set bin edges:
    angle_bin_edges = np.linspace(0,180,nbins+1)
    angle_bin_centers = np.zeros((len(angle_bin_edges)-1,1))
    for i in range(len(angle_bin_edges)-1):
        angle_bin_centers[i] = (angle_bin_edges[i]+angle_bin_edges[i+1])/2
            
    for i in range(i_angle_type):
        # Compute all angle values in trajectory
        # This returns an [nframes x n_angles] array
        ang_val_array = md.compute_angles(traj,ang_sub_arrays[str(i+1)])
        
        # Reshape arrays and convert to degrees:  
        ang_val_array = (180/np.pi)*np.reshape(ang_val_array, (nframes*n_i[i][0],1))
        
        # Histogram and plot results:
        
        n_out, bin_edges_out = np.histogram(
            ang_val_array, bins=angle_bin_edges,density=True)
            
        
        angle_hist_data[f"{inv_ang_dict[str(i+1)]}_density"]=n_out
        angle_hist_data[f"{inv_ang_dict[str(i+1)]}_bin_centers"]=angle_bin_centers
        
    plot_distribution(
        inv_ang_dict,
        angle_hist_data,
        xlabel="Bond angle (degrees)",
        ylabel="Probability density",
        xlim=[0,180],
        figure_title="Angle distributions",
        file_name=f"{plotfile}",
        marker_string='o-r',
    )
        
    return angle_hist_data
    
    
def calc_torsion_distribution(
    cgmodel, file, nbins=180, frame_start=0, frame_stride=1, frame_end=-1, plotfile="torsion_hist.pdf"
    ):
    """
    Calculate and plot all torsion distributions from a CGModel object and pdb or dcd trajectory

    :param cgmodel: CGModel() object
    :type cgmodel: class
    
    :param file: path to pdb or dcd trajectory file
    :type file: str
    
    :param nbins: number of bins spanning the range of -180 to 180 degrees, default = 180
    :type nbins: int
    
    :param frame_start: First frame in trajectory file to use for analysis.
    :type frame_start: int

    :param frame_stride: Advance by this many frames when reading trajectories.
    :type frame_stride: int

    :param frame_end: Last frame in trajectory file to use for analysis.
    :type frame_end: int
    
    :param plotfile: Base filename for saving torsion distribution pdf plots
    :type plotfile: str
    
    """
    
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
    
    # Get torsion list
    torsion_list = CGModel.get_torsion_list(cgmodel)
    
    # Assign torsion types
    torsion_types, torsion_array, torsion_sub_arrays, n_i, i_torsion_type, torsion_dict, inv_torsion_dict = \
        assign_torsion_types(cgmodel, torsion_list)
    
    # Create dictionary for saving torsion histogram data:
    torsion_hist_data = {}
    
    # Set bin edges:
    torsion_bin_edges = np.linspace(-180,180,nbins+1)
    torsion_bin_centers = np.zeros((len(torsion_bin_edges)-1,1))
    for i in range(len(torsion_bin_edges)-1):
        torsion_bin_centers[i] = (torsion_bin_edges[i]+torsion_bin_edges[i+1])/2
        
    for i in range(i_torsion_type):
        # Compute all torsion values in trajectory
        # This returns an [nframes x n_torsions] array
        torsion_val_array = md.compute_dihedrals(
            traj,torsion_sub_arrays[str(i+1)])
        
        # Reshape arrays and convert to degrees:  
        torsion_val_array = (180/np.pi)*np.reshape(torsion_val_array, (nframes*n_i[i][0],1))
        
        # Histogram and plot results:
        n_out, bin_edges_out = np.histogram(
            torsion_val_array, bins=torsion_bin_edges,density=True)
        
        torsion_hist_data[f"{inv_torsion_dict[str(i+1)]}_density"]=n_out
        torsion_hist_data[f"{inv_torsion_dict[str(i+1)]}_bin_centers"]=torsion_bin_centers  
      
    plot_distribution(
        inv_torsion_dict,
        torsion_hist_data,
        xlabel="Torsion angle (degrees)",
        ylabel="Probability density",
        xlim=[-180,180],
        figure_title="Torsion_distributions",
        file_name=f"{plotfile}",
        marker_string='o-b',
    )
      
    return torsion_hist_data
    

def calc_ramachandran(
    cgmodel,
    file_list,
    nbin_theta=180,
    nbin_alpha=180,
    frame_start=0,
    frame_stride=1,
    frame_end=-1,
    plotfile="ramachandran.pdf",
    backbone_angle_type = "bb_bb_bb",
    backbone_torsion_type = "bb_bb_bb_bb",
    colormap="Spectral",
):
    """
    Calculate and plot ramachandran plot for backbone bond bending-angle and torsion
    angle, given a CGModel object and pdb or dcd trajectory.

    :param cgmodel: CGModel() object
    :type cgmodel: class
    
    :param file_list: path to pdb or dcd trajectory file(s) - can be a list or single string
    :type file_list: str of list[str]
    
    :param nbin_theta: number of bins for bond-bending angle (spanning from 0 to 180 degrees)
    :type nbin_theta: int
    
    :param nbin_alpha: number of bins for torsion angle (spanning from -180 to +180 degrees)
    :type nbin_alpha:
    
    :param frame_start: First frame in trajectory file to use for analysis.
    :type frame_start: int

    :param frame_stride: Advance by this many frames when reading trajectories.
    :type frame_stride: int

    :param frame_end: Last frame in trajectory file to use for analysis.
    :type frame_end: int
    
    :param plotfile: Filename for saving torsion distribution pdf plots
    :type plotfile: str
    
    :param backbone_angle_type: particle sequence of the backbone angles (default="bb_bb_bb") - for now only single sequence permitted
    :type backbone_angle_type: str
    
    :param backbone_torsion_type: particle sequence of the backbone angles (default="bb_bb_bb_bb") - for now only single sequence permitted
    :type backbone_torsion_type: str    
    
    :param colormap: matplotlib pyplot colormap to use (default='Spectral')
    :type colormap: str (case sensitive)
    
    """
    
    # Convert file_list to list if a single string:
    if type(file_list) == str:
        # Single file
        file_list = file_list.split()
    
    # Store angle, torsion values by filename for computing global colormap
    ang_val_array = {}
    torsion_val_array = {}
    
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
        
        # Get angle list
        angle_list = CGModel.get_bond_angle_list(cgmodel)
        
        # Assign angle types:
        ang_types, ang_array, ang_sub_arrays, n_i, i_angle_type, ang_dict, inv_ang_dict = \
            assign_angle_types(cgmodel, angle_list)
        
        # Set bin edges:
        angle_bin_edges = np.linspace(0,180,nbin_theta+1)
        angle_bin_centers = np.zeros((len(angle_bin_edges)-1,1))
        for i in range(len(angle_bin_edges)-1):
            angle_bin_centers[i] = (angle_bin_edges[i]+angle_bin_edges[i+1])/2
                       
        for i in range(i_angle_type):
            if inv_ang_dict[str(i+1)] == backbone_angle_type:
                # Compute all angle values in trajectory
                # This returns an [nframes x n_angles] array
                ang_val_array[file] = md.compute_angles(traj,ang_sub_arrays[str(i+1)])
                
                # We will have different numbers of bond-bending angle and torsion angle.
                # We will set a convention of omitting the last angle value.
                
                # Convert to degrees and exclude last angle:  
                ang_val_array[file] = (180/np.pi)*ang_val_array[file][:,:-1]
                
                # Reshape array:
                ang_val_array[file] = np.reshape(ang_val_array[file], (nframes*(n_i[i]-1)[0],1))
            
        # Get torsion list
        torsion_list = CGModel.get_torsion_list(cgmodel)

        # Assign torsion types
        torsion_types, torsion_array, torsion_sub_arrays, n_j, i_torsion_type, torsion_dict, inv_torsion_dict = \
            assign_torsion_types(cgmodel, torsion_list)
        
        # Set bin edges:
        torsion_bin_edges = np.linspace(-180,180,nbin_alpha+1)
        torsion_bin_centers = np.zeros((len(torsion_bin_edges)-1,1))
        for i in range(len(torsion_bin_edges)-1):
            torsion_bin_centers[i] = (torsion_bin_edges[i]+torsion_bin_edges[i+1])/2
            
        for i in range(i_torsion_type):
            if inv_torsion_dict[str(i+1)] == backbone_torsion_type:
                # Compute all torsion values in trajectory
                # This returns an [nframes x n_torsions] array
                torsion_val_array[file] = md.compute_dihedrals(
                    traj,torsion_sub_arrays[str(i+1)])
                
                # Convert to degrees:  
                torsion_val_array[file] = (180/np.pi)*torsion_val_array[file]
                
                # Reshape array
                torsion_val_array[file] = np.reshape(torsion_val_array[file], (nframes*n_j[i][0],1))
        
    # 2d histogram the data and plot:
    hist_data, xedges, yedges = plot_ramachandran(
        file_list, ang_val_array, torsion_val_array, angle_bin_edges, torsion_bin_edges,
        plotfile, colormap)
    
    return hist_data, xedges, yedges
    
    
def plot_ramachandran(file_list, ang_val_array, torsion_val_array, angle_bin_edges, torsion_bin_edges,
    plotfile, colormap): 
    """Internal function for 2d histogramming angle data and creating ramachandran plots"""
    
    # Store 2d histogram output
    hist_data = {}
    xedges = {}
    yedges = {}
    image = {}
    
    # Determine optimal subplot layout
    nseries = len(file_list)
    nrow = int(np.ceil(np.sqrt(nseries)))
    ncol = int(np.ceil(nseries/nrow))
    
    # Initialize plot
    figure, axs = plt.subplots(
        nrows=nrow,
        ncols=ncol,
        figsize=(11,8.5),
        constrained_layout=True,
        sharex=True,
        sharey=True)
        
    subplot_id = 1
    
    for file in file_list:
    
        row = int(np.ceil(subplot_id/ncol))-1
        col = int((subplot_id-1)%ncol)
        
        # axs subplot object is only subscriptable in dimensions it has multiple entries in
        if nrow > 1 and ncol > 1: 
            hist_data_out, xedges_out, yedges_out, image_out = axs[row,col].hist2d(
                torsion_val_array[file][:,0],
                ang_val_array[file][:,0],
                bins=[torsion_bin_edges,angle_bin_edges],
                density=True,
            )
        
        if ncol > 1 and nrow == 1:
            hist_data_out, xedges_out, yedges_out, image_out = axs[col].hist2d(
                torsion_val_array[file][:,0],
                ang_val_array[file][:,0],
                bins=[torsion_bin_edges,angle_bin_edges],
                density=True,
            )
            
        if ncol == 1 and nrow == 1:
            hist_data_out, xedges_out, yedges_out, image_out = axs.hist2d(
                torsion_val_array[file][:,0],
                ang_val_array[file][:,0],
                bins=[torsion_bin_edges,angle_bin_edges],
                density=True,
            )        
        
        hist_data[file[:-4]] = hist_data_out
        xedges[file[:-4]] = xedges_out
        yedges[file[:-4]] = yedges_out
        image[file[:-4]] = image_out   
        
        #axs[row,col].set_title(file)
        
        subplot_id += 1
        
    plt.close()    
        
    # Renormalize data to global maximum:
    max_global = 0
    for key, val in hist_data.items():
        max_i = np.amax(val)
        if max_i > max_global:
            max_global = max_i  
       
    # Re-initialize plot
    figure, axs = plt.subplots(
        nrows=nrow,
        ncols=ncol,
        figsize=(11,8.5),
        constrained_layout=True,
        sharex=True,
        sharey=True)  

    # Update the colormap normalization:   
    cmap=plt.get_cmap(colormap)     
    norm=matplotlib.colors.Normalize(vmin=0,vmax=max_global)           
       
    subplot_id = 1
       
    for file in file_list:
        # Renormalize quadmesh images
        # There should be a way to do this using QuadMesh.set_norm.
        # For now just replot the data:
        row = int(np.ceil(subplot_id/ncol))-1
        col = int((subplot_id-1)%ncol)
        
        if nrow > 1 and ncol > 1: 
            hist_data_out, xedges_out, yedges_out, image_out = axs[row,col].hist2d(
                torsion_val_array[file][:,0],
                ang_val_array[file][:,0],
                bins=[torsion_bin_edges,angle_bin_edges],
                density=True,
                cmap=cmap,
                norm=norm,
            )
        
        if ncol > 1 and nrow == 1:
            hist_data_out, xedges_out, yedges_out, image_out = axs[col].hist2d(
                torsion_val_array[file][:,0],
                ang_val_array[file][:,0],
                bins=[torsion_bin_edges,angle_bin_edges],
                density=True,
                cmap=cmap,
                norm=norm,
            )
            
        if ncol == 1 and nrow == 1:
            hist_data_out, xedges_out, yedges_out, image_out = axs.hist2d(
                torsion_val_array[file][:,0],
                ang_val_array[file][:,0],
                bins=[torsion_bin_edges,angle_bin_edges],
                density=True,
                cmap=cmap,
                norm=norm,
            )                
        
        
        hist_data[file[:-4]] = hist_data_out
        xedges[file[:-4]] = xedges_out
        yedges[file[:-4]] = yedges_out
        image[file[:-4]] = image_out        
        
        # Add colorbar to right side
        
        if nrow > 1 and ncol > 1:
            plt.colorbar(
                matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap),
                ax=axs[:,col],
                shrink=0.6,
                label='probability density')
                    
        if ncol > 1 and nrow == 1:
            plt.colorbar(
                matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap),
                ax=axs[col],
                shrink=0.6,
                label='probability density')

        if nrow == 1 and ncol == 1:
            plt.colorbar(
                matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap),
                ax=axs,
                shrink=0.6,
                label='probability density')            
    
        subplot_id += 1
        
    
    # Add common axis labels:
    # We need to scale the axes spanning all subplots to avoid text overlap
    ax_common = figure.add_subplot(1, 1, 1, frameon=False)
    
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.xlabel("Alpha (degrees)", fontsize=14, labelpad=15)
    plt.ylabel("Theta (degrees)", fontsize=14, labelpad=15)
    
    plt.savefig(plotfile)
    plt.close()

    return hist_data, xedges, yedges
    
    
def fit_ramachandran_data(hist_data, xedges, yedges):
    """
    Calculate and plot ramachandran plot for backbone bond bending-angle and torsion
    angle, given a CGModel object and pdb or dcd trajectory.

    :param hist_data: dictionary containing 2D histogram data for one or more data series
    :type hist_data: dict {series_name: 2D numpy array}
    
    :param xedges: dictionary containing bin edges corresponding to x dimension of hist_data
    :type xedges: dict {series_name: 1D numpy array}
    
    :param yedges: dictionary containing bin edges corresponding to y dimension of hist_data
    :type yedges: dict {series_name: 1D numpy array}
    
    """
    
    # Fit to 2d symmetric gaussian:
    def two_gauss(xy,a,x0,y0,c):
        x, y = xy
        return a*np.exp(-((np.power((x-x0),2) + 0.5*np.power((y-y0),2))/(2*np.power(c,2))))
    
    param_opt = {}
    param_cov = {}
        
    for key,value in hist_data.items():
        z_data = value
        max_xy_index = np.unravel_index(np.argmax(z_data, axis=None), z_data.shape)
        max_x = (xedges[key][max_xy_index[0]]+xedges[key][max_xy_index[0]+1])/2
        max_y = (yedges[key][max_xy_index[1]]+yedges[key][max_xy_index[1]+1])/2
        param_guess = [0.01,max_x,max_y,5]
        x_centers = xedges[key][:-1] + (xedges[key][1]-xedges[key][0])/2
        y_centers = yedges[key][:-1] + (yedges[key][1]-yedges[key][0])/2
        x_data, y_data = np.meshgrid(x_centers, y_centers)
        
        # Ravel x,y data to a pair of 1D arrays
        xy_data = np.vstack((x_data.ravel(), y_data.ravel()))
        
        param_opt[key], param_cov[key] = curve_fit(two_gauss, xy_data, z_data.ravel(), param_guess)
    
    return param_opt, param_cov
    

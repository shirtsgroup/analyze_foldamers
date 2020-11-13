import os
import simtk.unit as unit
import mdtraj as md
import numpy as np
from sklearn.cluster import KMeans, OPTICS, DBSCAN
from sklearn.metrics import silhouette_score, silhouette_samples
from analyze_foldamers.parameters.angle_distributions import *
from analyze_foldamers.ensembles.cluster import *
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from cg_openmm.cg_model.cgmodel import CGModel
from cg_openmm.utilities.iotools import write_pdbfile_without_topology
from sklearn_extra.cluster import KMedoids
from scipy.optimize import minimize    
from scipy.spatial.distance import pdist, squareform


def cluster_torsions_KMedoids(
    file_list, cgmodel, n_clusters=2,
    frame_start=0, frame_stride=1, frame_end=-1,
    output_format="pdb", output_dir="cluster_output",
    backbone_torsion_type="bb_bb_bb_bb",
    filter=False, filter_ratio=0.05, plot_silhouette=True, plot_distance_hist=True):
    """
    Given PDB or DCD trajectory files and coarse grained model as input, this function performs K-medoids clustering on the poses in trajectory, and returns a list of the coordinates for the medoid pose of each cluster.

    :param file_list: A list of PDB or DCD files to read and concatenate
    :type file_list: List( str )

    :param cgmodel: A CGModel() class object
    :type cgmodel: class

    :param n_clusters: The number of clusters for K-medoids algorithm.
    :type n_clusters: int

    :param frame_start: First frame in pdb trajectory file to use for clustering.
    :type frame_start: int

    :param frame_stride: Advance by this many frames when reading pdb trajectories.
    :type frame_stride: int

    :param frame_end: Last frame in trajectory file to use for clustering.
    :type frame_end: int
    
    :param output_format: file format extension to write medoid coordinates to (default="pdb"), dcd also supported
    :type output_format: str
    
    :param output_dir: path to which cluster medoid structures and silhouette plots will be saved
    :type output_dir: str
    
    :param backbone_torsion_type: particle sequence of the backbone torsions (default="bb_bb_bb_bb") - for now only single sequence permitted
    :type backbone_torsion_type: str
    
    :param filter: option to apply neighborhood radius filtering to remove low-density data (default=False)
    :type filter: boolean
    
    :param filter_ratio: fraction of data points which pass through the neighborhood radius filter (default=0.05)
    :type filter_ratio: float
    
    :param plot_silhouette: option to create silhouette plot of clustering results (default=True)
    :type plot_silhouette: boolean
    
    :param plot_torsion_hist: option to plot a histogram of torsion euclidean distances (post-filtering)
    :type plot_torsion_hist: boolean
    
    :returns:
       - medoid_positions ( np.array( float * unit.angstrom ( n_clusters x num_particles x 3 ) ) ) - A 3D numpy array of poses corresponding to the medoids of all trajectory clusters.
       - medoid torsions ( np.array ( float * unit.degrees ( n_clusters x n_torsion ) - A 2D numpy array of the backbone torsion angles for each cluster medoid
       - cluster_sizes ( List ( int ) ) - A list of number of members in each cluster 
       - cluster_rmsd( np.array ( float ) ) - A 1D numpy array of rmsd (in cluster distance space) of samples to cluster centers
       - silhouette_avg - ( float ) - average silhouette score across all clusters
    """  
    
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    
    torsion_val_array, traj_all = get_torsion_matrix(file_list, cgmodel, frame_start, frame_stride, frame_end, backbone_torsion_type)
    
    # We need to precompute the euclidean distance matrix, accounting for periodic boundaries
    
    total = 0
    angle_range = np.full(torsion_val_array.shape[1],360)
    powers = np.full(torsion_val_array.shape[1],2)

    torsion_distances = np.zeros((torsion_val_array.shape[0], torsion_val_array.shape[0]))
    
    for i in range(torsion_val_array.shape[0]):
        for j in range(torsion_val_array.shape[0]): 
            delta = np.abs(torsion_val_array[i,:]-torsion_val_array[j,:])
            delta = np.where(delta > 0.5*angle_range, delta-angle_range, delta)
            torsion_distances[i,j] = np.sqrt(np.power(delta,powers).sum())
            
    if filter:
        # Filter distances:
        torsion_distances, dense_indices, filter_ratio_actual = \
            filter_distances(torsion_distances, filter_ratio=filter_ratio)
        
        traj_all = traj_all[dense_indices]
     
    if plot_distance_hist:
        distances_row = np.reshape(torsion_distances, (torsion_distances.shape[0]*torsion_distances.shape[1],1))
        
        # Remove the diagonal 0 elements:
        distances_row = distances_row[distances_row != 0]
        
        figure = plt.figure()
        n_out, bin_edges_out, patch = plt.hist(
            distances_row, bins=1000,density=True)
        plt.xlabel('rmsd')
        plt.ylabel('probability density')
        plt.savefig(f'{output_dir}/torsion_distances_hist.pdf')
        plt.close()  
        
    # Cluster with sklearn-extra KMedoids
    kmedoids = KMedoids(n_clusters=n_clusters,metric='precomputed').fit(torsion_distances)

    # Get labels
    labels = kmedoids.labels_
    
    # Get medoid indices
    medoid_indices = kmedoids.medoid_indices_
    
    # Get medoid coordinates:
    medoid_xyz = np.zeros([n_clusters,traj_all.n_atoms,3])
    for k in range(n_clusters):
        medoid_xyz[k,:,:] = traj_all[medoid_indices[k]].xyz[0]
        
    # Get medoid torsions:
    medoid_torsions = np.zeros([n_clusters, torsion_val_array.shape[1]])
    for k in range(n_clusters):
        medoid_torsions[k,:] = torsion_val_array[medoid_indices[k],:]
        
    # Write medoids to file
    write_medoids_to_file(cgmodel, medoid_xyz, output_dir, output_format)
    medoid_positions = medoid_xyz * unit.nanometer
    
    # Get indices of frames in each cluster:
    cluster_indices = {}
    cluster_sizes = []   
    for k in range(n_clusters):
        cluster_indices[k] = np.argwhere(labels==k)[:,0]
        cluster_sizes.append(len(cluster_indices[k]))
    
    # Assign intra-cluster distances to medoids
    dist_to_medoids = {}
    for k in range(n_clusters):
        dist_to_medoids[k] = torsion_distances[cluster_indices[k],medoid_indices[k]]
    
    # Compute cluster rmsd of samples to medoid within each cluster
    cluster_rmsd = np.zeros(n_clusters)
    for k in range(n_clusters):
        for i in range(len(cluster_indices[k])):
            cluster_rmsd[k] += np.power(dist_to_medoids[k][i],2)
        cluster_rmsd[k] /= len(cluster_indices[k])
        cluster_rmsd[k] = np.sqrt(cluster_rmsd[k])  

    # Get silhouette scores
    silhouette_avg = silhouette_score(torsion_distances, kmedoids.labels_)
    silhouette_sample_values = silhouette_samples(torsion_distances, kmedoids.labels_)
    
    if plot_silhouette:
        # Plot silhouette analysis
        plotfile=f"{output_dir}/silhouette_kmedoids_ncluster_{n_clusters}.pdf"
            
        make_silhouette_plot(
            kmedoids, silhouette_sample_values, silhouette_avg,
            n_clusters, cluster_rmsd, cluster_sizes, plotfile
            )        
    
    return (medoid_positions, medoid_torsions, cluster_sizes, cluster_rmsd, silhouette_avg)
    

def cluster_torsions_DBSCAN(
    file_list, cgmodel, min_samples=5, eps=0.5,
    frame_start=0, frame_stride=1, frame_end=-1, output_format="pdb", output_dir="cluster_output",
    backbone_torsion_type="bb_bb_bb_bb",
    filter=True, filter_ratio=0.05, plot_silhouette=True, plot_distance_hist=True):
    """
    Given PDB or DCD trajectory files and coarse grained model as input, this function performs DBSCAN clustering on the poses in the trajectory, and returns a list of the coordinates for the medoid pose of each cluster.

    :param file_list: A list of PDB or DCD files to read and concatenate
    :type file_list: List( str )

    :param cgmodel: A CGModel() class object
    :type cgmodel: class
    
    :param min_samples: minimum of number of samples in neighborhood of a point to be considered a core point (includes point itself)
    :type min_samples: int
    
    :param eps: DBSCAN parameter neighborhood distance cutoff
    :type eps: float

    :param frame_start: First frame in trajectory file to use for clustering.
    :type frame_start: int

    :param frame_stride: Advance by this many frames when reading trajectories.
    :type frame_stride: int

    :param frame_end: Last frame in trajectory file to use for clustering.
    :type frame_end: int
    
    :param output_format: file format extension to write medoid coordinates to (default="pdb"), dcd also supported
    :type output_format: str
    
    :param output_dir: directory to write clustering medoid and plot files
    :type output_dir: str
    
    :param backbone_torsion_type: particle sequence of the backbone torsions (default="bb_bb_bb_bb") - for now only single sequence permitted
    :type backbone_torsion_type: str    
    
    :param filter: option to apply neighborhood radius filtering to remove low-density data (default=True)
    :type filter: boolean
    
    :param filter_ratio: fraction of data points which pass through the neighborhood radius filter (default=0.05)
    :type filter_ratio: float
    
    :param plot_silhouette: option to create silhouette plot of clustering results (default=True)
    :type plot_silhouette: boolean
    
    :param plot_torsion_hist: option to plot a histogram of torsion euclidean distances (post-filtering)
    :type plot_torsion_hist: boolean

    :returns:
       - medoid_positions ( np.array( float * unit.angstrom ( n_clusters x num_particles x 3 ) ) ) - A 3D numpy array of poses corresponding to the medoids of all trajectory clusters.
       - medoid torsions ( np.array ( float * unit.degrees ( n_clusters x n_torsion ) - A 2D numpy array of the backbone torsion angles for each cluster medoid
       - cluster_sizes ( List ( int ) ) - A list of number of members in each cluster 
       - cluster_rmsd( np.array ( float ) ) - A 1D numpy array of rmsd (in cluster distance space) of samples to cluster centers
       - n_noise ( int ) - number of points classified as noise
       - silhouette_avg - ( float ) - average silhouette score across all clusters 
    """    
    
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    
    torsion_val_array, traj_all = get_torsion_matrix(file_list, cgmodel, frame_start, frame_stride, frame_end, backbone_torsion_type)
    
    # We need to precompute the euclidean distance matrix, accounting for periodic boundaries 
    
    total = 0
    angle_range = np.full(torsion_val_array.shape[1],360)
    powers = np.full(torsion_val_array.shape[1],2)

    torsion_distances = np.zeros((torsion_val_array.shape[0], torsion_val_array.shape[0]))  
    
    for i in range(torsion_val_array.shape[0]):
        for j in range(torsion_val_array.shape[0]): 
            delta = np.abs(torsion_val_array[i,:]-torsion_val_array[j,:])
            delta = np.where(delta > 0.5*angle_range, delta-angle_range, delta)
            torsion_distances[i,j] = np.sqrt(np.power(delta,powers).sum())
     
    if filter:
        # Filter distances:
        torsion_distances, dense_indices, filter_ratio_actual = \
            filter_distances(torsion_distances, filter_ratio=filter_ratio)
        
        traj_all = traj_all[dense_indices]
     
    if plot_distance_hist:
        distances_row = np.reshape(torsion_distances, (torsion_distances.shape[0]*torsion_distances.shape[1],1))
        
        # Remove the diagonal 0 elements:
        distances_row = distances_row[distances_row != 0]
        
        figure = plt.figure()
        n_out, bin_edges_out, patch = plt.hist(
            distances_row, bins=1000,density=True)
        plt.xlabel('rmsd')
        plt.ylabel('probability density')
        plt.savefig(f'{output_dir}/torsion_distances_hist.pdf')
        plt.close()    
    
    # Cluster with sklearn DBSCAN
    dbscan = DBSCAN(min_samples=min_samples,eps=eps,metric='precomputed').fit(torsion_distances)
    # The produces a cluster labels from 0 to n_clusters-1, and assigns -1 to noise points
    
    # Get labels
    labels = dbscan.labels_
    
    # Number of clusters:
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    
    # Number of noise points:
    n_noise = list(labels).count(-1)    
    
    # Get indices of frames in each cluster:
    cluster_indices = {}
    cluster_sizes = []   
    for k in range(n_clusters):
        cluster_indices[k] = np.argwhere(labels==k)[:,0]
        cluster_sizes.append(len(cluster_indices[k]))
        
    # Get indices of frames classified as noise:
    noise_indices = np.argwhere(labels==-1)[:,0]
        
    # Find the structure closest to each center (medoid):
    # OPTICS/DBSCAN does not have a built-in function to transform to cluster-distance space,
    # as the centroids of the clusters are not physically meaningful in general. However, as
    # RMSD between structures is our only clustering feature, the cluster centers (regions of
    # high density) will likely be representative structures of each cluster.

    # Following the protocol outlined in MDTraj example:
    # http://mdtraj.org/1.9.3/examples/centroids.html
    
    # Create distance matrices within each cluster:
    torsion_distances_k = {}
    for k in range(n_clusters):
        torsion_distances_k[k] = np.zeros((cluster_sizes[k],cluster_sizes[k]))
        for i in range(cluster_sizes[k]):
            for j in range(cluster_sizes[k]):
                torsion_distances_k[k][i,j] = torsion_distances[cluster_indices[k][i],cluster_indices[k][j]]
    
    # Compute medoid based on similarity scores:
    medoid_index = []
    for k in range(n_clusters):
        medoid_index.append(
            np.exp(-torsion_distances_k[k] / torsion_distances_k[k].std()).sum(axis=1).argmax()
        )
        
    medoid_xyz = np.zeros([n_clusters,traj_all.n_atoms,3])
    for k in range(n_clusters):
        medoid_xyz[k,:,:] = traj_all[medoid_index[k]].xyz[0]
    
    # Write medoids to file
    write_medoids_to_file(cgmodel, medoid_xyz, output_dir, output_format)
    medoid_positions = medoid_xyz * unit.nanometer
    
    # Get medoid torsions:
    medoid_torsions = np.zeros([n_clusters, torsion_val_array.shape[1]])
    for k in range(n_clusters):
        medoid_torsions[k,:] = torsion_val_array[medoid_index[k],:]
    
    # Compute intra-cluster rmsd of samples to medoid based on structure rmsd  
    cluster_rmsd = np.zeros(n_clusters)
    
    for k in range(n_clusters):
        cluster_rmsd[k] = np.sqrt(((torsion_distances_k[k][medoid_index[k]]**2).sum())/len(cluster_indices[k]))
    
    # Get silhouette scores
    try:
        silhouette_sample_values = silhouette_samples(torsion_distances, labels)
        silhouette_avg = np.mean(silhouette_sample_values[labels!=-1])
    
        if plot_silhouette:
            # Plot silhouette analysis
            plotfile = f"{output_dir}/silhouette_dbscan_min_sample_{min_samples}_eps_{eps}.pdf"
                
            make_silhouette_plot(
                dbscan, silhouette_sample_values, silhouette_avg,
                n_clusters, cluster_rmsd, cluster_sizes, plotfile
                )
    except ValueError:
        print("There are either no clusters, or no noise points identified. Try adjusting DBSCAN min_samples, eps parameters.")
        silhouette_avg = None

    return medoid_positions, medoid_torsions, cluster_sizes, cluster_rmsd, n_noise, silhouette_avg  
    
    
def get_torsion_matrix(file_list, cgmodel, frame_start, frame_stride, frame_end, backbone_torsion_type):
    """Internal function for reading trajectory files and computing torsions"""

    # Load files as {replica number: replica trajectory}
    rep_traj = {}
    for i in range(len(file_list)):
        if file_list[0][-3:] == 'dcd':
            rep_traj[i] = md.load(file_list[i],top=md.Topology.from_openmm(cgmodel.topology))
        else:
            rep_traj[i] = md.load(file_list[i])

    # Combine all trajectories, selecting specified frames
    if frame_end == -1:
        frame_end = rep_traj[0].n_frames

    if frame_start == -1:
        frame_start == frame_end

    traj_all = rep_traj[0][frame_start:frame_end:frame_stride]

    for i in range(len(file_list)-1):
        traj_all = traj_all.join(rep_traj[i+1][frame_start:frame_end:frame_stride])
    
    # Get torsion list:
    torsion_list = CGModel.get_torsion_list(cgmodel)
   
    # Assign torsion types:
    torsion_types, torsion_array, torsion_sub_arrays, n_i, i_torsion_type, torsion_dict, inv_torsion_dict = \
        assign_torsion_types(cgmodel, torsion_list)   
        
    # Compute specified torsion angles over all frames:        
    for i in range(i_torsion_type):
        if inv_torsion_dict[str(i+1)] == backbone_torsion_type:
            # Compute all torsion values in trajectory
            # This returns an [nframes x n_torsions] array
            torsion_val_array = md.compute_dihedrals(
                traj_all,torsion_sub_arrays[str(i+1)])
            
            # Convert to degrees:  
            torsion_val_array = (180/np.pi)*torsion_val_array  
        
    return torsion_val_array, traj_all   
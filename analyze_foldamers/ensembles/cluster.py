import os
import simtk.unit as unit
import mdtraj as md
import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import OPTICS, DBSCAN, cluster_optics_dbscan
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from cg_openmm.cg_model.cgmodel import CGModel
from cg_openmm.utilities.iotools import write_pdbfile_without_topology
    
    
def get_cluster_medoid_positions_DBSCAN(
    file_list, cgmodel, min_samples=5, eps=0.5, frame_start=0, frame_stride=1, frame_end=-1, output_format="pdb", output_dir="cluster_output",
    plot_silhouette=True):
    """
        Given PDB or DCD trajectory files and coarse grained model as input, this function performs DBSCAN clustering on the poses in the PDB file, and returns a list of the coordinates for the medoid pose of each cluster.

        :param file_list: A list of PDB or DCD files to read and concatenate
        :type file_list: List( str )

        :param cgmodel: A CGModel() class object
        :type cgmodel: class

        :param frame_start: First frame in pdb trajectory file to use for clustering.
        :type frame_start: int

        :param frame_stride: Advance by this many frames when reading pdb trajectories.
        :type frame_stride: int

        :param frame_end: Last frame in pdb trajectory file to use for clustering.
        :type frame_end: int
        
        :param output_format: file format extension to write medoid coordinates to (default="pdb"), dcd also supported
        :type output_format: str

        :returns:
        - medoid_positions ( np.array( float * unit.angstrom ( n_clusters x num_particles x 3 ) ) ) - A 3D numpy array of poses corresponding to the medoids of all trajectory clusters.
        - cluster_sizes ( List ( int ) ) - A list of number of members in each cluster 
        - cluster_rmsd( np.array ( float ) ) - A 1D numpy array of rmsd (in cluster distance space) of samples to cluster centers
        """    
    
    distances, traj_all = get_rmsd_matrix(file_list, cgmodel, frame_start, frame_stride, frame_end)
    
    # Cluster with sklearn DBSCAN
    dbscan = DBSCAN(min_samples=min_samples,eps=eps,metric='precomputed').fit(distances)
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
    distances_k = {}
    for k in range(n_clusters):
        distances_k[k] = np.zeros((cluster_sizes[k],cluster_sizes[k]))
        for i in range(cluster_sizes[k]):
            for j in range(cluster_sizes[k]):
                distances_k[k][i,j] = distances[cluster_indices[k][i],cluster_indices[k][j]]
    
    # Compute medoid based on similarity scores:
    medoid_index = []
    for k in range(n_clusters):
        medoid_index.append(
            np.exp(-distances_k[k] / distances_k[k].std()).sum(axis=1).argmax()
        )
        
    medoid_xyz = np.zeros([n_clusters,traj_all.n_atoms,3])
    for k in range(n_clusters):
        medoid_xyz[k,:,:] = traj_all[medoid_index[k]].xyz[0]
    
    # Write medoids to file
    write_medoids_to_file(cgmodel, medoid_xyz, output_dir, output_format)
    medoid_positions = medoid_xyz * unit.nanometer
    
    # Compute intra-cluster rmsd of samples to medoid based on structure rmsd
    cluster_rmsd = np.zeros(n_clusters)
    for k in range(n_clusters):
        cluster_rmsd[k] = np.sqrt(((distances_k[k][medoid_index[k]]**2).sum())/len(cluster_indices[k]))    
    
    # Get silhouette scores
    try:
        silhouette_sample_values = silhouette_samples(distances, labels)
        silhouette_avg = np.mean(silhouette_sample_values[labels!=-1])
    
        if plot_silhouette:
            # Plot silhouette analysis
            plotfile = f"{output_dir}/silhouette_dbscan_min_sample_{min_samples}_eps_{eps}.pdf"
                
            make_silhouette_plot(
                dbscan, silhouette_sample_values, silhouette_avg,
                n_clusters, cluster_rmsd, cluster_sizes, plotfile
                )
    except ValueError:
        print("No clusters identified - try adjusting DBSCAN min_samples, eps parameters")
        silhouette_avg = None

    return medoid_positions, cluster_sizes, cluster_rmsd, n_noise, silhouette_avg  
    
    
def get_cluster_medoid_positions_OPTICS(
    file_list, cgmodel, min_samples=5, xi=0.05, frame_start=0, frame_stride=1, frame_end=-1, output_format="pdb", output_dir="cluster_output",
    plot_silhouette=True):
    """
        Given PDB or DCD trajectory files and coarse grained model as input, this function performs OPTICS clustering on the poses in the PDB file, and returns a list of the coordinates for the medoid pose of each cluster.

        :param file_list: A list of PDB or DCD files to read and concatenate
        :type file_list: List( str )

        :param cgmodel: A CGModel() class object
        :type cgmodel: class

        :param frame_start: First frame in pdb trajectory file to use for clustering.
        :type frame_start: int

        :param frame_stride: Advance by this many frames when reading pdb trajectories.
        :type frame_stride: int

        :param frame_end: Last frame in pdb trajectory file to use for clustering.
        :type frame_end: int
        
        :param output_format: file format extension to write medoid coordinates to (default="pdb"), dcd also supported
        :type output_format: str

        :returns:
        - medoid_positions ( np.array( float * unit.angstrom ( n_clusters x num_particles x 3 ) ) ) - A 3D numpy array of poses corresponding to the medoids of all trajectory clusters.
        - cluster_sizes ( List ( int ) ) - A list of number of members in each cluster 
        - cluster_rmsd( np.array ( float ) ) - A 1D numpy array of rmsd (in cluster distance space) of samples to cluster centers
        """    
    
    distances, traj_all = get_rmsd_matrix(file_list, cgmodel, frame_start, frame_stride, frame_end)
    
    # Cluster with sklearn OPTICS
    optic = OPTICS(min_samples=min_samples,xi=xi,cluster_method='xi',metric='precomputed').fit(distances)
    # The produces a cluster labels from 0 to n_clusters-1, and assigns -1 to noise points
    
    # Get labels
    labels = optic.labels_
    
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
    distances_k = {}
    for k in range(n_clusters):
        distances_k[k] = np.zeros((cluster_sizes[k],cluster_sizes[k]))
        for i in range(cluster_sizes[k]):
            for j in range(cluster_sizes[k]):
                distances_k[k][i,j] = distances[cluster_indices[k][i],cluster_indices[k][j]]
    
    # Compute medoid based on similarity scores:
    medoid_index = []
    for k in range(n_clusters):
        medoid_index.append(
            np.exp(-distances_k[k] / distances_k[k].std()).sum(axis=1).argmax()
        )
        
    medoid_xyz = np.zeros([n_clusters,traj_all.n_atoms,3])
    for k in range(n_clusters):
        medoid_xyz[k,:,:] = traj_all[medoid_index[k]].xyz[0]
        
    # Write medoids to file
    write_medoids_to_file(cgmodel, medoid_xyz, output_dir, output_format)
    medoid_positions = medoid_xyz * unit.nanometer
    
    # Compute intra-cluster rmsd of samples to medoid based on structure rmsd
    cluster_rmsd = np.zeros(n_clusters)
    for k in range(n_clusters):
        cluster_rmsd[k] = np.sqrt(((distances_k[k][medoid_index[k]]**2).sum())/len(cluster_indices[k]))
    
    # Get silhouette scores
    silhouette_sample_values = silhouette_samples(distances, labels)
    silhouette_avg = np.mean(silhouette_sample_values[labels!=-1])
    
    # Get silhouette scores
    try:
        silhouette_sample_values = silhouette_samples(distances, labels)
        silhouette_avg = np.mean(silhouette_sample_values[labels!=-1])
    
        if plot_silhouette:
            # Plot silhouette analysis
            plotfile = f"{output_dir}/silhouette_optics_min_sample_{min_samples}_xi_{xi}.pdf"
                
            make_silhouette_plot(
                optic, silhouette_sample_values, silhouette_avg,
                n_clusters, cluster_rmsd, cluster_sizes, plotfile
                )
    except ValueError:
        print("No clusters identified - try adjusting OPTICS min_samples, xi parameters")
        silhouette_avg = None
        
    return medoid_positions, cluster_sizes, cluster_rmsd, n_noise, silhouette_avg
    
    
def get_rmsd_matrix(file_list, cgmodel, frame_start, frame_stride, frame_end):
    """Internal function for reading trajectory files and computing rmsd"""
    
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

    # Align structures with first frame as reference:
    for i in range(1,traj_all.n_frames):
        md.Trajectory.superpose(traj_all[i],traj_all[0])
        # This rewrites to traj_all

    # Compute pairwise rmsd:
    distances = np.empty((traj_all.n_frames, traj_all.n_frames))
    for i in range(traj_all.n_frames):
        distances[i] = md.rmsd(traj_all, traj_all, i)
        
    return distances, traj_all
    

def write_medoids_to_file(cgmodel, medoid_positions, output_dir, output_format):
    """Internal function for writing medoid coordinates to file"""

    # Write medoids to file
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    
    n_clusters = medoid_positions.shape[0]
    
    for k in range(n_clusters):
        positions = medoid_positions[k] * unit.nanometer
        file_name = str(f"{output_dir}/medoid_{k}.{output_format}")
        if output_format=="dcd":
            dcdtraj = md.Trajectory(
                xyz=positions.value_in_unit(unit.nanometer),
                topology=md.Topology.from_openmm(cgmodel.topology),
            )
            md.Trajectory.save_dcd(dcdtraj,file_name)
        else:
            cgmodel.positions = positions
            write_pdbfile_without_topology(cgmodel, file_name)
            
    return
    

def get_cluster_medoid_positions_KMeans(
    file_list,
    cgmodel,
    n_clusters=2,
    frame_start=0,
    frame_stride=1,
    frame_end=-1,
    output_format="pdb",
    output_dir="cluster_output",
    plot_silhouette=True,
    plot_cluster_distance=True,
    ):
    """
        Given PDB or DCD trajectory files and coarse grained model as input, this function performs K-means clustering on the poses in the PDB file, and returns a list of the coordinates for the medoid pose of each cluster.

        :param file_list: A list of PDB or DCD files to read and concatenate
        :type file_list: List( str )

        :param cgmodel: A CGModel() class object
        :type cgmodel: class

        :param n_clusters: The number of clusters for KMeans algorithm.
        :type n_clusters: int

        :param frame_start: First frame in pdb trajectory file to use for clustering.
        :type frame_start: int

        :param frame_stride: Advance by this many frames when reading pdb trajectories.
        :type frame_stride: int

        :param frame_end: Last frame in pdb trajectory file to use for clustering.
        :type frame_end: int
        
        :param output_format: file format extension to write medoid coordinates to (default="pdb"), dcd also supported
        :type output_format: str
        
        :param output_dir: path to which cluster medoid structures and silhouette plots will be saved
        :type output_dir: str
        
        :param plot_silhouette: option to create silhouette plot of clustering results (default=True)
        :type plot_silhouette: bool
        
        :param plot_cluster_distance: option to create plots of distances (in cluster space) of each point to cluster centers (default=True)
        :type plot_cluster_distance: bool
        
        :returns:
        - medoid_positions ( np.array( float * unit.angstrom ( n_clusters x num_particles x 3 ) ) ) - A 3D numpy array of poses corresponding to the medoids of all trajectory clusters.
        - cluster_sizes ( List ( int ) ) - A list of number of members in each cluster 
        - cluster_rmsd( np.array ( float ) ) - A 1D numpy array of rmsd (in cluster distance space) of samples to cluster centers
        """

    distances, traj_all = get_rmsd_matrix(file_list, cgmodel, frame_start, frame_stride, frame_end)

    # Cluster with sklearn KMeans
    kmeans = KMeans(n_clusters=n_clusters).fit(distances)

    # Get indices of frames in each cluster:
    cluster_indices = {}
    for k in range(n_clusters):
        cluster_indices[k] = np.argwhere(kmeans.labels_==k)[:,0]

    # Find the structure closest to each center (medoid):
    dist_to_centroids = KMeans.transform(kmeans,distances) # (n_samples x n_clusters)
    closest_indices = np.argmin(dist_to_centroids,axis=0)
    closest_xyz = np.zeros([n_clusters,traj_all.n_atoms,3])
    for k in range(n_clusters):
        closest_xyz[k,:,:] = traj_all[closest_indices[k]].xyz[0]
        
    # Compute cluster rmsd of samples to cluster center within each cluster
    cluster_rmsd = np.zeros(n_clusters)
    for k in range(n_clusters):
        for i in range(len(cluster_indices[k])):
            cluster_rmsd[k] += dist_to_centroids[cluster_indices[k][i]][k]**2
        cluster_rmsd[k] /= len(cluster_indices[k])
        cluster_rmsd[k] = np.sqrt(cluster_rmsd[k])
            
    # Write medoids to file
    write_medoids_to_file(cgmodel, closest_xyz, output_dir, output_format)
    medoid_positions = closest_xyz * unit.nanometer
    
    # Get cluster sizes
    cluster_sizes = [] # Number of samples in each cluster
    
    for k in range(n_clusters):
        cluster_sizes.append(len(cluster_indices[k]))
        
    # Get silhouette scores
    silhouette_avg = silhouette_score(distances, kmeans.labels_)
    silhouette_sample_values = silhouette_samples(distances, kmeans.labels_)
    
    if plot_silhouette:
        # Plot silhouette analysis
        plotfile=f"{output_dir}/silhouette_kmeans_ncluster_{n_clusters}.pdf"
            
        make_silhouette_plot(
            kmeans, silhouette_sample_values, silhouette_avg,
            n_clusters, cluster_rmsd, cluster_sizes, plotfile
            )
    
    if plot_cluster_distance:
        # Plot cluster-distance results
        plotfile=f"{output_dir}/kmeans_distances_ncluster_{n_clusters}.pdf"
        
        make_cluster_distance_plots(n_clusters, kmeans, dist_to_centroids, plotfile)
        
    return (medoid_positions, cluster_sizes, cluster_rmsd, silhouette_avg)
    

def make_cluster_distance_plots(n_clusters,cluster_fit,dist_to_centroids,plotfile):
    """Internal function for creating cluster distance plots"""

    n_sub = 0
    for k in range(n_clusters):
        n_sub += k
    
    nrow = int(np.ceil(np.sqrt(n_sub)))
    ncol = int(np.ceil(n_sub/nrow))
    
    fig2, axs2 = plt.subplots(nrow,ncol,figsize=(8,10))
    
    subplot_id = 1
    for k in range(n_clusters):
        for j in range(k+1,n_clusters):
            row = int(np.ceil(subplot_id/ncol))-1
            col = int((subplot_id-1)%ncol)
            
            # axs subplot object is only subscriptable in dimensions it has multiple entries in
            if nrow > 1 and ncol > 1:
                # Color data by cluster assignment:
                for c in range(n_clusters):
                    color = cm.nipy_spectral(float(c)/n_clusters)
                    axs2[row,col].plot(
                        dist_to_centroids[np.argwhere(cluster_fit.labels_==c),k],
                        dist_to_centroids[np.argwhere(cluster_fit.labels_==c),j],
                        '.',
                        markeredgecolor=color,
                        markerfacecolor=color,
                        label=f'k={c}',
                    )
                axs2[row,col].set_xlabel(f'Distance to centroid {k}',
                    {'fontsize': 8})
                axs2[row,col].set_ylabel(f'Distance to centroid {j}',
                    {'fontsize': 8})
                    
                if row == 0 and col == 0:
                    # Add legend:
                    axs2[row,col].legend()
                    
            if ncol > 1 and nrow == 1:
                for c in range(n_clusters):
                    color = cm.nipy_spectral(float(c)/n_clusters)
                    axs2[col].plot(
                        dist_to_centroids[np.argwhere(cluster_fit.labels_==c),k],
                        dist_to_centroids[np.argwhere(cluster_fit.labels_==c),j],
                        '.',
                        markeredgecolor=color,
                        markerfacecolor=color,
                        label=f'k={c}',
                    )
                axs2[col].set_xlabel(f'Distance to centroid {k}',
                    {'fontsize': 8})
                axs2[col].set_ylabel(f'Distance to centroid {j}',
                    {'fontsize': 8})
                    
                if row == 0 and col == 0:
                    # Add legend:
                    axs2[col].legend()

            if ncol == 1 and nrow == 1:
                for c in range(n_clusters):
                    color = cm.nipy_spectral(float(c)/n_clusters)
                    axs2.plot(
                        dist_to_centroids[np.argwhere(cluster_fit.labels_==c),k],
                        dist_to_centroids[np.argwhere(cluster_fit.labels_==c),j],
                        '.',
                        markeredgecolor=color,
                        markerfacecolor=color,
                        label=f'k={c}',
                    )
                axs2.set_xlabel(f'Distance to centroid {k}',
                    {'fontsize': 8})
                axs2.set_ylabel(f'Distance to centroid {j}',
                    {'fontsize': 8})
                    
                # Add legend:
                axs2.legend()
                    
                 
            subplot_id += 1
    
    plt.tight_layout()
    plt.savefig(plotfile)
            
    
def make_silhouette_plot(
    cluster_fit, silhouette_sample_values, silhouette_avg,
    n_clusters, cluster_rmsd, cluster_sizes, plotfile
    ):
    
    """Internal function for creating silhouette plot"""

    fig1, ax1 = plt.subplots(1,1,figsize=(8,6))
    
    y_lower = 10
    
    for k in range(n_clusters):
        kth_cluster_sil_values = silhouette_sample_values[cluster_fit.labels_==k]
        
        kth_cluster_sil_values.sort()
        
        y_upper = y_lower + cluster_sizes[k]
        
        color = cm.nipy_spectral(float(k)/n_clusters)
        ax1.fill_betweenx(
            np.arange(y_lower, y_upper),0,
            kth_cluster_sil_values,
            facecolor=color, edgecolor=color, alpha=0.7)
            
        ax1.text(-0.05, y_lower + 0.5 * cluster_sizes[k], str(k))
        
        y_lower = y_upper + 10
        
    ax1.set_xlabel("Silhouette coefficient")
    ax1.set_ylabel("Cluster label")
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
    ax1.set_yticks([]) # Clear y ticks/labels
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
    
    xlim = ax1.get_xlim()
    ylim = ax1.get_ylim()
    
    for k in range(n_clusters):
        ax1.text(xlim[0]+0.75*(xlim[1]-xlim[0]), (0.9-(k/25))*(ylim[1]-ylim[0]),
            f"Cluster {k} RMSD: {cluster_rmsd[k]:.3f}",
            {'fontsize': 8},
            horizontalalignment='left')
    
    plt.tight_layout()
    plt.savefig(plotfile) 
    
    return
    
    
def concatenate_trajectories(pdb_file_list, combined_pdb_file="combined.pdb"):
    """
        Given a list of PDB files, this function reads their coordinates, concatenates them, and saves the combined coordinates to a new file (useful for clustering with MSMBuilder).

        :param pdb_file_list: A list of PDB files to read and concatenate
        :type pdb_file_list: List( str )

        :param combined_pdb_file: The name of file/path in which to save a combined version of the PDB files, default = "combined.pdb"
        :type combined_pdb_file: str

        :returns:
          - combined_pdb_file ( str ) - The name/path for a file within which the concatenated coordinates will be written.

        """
    traj_list = []
    for pdb_file in pdb_file_list:
        traj = md.load(pdb_file)
        traj_list.append(traj)
    return combined_pdb_file    
    

def align_structures(reference_traj, target_traj):
    """
        Given a reference trajectory, this function performs a structural alignment for a second input trajectory, with respect to the reference.

        :param reference_traj: The trajectory to use as a reference for alignment.
        :type reference_traj: `MDTraj() trajectory <http://mdtraj.org/1.6.2/api/generated/mdtraj.Trajectory.html>`_

        :param target_traj: The trajectory to align with the reference.
        :type target_traj: `MDTraj() trajectory <http://mdtraj.org/1.6.2/api/generated/mdtraj.Trajectory.html>`_

        :returns:
          - aligned_target_traj ( `MDTraj() trajectory <http://mdtraj.org/1.6.2/api/generated/mdtraj.Trajectory.html>`_ ) - The coordinates for the aligned trajectory.

        """

    aligned_target_traj = target_traj.superpose(reference_traj)

    return aligned_target_traj

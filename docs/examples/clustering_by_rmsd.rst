Clustering by RMSD
=========================

Clustering by root-mean-square deviation of particle coordinates amongst frames in a
trajectory or set of trajectories is useful for identifying frequently visited
molecular conformations. RMSD clustering can identify the native structure in a 
system which undergoes folding and unfolding transitions, for example.

In this example, topology from a CGModel and particle positions from a set of
replica exchange MD trajectories is input into the DBSCAN clustering algorithm.

.. code-block:: python

    from analyze_foldamers.ensembles.cluster import *
    from cg_openmm.cg_model.cgmodel import CGModel
    import pickle

    # Load in cgmodel created with cg_openmm:
    cgmodel = pickle.load(open("stored_cgmodel.pkl","rb"))

    # Specify path to output directory:
    output_dir = "output_directory"
    
    # Create list of trajectory files for clustering analysis:
    number_replicas = 12
    pdb_file_list = []
    for i in range(number_replicas):
        pdb_file_list.append(f"{output_dir}/replica_{i+1}.pdb")

    # Set clustering parameters:
    frame_start = 20000      # Set the starting frame (i.e., after the equilibration period)
    frame_stride = 100       # skip this many frames to reduce memory requirement
    min_samples = 100        # Minimum number of neighbors for DBSCAN core points
    eps = 0.10               # DBSCAN neighborhood distance cutoff
                             # Distance units need to match those in the trajectory files
    filter = True            # Use pre-clustering density filtering to remove low density data
    filter_ratio = 0.25      # Remove 75% of lowest density data
    core_points_only = False # Use both core and non-core points to determine 'medoid'

    # Run DBSCAN clustering:
    (medoid_positions, cluster_sizes, cluster_rmsd, n_noise,
    silhouette_avg, labels, original_indices) = get_cluster_medoid_positions_DBSCAN(
        file_list=pdb_file_list,
        cgmodel=cgmodel,
        min_samples=min_samples,
        eps=eps,
        frame_start=frame_start
        frame_stride=frame_stride,
        filter=filter,
        filter_ratio=filter_ratio,
        output_dir=output_dir,
        core_points_only=core_points_only,        
    )

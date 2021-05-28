.. analyze_foldamers documentation master file, created by
   sphinx-quickstart on Thu Mar 15 13:55:56 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

analyze_foldamers
=========================================================

``analyze_foldamers`` is a Python package for structural analysis of coarse-grained
oligomer molecular simulations.

 - Identify native structures using RMSD or torsion-based clustering of trajectories
 - Generate histograms of all bonded parameters (bonds, angles, torsions)
 - Generate 2D histograms of any combination of bonded observables
 - Determine helical parameters for folded oligomer structures
 - Tools for automated rendering of molecular snapshots with VMD
 
.. toctree::
   :maxdepth: 2
   :caption: Getting started

   installation
   examples
   
.. toctree::
   :maxdepth: 2
   :caption: Bonded distributions
   
   bonded_distributions
   
.. toctree::
   :maxdepth: 2
   :caption: Clustering
   
   clustering
   
.. toctree::
   :maxdepth: 2
   :caption: Helical fitting
   
   helical
   
.. toctree::
   :maxdepth: 2
   :caption: Visualization
   
   visualization

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

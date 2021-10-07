Installation
===============

Currently, only the development version of ``analyze_foldamers`` is available.

Download analyze_foldamers from GitHub using:

``git clone https://github.com/shirtsgroup/analyze_foldamers.git``

In the base directory of analyze_foldamers, install using:

``python setup.py install``

.. note::
   A working version of VMD is required to use the molecular snapshot tools in analyze_foldamers

Conda environment
-----------------

``analyze_foldamers`` is currently tested and maintained on Python 3.6, 3.7, and 3.8.

With CGModel support
~~~~~~~~~~~~~~~~~~~~
To use CGModel objects created with ``cg_openmm``, additional dependencies need to be specified
when creating an anaconda environment for analyze_foldamers:

.. code-block:: bash
    
   conda create -n cg_openmm_env python=3.X mdtraj mpi4py numpy openmm openmmtools physical_validation
   pymbar scikit-learn scikit-learn-extra scipy
   
In addition, an installed version of cg_openmm is required to use CGModel objects.

Download cg_openmm from GitHub using:

``git clone https://github.com/shirtsgroup/cg_openmm.git``

In the base directory of cg_openmm, install using:

``python setup.py install``
   
Without CGModel support
~~~~~~~~~~~~~~~~~~~~~~~
.. note::
   We are currently working on eliminating the dependency of analyze_foldamers on cg_openmm.
   This option is not yet supported.

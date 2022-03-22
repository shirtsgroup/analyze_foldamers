import os
import timeit

import matplotlib.pyplot as pyplot
import numpy as np
from analyze_foldamers.parameters.helical_fitting import *
from cg_openmm.cg_model.cgmodel import CGModel
from openmm import unit
from openmm.app.pdbfile import PDBFile

positions = PDBFile(
    str(str(os.getcwd().split("examples")[0]) + "ensembles/12_1_1_0/helix.pdb")
).getPositions()

cgmodel = CGModel(positions=positions)
pitch, radius, monomers_per_turn, residual = get_helical_parameters(cgmodel)
print(pitch, radius, monomers_per_turn, residual)

cgmodel = orient_along_z_axis(cgmodel)

show_helical_fit(cgmodel)

p2 = calculate_p2(cgmodel)
print(p2)

exit()

import os
import timeit

import matplotlib.pyplot as pyplot
import numpy as np
from analyze_foldamers.parameters.helical_fitting import (
    get_helical_data, get_helical_parameters)
from cg_openmm.cg_model.cgmodel import CGModel
from openmm import unit
from openmm.app.pdbfile import PDBFile

positions = PDBFile(
    str(str(os.getcwd().split("examples")[0]) + "ensembles/12_1_1_0/helix.pdb")
).getPositions()

cgmodel = CGModel(positions=positions)
pitch, radius, monomers_per_turn, residual = get_helical_parameters(cgmodel)
print(pitch, radius, monomers_per_turn, residual)

exit()

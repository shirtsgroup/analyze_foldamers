import numpy as np
import pandas as pd

"""
Analysis functions/objects for visualizing energy distributions
"""


file_read_types = \
{ 
    "csv" : pd.read_csv,
    # Expandable to other file types
}

def read_energies(energy_file, type = "csv", **kwargs):
    energies = file_read_types[type](energy_file, **kwargs)
    return(energies)
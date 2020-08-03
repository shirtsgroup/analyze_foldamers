"""
analyze_foldamers
Tools for structural analysis of coarse-grained foldamer simulations
"""

# Add imports here
from .cluster import *

# Handle versioneer
from ._version import get_versions
versions = get_versions()
__version__ = versions['version']
__git_revision__ = versions['full-revisionid']
del get_versions, versions

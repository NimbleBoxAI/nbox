"""
This is the hidden closet with all messy and cool things.

This submodule has files for managing the adapters for different frameworks. The important
repositories with the logic start with ``on_``, why?

#. ``on_ml``: contains the code for exporting and sharing the weights across different\
    ml libraries like pytorch, tensorflow, haiku, etc.
#. ``on_operators``: contains the code for exporting and importing operators from frameworks\
    like Airflow, Prefect, etc.
#. ``on_functions``: This library contains the code for static parsing of python and\
    creating the NBX-JobFlow
"""

# if you like what you see and want to work on more things like this, reach out research@nimblebox.ai

from .on_operators import *

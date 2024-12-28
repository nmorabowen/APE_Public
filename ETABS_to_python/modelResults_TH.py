import numpy as np
import pandas as pd
from rapidfuzz import process
import matplotlib.pyplot as plt

from plotApeConfig import blueAPE, grayConcrete, set_default_plot_params, color_palette
set_default_plot_params()

from .modelResults_utilities import modelResults_utilities

class modelResults_TH:
    def __init__(self, model):
        self.model=model
        
        # Get the ETABS table
        story_forces_dict=self.model.tables.get_table_data("Story Forces")
        self.df=story_forces_dict['dataFrame']

# Class aimed to working and drawing story shear and forces from ETABS models

import numpy as np
import pandas as pd
from rapidfuzz import process
import matplotlib.pyplot as plt

from plotApeConfig import blueAPE, grayConcrete, set_default_plot_params, color_palette
set_default_plot_params()

from .modelResults_utilities import modelResults_utilities

class modelResults_storyForces(modelResults_utilities):
    def __init__(self, model, units=None):
        self.model=model
        
        # Internal variable to control color
        self.color_palette=color_palette[0]
        
        # Get the ETABS table
        story_forces_dict=self.model.tables.get_table_data("Story Forces")
        self.df=story_forces_dict['dataFrame']
        
        # Create stories elevation array
        self.stories_dict=self.model.tables.get_stories_table()
        self.stories_array=self.stories_dict['elevations_array']
        

        # Error checking for units
        if units is not None:
            if not isinstance(units, list):
                raise TypeError("The 'units' parameter must be a list.")
            if len(units) != 2:
                raise ValueError("The 'units' list must contain exactly two elements: [force_unit, displacement_unit].")
        
        if units is None:
            selected_units={"force":1,
                            "length":1}
            self.units=selected_units
        else:
            selected_units={"force":units[0],
                            "length":units[1]}
            self.units=selected_units
        
        self._modify_df()
        self._get_elevations_array()
    
    
    def _modify_df(self):
        # This method transform some parts of the dataFrame to make it workable
        df=self.df
        story_mapping=self.stories_dict['story_mapping']
        
        # Add elevations coordinates to the dataFrame
        df=self.elevation_mapping(df=df, story_mapping=story_mapping)
  
        # Convert numeric columns explicitly
        for col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col])  # Attempt to convert each column to numeric
            except ValueError:
                pass  # Skip columns that cannot be converted
            
        self.df=df
        
    def get_load_cases(self):
        load_cases_names=self.get_ouputCase_names(self.df)
        return load_cases_names
    
    def _get_elevations_array(self):
        # Create a stacked version of the elevations to plot in a step wise fashion
        elevations_array=np.repeat(self.stories_array,2)[1:-1]/self.units['length']
        self.elevations_array=np.flip(elevations_array)
    
    def get_forces(self, outputCase):
        # This method outputs the story base shear and forces
        dataFrame=self.df
        
        # We will output the most likely match for the load case
        # Define a list of valid OutputCase values
        valid_output_cases = dataFrame['OutputCase'].unique()
        
        # Perform fuzzy matching to find the closest valid OutputCase
        best_match, score, _ = process.extractOne(outputCase, valid_output_cases)
        
        # If the match score is below a certain threshold, raise an error
        if score < 80:  # Threshold can be adjusted based on requirements
            print(f"No close match found for OutputCase '{outputCase}'. Closest match '{best_match}' with score {score}.")
            raise ValueError(f"No close match found for OutputCase '{outputCase}'. Closest match '{best_match}' with score {score}.")

        # Use the best match as the output case
        outputCase = best_match
        
        # Determine the column to use based on 'X' or 'Y' in the outputCase
        column = 'VX' if 'X' in outputCase.upper() else 'VY'
        
        filterCase=(dataFrame['OutputCase'] == outputCase)
        filterTop=(dataFrame['Location'] == 'Top')
        filterBot=(dataFrame['Location'] == 'Bottom')
        
        # Determine the StepType parameters
        # For the response spectrum cases the step type values are max/min, for the static cases it is none, we will filter them using the max value
        if 'Max' in dataFrame.loc[dataFrame['OutputCase'] == outputCase,'StepType'].values:
            stepType='Max'
            filterCase &= (dataFrame['StepType'] == stepType)
                 
        # Filter data and compute results
        baseShear_Top = dataFrame.loc[filterCase & filterTop, column].to_numpy()/self.units['force']
        baseShear_Bot = dataFrame.loc[filterCase & filterBot, column].to_numpy()/self.units['force']
        storyForce = np.diff(baseShear_Top, prepend=0)
        
        baseShear=np.empty(baseShear_Top.size+baseShear_Bot.size, dtype=baseShear_Top.dtype)
        baseShear[0::2]=baseShear_Bot
        baseShear[1::2]=baseShear_Top
        
        return baseShear, storyForce
    
    def _plotBaseShear(self, outputCase, ax=None, linewidth=1, linstyle='-', color=blueAPE, appendString=None):
        # Method to plot the base shear for a single plot
        
        stories=self.elevations_array
        
        if appendString is not None:
            plotLabel=outputCase+'-'+appendString
        else:
            plotLabel=outputCase
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(5,5))
          
        baseShear, _ = self.get_forces(outputCase)
        ax.plot(baseShear, stories, color=color, linewidth=linewidth, linestyle=linstyle, label=plotLabel)
        ax.plot(-baseShear, stories, color=color, linewidth=linewidth, linestyle=linstyle)
        
        ax.set_yticks(self.stories_array)
        
        return ax
    
    def plotBaseShear(self, outputCaseList, ax=None, show=True):
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(5,5))
            
        for i,outputCase in enumerate(outputCaseList):
            self._plotBaseShear(outputCase=outputCase, ax=ax, color=color_palette[i])
            
        # Add labels, title, grid, and legend
        ax.set_title("Story Shears", fontsize=12)
        ax.set_xlabel("Story Shear", fontsize=10)
        ax.set_ylabel("Story Level", fontsize=10)
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.7)

        # Show the plot
        plt.tight_layout()
        
        if show is True:
            plt.show()

    def _plotStoryForces(self, outputCase, ax=None, linewidth=0.5, linstyle='--', color=blueAPE, marker='.', appendString=None):
        # Method to plot the story forces for a single plot
        
        stories=np.flip(self.stories_array)
        
        if appendString is not None:
            plotLabel=outputCase+'-'+appendString
        else:
            plotLabel=outputCase
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(5,5))
          
        _, storyForces = self.get_forces(outputCase)
        # Add the base value as zero
        storyForces=np.insert(storyForces, len(storyForces), 0)
        
        # Create the vertical bar plot for positive and negative forces
        ax.barh(stories, storyForces, color=color, height=1, align='center', alpha=0.5)
        ax.barh(stories, -storyForces, color=color, height=1, align='center', alpha=0.5)
        
        ax.plot(storyForces, stories, color=color, linewidth=linewidth, linestyle=linstyle, label=plotLabel, marker=marker)
        ax.plot(-storyForces, stories, color=color, linewidth=linewidth, linestyle=linstyle, marker=marker)
        
        ax.set_yticks(self.stories_array)
        
        return ax

    def plotStoryForces(self, outputCaseList, ax=None, show=True):
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(5,5))
            
        for i,outputCase in enumerate(outputCaseList):
            self._plotStoryForces(outputCase=outputCase, ax=ax, color=color_palette[i])
            
        # Add labels, title, grid, and legend
        ax.set_title("Story Forces", fontsize=12)
        ax.set_xlabel("Story Force", fontsize=10)
        ax.set_ylabel("Story Level", fontsize=10)
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.7)

        # Show the plot
        plt.tight_layout()
        plt.show()
        
        if show is True:
            plt.show()
        
        
def plotMultipleModelsBaseShear(dictionary, figsize=(8,5)):
    """
    Plot base shear results from multiple models on the same graph.
    
    Parameters:
    -----------
    dictionary : dict
        Dictionary containing model information in the format:
        {
            'id': {
                'name': str,         # Name identifier for the model
                'model': model,      # Model object
                'outputCase': list   # List of output cases to plot
            }
        }
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot each model's results
    for i, (model_id, model_info) in enumerate(dictionary.items()):
        model_name = model_info['name']
        model = model_info['model']
        cases = model_info['outputCase']
            
        for j, case in enumerate(cases):
            model._plotBaseShear(outputCase=case, ax=ax, color=color_palette[i+j], appendString=model_name)
    
    # Customize plot appearance
    ax.set_title("Base Shear Comparison", fontsize=12)
    ax.set_xlabel("Base Shear", fontsize=10)
    ax.set_ylabel("Story Level", fontsize=10)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, linestyle="--", alpha=0.7)
    
    # Adjust layout to prevent legend cutoff
    plt.tight_layout()
    
    return fig, ax

def plotMultipleModelsStoryForces(dictionary, figsize=(8,5)):
    """
    Plot base shear results from multiple models on the same graph.
    
    Parameters:
    -----------
    dictionary : dict
        Dictionary containing model information in the format:
        {
            'id': {
                'name': str,         # Name identifier for the model
                'model': model,      # Model object
                'outputCase': list   # List of output cases to plot
            }
        }
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot each model's results
    for i, (model_id, model_info) in enumerate(dictionary.items()):
        model_name = model_info['name']
        model = model_info['model']
        cases = model_info['outputCase']
            
        for j, case in enumerate(cases):
            model._plotStoryForces(outputCase=case, ax=ax, color=color_palette[i+j], appendString=model_name)
    
    # Customize plot appearance
    ax.set_title("Story Forces Comparison", fontsize=12)
    ax.set_xlabel("Story Force", fontsize=10)
    ax.set_ylabel("Story Level", fontsize=10)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, linestyle="--", alpha=0.7)
    
    # Adjust layout to prevent legend cutoff
    plt.tight_layout()
    
    return fig, ax
import numpy as np
import pandas as pd
from rapidfuzz import process
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, AutoMinorLocator, MultipleLocator

from plotApeConfig import blueAPE, grayConcrete, set_default_plot_params, color_palette
set_default_plot_params()

from .modelResults_utilities import modelResults_utilities

class modelResults_wallForces(modelResults_utilities):
    def __init__(self, model, design_parameters, units=None):
        self.model=model
        self.overstrength=design_parameters['overstrength']
        self.dynamic_amplification=design_parameters['dynamic_amplification']
        
        # Calculate the shear amplification factor
        self.shear_amplification()
        
        # Get the ETABS table
        wall_forces_dict=self.model.tables.get_table_data("Design Forces - Piers")
        self.df=wall_forces_dict['dataFrame']
        
        # Create stories elevation array
        self.stories_dict=self.model.tables.get_stories_table()
        self.stories_array=self.stories_dict['elevations_array']
        
        # Call required methods
        self._modify_df()
        self._story_elevations()
        
        # Get info
        self.pier_labels=self.get_pier_labels()
        self.load_combinations, self.load_combinations_static, self.load_combinations_dynamic = self.get_load_combinations()

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
        
    def shear_amplification(self):
        self.shear_amplification=min(3,self.overstrength*self.dynamic_amplification)
    
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
            
        # Create a filter parameter to parse dynamic and static loads combinations
        
        def state_conditional(item):
            if "E" in item:
                state='Dynamic'
            else:
                state='Static'
            return state

        # Asignamos el estado a cada carga
        df['State']=df['Combo'].apply(state_conditional)
        
        self.df=df
        
    def get_pier_labels(self):
        return self.df['Pier'].unique()
    
    def get_load_combinations(self):
        load_combinations=self.df['Combo'].unique()
        load_combinations_static=self.df[self.df['State']=='Static']['Combo'].unique()
        load_combinations_dynamic=self.df[self.df['State']=='Dynamic']['Combo'].unique()
        
        return load_combinations, load_combinations_static, load_combinations_dynamic
    
    def _story_elevations(self):
        # This method transform some parts of the dataFrame to make it workable
        df=self.df
        
        story_mapping=self.stories_dict['story_mapping']
        
        # Add elevations coordinates to the dataFrame
        df=self.elevation_mapping_TopBottom(df=df, story_mapping=story_mapping)
  
        # Convert numeric columns explicitly
        for col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col])  # Attempt to convert each column to numeric
            except ValueError:
                pass  # Skip columns that cannot be converted
            
        self.df=df
        
    def _wall_filter(self, wall_label, combo_label):
        filter_df=self.df[(self.df['Pier']==wall_label) & (self.df['Combo']==combo_label)]
        
        return filter_df
    
    def plot_wall_forces(self, pier_label, figsize=(10, 6), show_grid=True):
        
        load_combinations=self.load_combinations
            
        # Create subplots
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        
        plt.suptitle(f'Design Forces - Wall: {pier_label}')
        
        # Define the plotting parameters
        plot_params = [
            {'data': 'P', 'xlabel': 'Axial Force', 'symbol': 'P_u'},
            {'data': 'M3', 'xlabel': 'Bending Moment', 'symbol': 'M_u'},
            {'data': 'V2', 'xlabel': 'Shear Force', 'symbol': 'V_u'}
        ]
        
        # Plot each force type
        for color_idx, combo in enumerate(load_combinations):
            wall_data = self._wall_filter(pier_label, combo)
            
            for ax_idx, params in enumerate(plot_params):
                ax = axes[ax_idx]
                force_type = params['data']
                
                # Plot the data
                ax.plot(wall_data[force_type], wall_data['Elevation'], 
                    marker='.', linewidth=0.5, label=combo, color=color_palette[color_idx])
                
                # Configure axis
                ax.grid(show_grid)
                ax.set_xlabel(f"{params['xlabel']} - ${params['symbol']}")
                ax.set_ylabel(f'Height')
                
                # Set y-axis limits and ticks
                ax.set_ylim(wall_data['Elevation'].min(), wall_data['Elevation'].max())
                ax.set_yticks(wall_data['Elevation'])
        
        # Adjust layout to prevent overlap
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        return fig, axes
    
    def plot_wall_force_envelopes(self, pier_label, figsize=(10, 6)):

        load_combinations = self.load_combinations
        shear_amplification_factor = self.shear_amplification

        # Filter data for the specified wall and load combinations
        wall_data = self.df[(self.df['Pier'] == pier_label) & self.df['Combo'].isin(load_combinations)]

        # Calculate envelopes for each force type
        envelopes = {
            'P': wall_data.groupby('Elevation')['P'].agg(['min', 'max']),
            'M3': wall_data.groupby('Elevation')['M3'].agg(['min', 'max']),
            'V2': wall_data.groupby('Elevation')['V2'].agg(['min', 'max'])
        }

        # Create subplots
        fig, axes = plt.subplots(1, 3, figsize=figsize)

        # Set title
        plt.suptitle(f'Design Force Envelopes - Wall: {pier_label}')

        # Define the plotting parameters
        plot_params = [
            {'data': 'P', 'xlabel': 'Axial Force', 'symbol': 'P_u'},
            {'data': 'M3', 'xlabel': 'Bending Moment', 'symbol': 'M_u'},
            {'data': 'V2', 'xlabel': 'Shear Force', 'symbol': 'V_u'}
        ]

        # Plot each force type
        for ax_idx, params in enumerate(plot_params):
            ax = axes[ax_idx]
            force_type = params['data']
            envelope = envelopes[force_type]

            # Plot min/max envelopes
            ax.plot(envelope['min'], envelope.index, marker='.', linewidth=1.5,
                    color='black', label='Envelope')
            ax.plot(envelope['max'], envelope.index, marker='.', linewidth=1.5,
                    color='black')

            # Add amplified shear if it's the shear force plot
            if force_type == 'V2' and shear_amplification_factor != 1.0:
                ax.plot(envelope['min'] * shear_amplification_factor, envelope.index,
                        marker='.', linewidth=1.5, color='blue', linestyle='--',
                        label=f'Amplified ({shear_amplification_factor}×)')
                ax.plot(envelope['max'] * shear_amplification_factor, envelope.index,
                        marker='.', linewidth=1.5, color='blue', linestyle='--')

            # Configure axis labels
            ax.set_xlabel(f"{params['xlabel']} - ${params['symbol']}$")
            ax.set_ylabel('Elevation [m]')

            # Set y-axis range
            ax.set_ylim(envelope.index.min(), envelope.index.max())

            # Only use minor ticks
            ax.tick_params(axis='both', which='major', length=0)  # hide major ticks
            ax.minorticks_on()
            ax.tick_params(axis='both', which='minor', length=4, width=1, color='black')

            # Optional: customize density of minor ticks
            ax.yaxis.set_minor_locator(MultipleLocator(1))     # minor ticks every 1 unit in elevation
            ax.xaxis.set_minor_locator(AutoMinorLocator(5))    # 4 minor ticks between x-ticks

            # Add legend if needed
            if force_type == 'V2' and shear_amplification_factor != 1.0:
                ax.legend()

        # Adjust layout
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        return fig, axes
    
    
    

        
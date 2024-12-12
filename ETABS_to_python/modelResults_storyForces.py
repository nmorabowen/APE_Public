
# Class aimed to working and drawing story shear and forces from ETABS models


class modelResults_storyForces:
    def __init__(self, df, elevations, units=None):
        self.df=df
        self.color_index=0
        self.elevations=elevations

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
        self._get_elevations()
    
    
    def _modify_df(self):
        # This method transform some parts of the dataFrame to make it workable
        df=self.df
        # Removemos los caracteres de texto N+ y N- del data frame
        df['Story'] = df['Story'].str.replace('N([+-])', r'\1', regex=True).astype(float)

        # Convert numeric columns explicitly
        for col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col])  # Attempt to convert each column to numeric
            except ValueError:
                pass  # Skip columns that cannot be converted
            
        self.df=df
    
    def _get_elevations(self):
        elevations_array=np.repeat(self.elevations,2)[1:-1]/self.units['length']
        self.elevations_array=np.flip(elevations_array)
    
    def get_load_cases(self):
        load_cases=df['OutputCase'].unique()
        return load_cases
    
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
        if 'Max' in df.loc[df['OutputCase'] == outputCase,'StepType'].values:
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
    
    def _plotBaseShear(self, outputCase, ax=None, linewidth=1, linstyle='-', color=blueAPE):
        # Method to plot the base shear for a single plot
        
        stories=self.elevations_array
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(5,5))
          
        baseShear, _ = self.get_forces(outputCase)
        ax.plot(baseShear, stories, color=color, linewidth=linewidth, linestyle=linstyle, label=outputCase)
        ax.plot(-baseShear, stories, color=color, linewidth=linewidth, linestyle=linstyle)
        
        ax.set_yticks(self.elevations)
        
        return ax
    
    def _plotStoryForces(self, outputCase, ax=None, linewidth=1, linstyle='-', color=blueAPE):
        # Method to plot the story forces for a single plot
        
        stories=self.elevations_array[0::2]
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(5,5))
          
        _, storyForces = self.get_forces(outputCase)
        ax.plot(storyForces, stories, color=color, linewidth=linewidth, linestyle=linstyle, label=outputCase)
        ax.plot(-storyForces, stories, color=color, linewidth=linewidth, linestyle=linstyle)
        
        ax.set_yticks(self.elevations)
        
        return ax
    
    def plotBaseShear(self, outputCaseList, ax=None):
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(5,5))
            
        for i,outputCase in enumerate(outputCaseList):
            self._plotBaseShear(outputCase=outputCase, ax=ax, color=color_palette[i])
            
        # Add labels, title, grid, and legend
        ax.set_title("Base Shear vs. Story", fontsize=12)
        ax.set_xlabel("Base Shear [tf]", fontsize=10)
        ax.set_ylabel("Story Level", fontsize=10)
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.7)

        # Show the plot
        plt.tight_layout()
        plt.show()
        
        plt.show()
        
    def plotStoryForces(self, outputCaseList, ax=None):
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(5,5))
            
        for i,outputCase in enumerate(outputCaseList):
            self._plotStoryForces(outputCase=outputCase, ax=ax, color=color_palette[i])
            
        # Add labels, title, grid, and legend
        ax.set_title("Base Shear vs. Story", fontsize=12)
        ax.set_xlabel("Base Shear [tf]", fontsize=10)
        ax.set_ylabel("Story Level", fontsize=10)
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.7)

        # Show the plot
        plt.tight_layout()
        plt.show()
        
        plt.show()
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import comtypes.client
import ctypes

class ETABS_APE:
    def __init__(self, filePath=None, processID=None) -> None:
        
        # Call method to connect to the model instance and return the object
        self.filePath=filePath
        self.processID=processID
        self.SapModel=self.connect_to_etabs()
        
        # Connect with helper classes
        self.tables=ETABS_APE_TABLES(self.SapModel)
        self.nodes=ETABS_APE_NODES(self.SapModel)


    def connect_to_etabs(self):
        """Connect to a running ETABS instance, either active or specified by processID."""
        
        try:
            if self.processID is not None:
                # Attach to a specific ETABS instance using process ID
                etabs = comtypes.client.GetObjectProcess("CSI.ETABS.API.ETABSObject", self.processID)
            else:
                # Attach to the active ETABS instance
                etabs = comtypes.client.GetActiveObject("CSI.ETABS.API.ETABSObject")
            
            SapModel = etabs.SapModel
            
            # If filePath is provided, open the model
            if self.filePath:
                SapModel.File.OpenFile(self.filePath)
                
            return SapModel
        
        except Exception as e:
            print(f"Error connecting to ETABS: {e}")
            return None
        
    
    def _create_units_dictionaries(self):
        """
        Helper method to create dictionaries that map unit codes to their respective names.

        This method generates three dictionaries that map:
        - Force unit codes to their names (e.g., `1` -> `lb`, `4` -> `kN`).
        - Length unit codes to their names (e.g., `4` -> `mm`, `6` -> `m`).
        - Temperature unit codes to their names (e.g., `1` -> `F`, `2` -> `C`).

        Returns:
            tuple[dict[int, str], dict[int, str], dict[int, str]]: 
                A tuple containing three dictionaries:
                    - `eForce_dict` (dict[int, str]): Maps force unit codes to unit names.
                    - `eLength_dict` (dict[int, str]): Maps length unit codes to unit names.
                    - `eTemperature_dict` (dict[int, str]): Maps temperature unit codes to unit names.
        """
        
        eForce_dict = {
            0: "NotApplicable",
            1: "lb",
            2: "kip",
            3: "N",
            4: "kN",
            5: "kgf",
            6: "tonf"
        }
        
        eLength_dict = {
            0: "NotApplicable",
            1: "inch",
            2: "ft",
            3: "micron",
            4: "mm",
            5: "cm",
            6: "m"
        }
        
        eTemperature_dict = {
            0: "NotApplicable",
            1: "F",
            2: "C"
        }
        
        return eForce_dict, eLength_dict, eTemperature_dict

    def display_units(self):
        eForce_dict, eLength_dict, eTemperature_dict = self._create_units_dictionaries()
        
        # Iterate over each dictionary and display key-value pairs
        print("Force Units:")
        for key, value in eForce_dict.items():
            print(f"{key}: {value}")
        
        print("\nLength Units:")
        for key, value in eLength_dict.items():
            print(f"{key}: {value}")
        
        print("\nTemperature Units:")
        for key, value in eTemperature_dict.items():
            print(f"{key}: {value}")

    def set_model_units(self, eForce=3, eLength=4, eTemp=2, verbose=True):
        """Method to change the base model units, the default values correspond to N-mm (similar to the baseUnits library)
        To get the ETABS units codes run the _create_unit_dictionary() method

        Args:
            eForce (int, optional): _description_. Defaults to 3.
            eLength (int, optional): _description_. Defaults to 4.
            eTemp (int, optional): _description_. Defaults to 2.

        Raises:
            Exception: _description_
        """
        # Set units
        check = self.SapModel.SetPresentUnits_2(eForce, eLength, eTemp)
        
        if verbose is True:
            self.get_model_units()
        
        if check != 0:
            raise Exception("Failed to set units")
        
    def get_model_units(self):
        """
        Method to output the current model units, both in terms of API codes as names 
        """
        
        presentUnits=self.SapModel.GetPresentUnits()
        force_code, length_code, temperature_code, _ =presentUnitsNames=self.SapModel.GetPresentUnits_2()
        eForce_dict, eLength_dict, eTemperature_dict = self._create_units_dictionaries()
        
        print(f'The model base units code is: {presentUnits}')
        
        print(f'The model units are: {presentUnitsNames}')
        print(f'The force unit is: {eForce_dict[force_code]}')
        print(f'The length unit is: {eLength_dict[length_code]}')
        print(f'The temperature unit is: {eTemperature_dict[temperature_code]}')
        
    
        
class ETABS_APE_NODES:
    
    def __init__(self, SapModel) -> None:
        self.SapModel=SapModel
    
    def get_nodes_list(self):
        ret = self.SapModel.PointObj.GetNameList()
    
        return ret[1]

    def get_node_coord(self, node):

        ret = self.SapModel.PointObj.GetCoordCartesian(node)

        return ret

class ETABS_APE_TABLES:
    
    """Helper Class to manage ETABS database table interactions."""
    
    def __init__(self, SapModel) -> None:
        self.SapModel=SapModel
    
    def get_table_data(self, table_key):
        """Generic function to get any table data from ETABS"""
        
        ret = self.SapModel.DatabaseTables.GetTableForDisplayArray(
            table_key, # ETABS Table name
            [],
            "",
            0,
            [],
            0,
            []
        )
        
        # Check if we got any data
        if not ret[4]:  # If TableData is empty
            print(f"No data found in table: {table_key}")
            return pd.DataFrame()  # Return empty DataFrame
        
        # Convert to DataFrame
        header = np.array(ret[2])
        data = np.array(ret[4])
        
        # Check if we have any data before reshaping
        if data.size == 0:
            print(f"No data found in table: {table_key}")
            return pd.DataFrame(columns=header)  # Return empty DataFrame with correct columns
            
        data_reshaped = data.reshape(-1, len(header))
        
        results_dict={'ret':ret,
                      'dataFrame':pd.DataFrame(data=data_reshaped, columns=header)}
        
        return results_dict
    
    def get_stories_table(self):
        """
        Retrieve the stories table from the ETABS model.

        Returns:
            dict: A dictionary containing:
                - 'ret': Raw results returned by the ETABS API.
                - 'elevations_array': A NumPy array of story elevations.
                - 'stories_df': A Pandas DataFrame with story information.
        """
        
        # Initialize variables to store story data
        
        BaseElevation=0
        NumberStories=0
        StoryNames=[]
        StoryElevations=[]
        StoryHeights=[]
        IsMasterStory = []
        SimilarToStory = []
        SpliceAbove = []
        SpliceHeight = []
        color = []

        # Get the base name and elevation
        base_dict=self.get_table_data("Tower and Base Story Definitions")
        base_df=base_dict['dataFrame']
        base_elevation=base_df['BSElev'].iloc[0]
        base_name=base_df['BSName'].iloc[0]

        # Retrieve story information
        ret_stories = self.SapModel.Story.GetStories_2(BaseElevation, NumberStories, StoryNames, StoryElevations, StoryHeights, IsMasterStory, SimilarToStory, SpliceAbove, SpliceHeight, color)
        
        elevations_array=np.array(ret_stories[3])
        elevations_array=np.insert(elevations_array, 0,base_elevation)
        
        storyNames=ret_stories[2]
        storyNames=np.insert(storyNames, 0,base_name)
        
        # Assumming that the base name is N+/-Number (This need to be generalized)
        story_mapping=dict(zip(storyNames, elevations_array))
        
        results_dict={'ret': ret_stories,
                      'elevations_array':elevations_array,
                      'base_elevation': BaseElevation,
                      'story_names':storyNames,
                      'story_mapping':story_mapping}
            
        return results_dict
    
    def list_available_tables(self):
        """List all available tables in ETABS"""
        tables = self.SapModel.DatabaseTables.GetAvailableTables()
        print("\nAvailable tables:")
        for i, table in enumerate(tables[1], 1):
            print(f"{i}. {table}")
            
    def get_joint_nodes_coordinates(self):
        
        """Get the Joint table filtered only by joint nodes (ignore shell nodes)

        Returns:
            df_filtered: Filtered node fataFrame
        """
        results=self.get_table_data("Objects and Elements - Joints")
        df=results['dataFrame']
        df_filtered=df[(df['ObjType']=='Joint')]
        
        return df_filtered
        
    
class drift:
    
    def __init__(self, model) -> None:
        self.model=model
        
        # Create the drift dataFrame
        self.drift_results=self.model.tables.get_table_data("Joint Drifts")
        self.drift_df=self.drift_results['dataFrame']
        # Create stories elevation array
        self.stories_dict=self.model.tables.get_stories_table()
        self.stories_array=self.stories_dict['elevations_array']
        
        # Add elevations to drift dataFrame
        self.elevation_mapping()
         
    def elevation_mapping(self):
        """Method to map the elevations in the dataFrame according to the story name, need to have the dataframe created and also the stories dictionaries
        """
        story_mapping=self.stories_dict['story_mapping']
        self.drift_df['Elevation']=self.drift_df['Story'].map(story_mapping)
    
    def get_ouputCase_names(self):
        outputCase=self.drift_df['OutputCase'].unique()
        
        print("\nAvailable OutputCase:")
        for i, table in enumerate(outputCase):
            print(f"{i}. {table}")
            
    
    def plot_drift_unique_name(self, uniqueName:str, outputCase:str, ax=None):
        
        # Drift Table
        drift_df=self.drift_df
        
        # Data frame filters
        outputCase_Filter=outputCase
        uniqueName_filter=uniqueName
        stepType_filter='Max'
        stepNumber_filter='None'
        
        filter_df=drift_df[
            (drift_df['OutputCase']==outputCase_Filter) &
            (drift_df['Label']==uniqueName_filter) &
            (drift_df['StepType']==stepType_filter) &
            (drift_df['StepNumber']==stepNumber_filter)
        ]

        if ax is None:
            fig, ax = plt.subplots(figsize=(6,6))
        
        
        # Plot the drift
        ax.plot(filter_df['DriftX'], filter_df['Elevation'], marker='o', label=f"DriftX - {uniqueName}")
        
        # Add labels, title, and legend
        ax.set_xlabel("Elevation (m)")
        ax.set_ylabel("DriftX")

        ax.legend()

        
        plt.show()
        
        return filter_df


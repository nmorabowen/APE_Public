class modelResults_utilities:
    """
    A mixin class to support the model results operations for the ModelResults classes.

    This class contains utility methods for handling model result operations, such as 
    mapping elevation values from story names and retrieving available output cases 
    in a results DataFrame.
    """
    
    def elevation_mapping(self, df, story_mapping):
        """
        Method to map the elevations in the DataFrame according to the story name.

        This method maps the story names in the 'Story' column of the provided DataFrame
        to the corresponding elevations based on the provided `story_mapping` dictionary, 
        and adds a new column 'Elevation' to the DataFrame.

        Parameters:
        df (pandas.DataFrame): The DataFrame containing the 'Story' column to be mapped.
        story_mapping (dict): A dictionary where the keys are story names and the values are corresponding elevations.

        Returns:
        pandas.DataFrame: The input DataFrame with an additional 'Elevation' column containing the mapped elevation values.
        
        Example:
        # Sample usage:
        df = pd.DataFrame({'Story': ['Story1', 'Story2', 'Story3']})
        story_mapping = {'Story1': 10, 'Story2': 20, 'Story3': 30}
        df = elevation_mapping(df, story_mapping)
        print(df)
        """
        df['Elevation'] = df['Story'].map(story_mapping)
        return df
    
    def get_ouputCase_names(self, df):
        """
        Method to print the unique output case names from the provided DataFrame.

        This method retrieves and prints all unique output case names from the 'OutputCase' column of the DataFrame.

        Parameters:
        df (pandas.DataFrame): The DataFrame containing the 'OutputCase' column.

        Returns:
        None: This method prints the available output case names and does not return any value.

        Example:
        # Sample usage:
        df = pd.DataFrame({'OutputCase': ['Case1', 'Case2', 'Case3', 'Case1']})
        get_ouputCase_names(df)
        """
        outputCase = df['OutputCase'].unique()
        
        print("\nAvailable OutputCase:")
        for i, table in enumerate(outputCase):
            print(f"{i}. {table}")
            
        return outputCase

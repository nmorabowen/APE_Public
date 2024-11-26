import pandas as pd
import os
from rapidfuzz import process  # For fuzzy matching

class AISCDatabase:
    def __init__(self):
        """
        Initialize the AISCDatabase object by always loading the database 
        from the predefined pickle file.
        """
        self.file_path = os.path.join(os.path.dirname(__file__), "AISC_Database.pkl")  # Hardcoded path to the pickle file
        self.df = self._load_database()
        
    def _load_database(self):
        """
        Load the database from the predefined pickle file.
        """
        try:
            return pd.read_pickle(self.file_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"The database file '{self.file_path}' does not exist.")
        except Exception as e:
            raise RuntimeError(f"An error occurred while loading the database: {e}")
        
    def get_properties(self, section_name, properties=None, verbose=False):
        """
        Retrieve properties for a given section name.
        
        Parameters:
        - section_name (str): The name of the section to look up.
        - properties (list, optional): A list of properties to filter for. 
          If None, return all geometric properties by default.
        
        Returns:
        - pd.Series: A pandas Series containing the filtered properties.
        """
        # Default geometric properties to filter
        geometric_columns = ["AISC_Manual_Label","bf", "tf", "h", "tw","W"]
        properties = properties or geometric_columns

        # Ensure case-insensitive matching
        section_name = section_name.upper()
        self.df["AISC_Manual_Label"] = self.df["AISC_Manual_Label"].str.upper()

        # Try exact matching
        row = self.df[self.df["AISC_Manual_Label"] == section_name]

        # If no exact match, use fuzzy matching to find the closest match
        if row.empty:
            print(f"Section '{section_name}' not found. Attempting fuzzy matching...")
            closest_match = process.extractOne(section_name, self.df["AISC_Manual_Label"].unique())
            if closest_match:
                print(f"Closest match found: '{closest_match[0]}' with a similarity score of {closest_match[1]}")
                row = self.df[self.df["AISC_Manual_Label"] == closest_match[0]]
            else:
                raise ValueError(f"No similar sections found for '{section_name}'.")

        # Return only the selected properties
        result = row[properties].iloc[0]
        
        if verbose is True:
            print(f"Properties for section '{section_name}':\n{result}")
        
        return result

# Examples in the __main__ block
if __name__ == "__main__":
    # Initialize the database
    database = AISCDatabase()

    # Display the first few rows
    print("First few rows of the database:")
    print(database.df.head())

    # Example: Get properties for a specific section
    results=database.get_properties(section_name="w12x6", verbose=True)
    
    

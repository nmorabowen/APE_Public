import numpy as np
import glob
from collections import defaultdict
import os

class CDATA:
    """This is a mixin class to parse through the info contained in a CDATA section."""
    
    def _get_cdata_info(self):

        file_list=self._get_file_list(extension='cdata', verbose=False)
        
        return file_list
    
    def list_CDATA_files(self):
        """
        Lists all base names (keys) of the CDATA files from the file list.

        Returns:
            None
        """
        
        file_list=self._get_file_list(extension='cdata', verbose=False)
        for file in file_list.keys():
            print(f'{file}')
    
    def _extract_selection_set_ids_for_file(self, file_path, selection_set_ids=None):
        """
        Extracts selection set IDs and associated data (nodes and elements) from the given file using NumPy for optimization.

        Args:
            file_path (str): Path to the .cdata file.
            selection_set_ids (list, optional): List of selection set IDs to extract. If None, process all.

        Returns:
            list: A list of dictionaries containing selection set data.
        """
        
        if selection_set_ids is not None and not isinstance(selection_set_ids, list):
            raise ValueError("selection_set_ids must be a list of integers or None.")
        
        selection_sets = []  # Store extracted data

        try:
            with open(file_path, 'r') as file:
                lines = file.readlines()

                # Convert lines to a NumPy array for efficient slicing
                lines_array = np.array(lines, dtype=str)

                # Iterate through lines to find *SELECTION_SET
                for i, line in enumerate(lines_array):
                    if line.strip() == "*SELECTION_SET":
                        # Get SET_ID info
                        set_id = int(lines_array[i + 1].strip())

                        # Skip if this selection set ID is not in the provided list
                        if selection_set_ids is not None and set_id not in selection_set_ids:
                            continue

                        # Extract the SET_NAME, keeping only the name
                        raw_set_name = lines_array[i + 2].strip()
                        name_length = int(raw_set_name.split()[0])  # Extract the length of the name
                        set_name = raw_set_name[len(str(name_length)) + 1 : len(str(name_length)) + 1 + name_length]  # Extract the actual name
                        
                        number_of_nodes = int(lines_array[i + 3].strip())
                        number_of_elements = int(lines_array[i + 4].strip())

                        # Initialize selection set data
                        selection_set = {
                            "SET_ID": set_id,
                            "SET_NAME": set_name,
                        }

                        # Extract nodes if number_of_nodes > 0
                        if number_of_nodes > 0:
                            nodes_start_line = i + 5
                            nodes_end_line = nodes_start_line + (number_of_nodes + 9) // 10
                            node_lines = lines_array[nodes_start_line:nodes_end_line]
                            # Combine and convert to a NumPy array
                            selection_set["NODES"] = np.fromstring(" ".join(node_lines).strip(), sep=" ", dtype=int)

                        # Extract elements if number_of_elements > 0
                        if number_of_elements > 0:
                            elements_start_line = nodes_end_line
                            elements_end_line = elements_start_line + (number_of_elements + 9) // 10
                            element_lines = lines_array[elements_start_line:elements_end_line]
                            # Combine and convert to a NumPy array
                            selection_set["ELEMENTS"] = np.fromstring(" ".join(element_lines).strip(), sep=" ", dtype=int)

                        # Add the parsed selection set to the list
                        selection_sets.append(selection_set)

        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
            return []

        return selection_sets
    
    def extract_selection_set_ids(self, fileName, selection_set_ids=None):
        """
        NOTE: THE BASE CLASS SHOULD EVETUALLY POIT TO THE NAME OF THE FILE, NOT THE FILE ITSELF
        
        Aggregates nodes and elements while maintaining the structure of each selection set.

        Args:
            fileName (str): Name of the `.cdata` file to process.
            selection_set_ids (list, optional): List of selection set IDs to extract. If None, all sets are included.

        Returns:
            dict: A dictionary where each key is a selection set ID, and the value is another dictionary
                containing 'SET_NAME', 'NODES', and 'ELEMENTS'.
        """
        if selection_set_ids is not None and not isinstance(selection_set_ids, list):
            raise ValueError("selection_set_ids must be a list of integers or None.")

        aggregated_data = {}

        # Get the list of `.cdata` files
        file_mapping = self._get_cdata_info()
        
        if fileName not in file_mapping.keys():
            raise ValueError(
                f"File name '{fileName}' not found in the file mapping. The name should be one of the following: {list(file_mapping.keys())}"
            )

        # Get a list of all the file paths
        file_list = [item['file'] for sublist in file_mapping.values() for item in sublist]

        for file_path in file_list:
            # Extract selection sets for the current file
            selection_sets = self._extract_selection_set_ids_for_file(file_path, selection_set_ids=selection_set_ids)

            # Aggregate each selection set by its SET_ID
            for selection_set in selection_sets:
                set_id = selection_set["SET_ID"]

                # Skip if the set ID is not in the provided list
                if selection_set_ids is not None and set_id not in selection_set_ids:
                    continue

                if set_id not in aggregated_data:
                    # Initialize a new entry for this selection set
                    aggregated_data[set_id] = {
                        "SET_NAME": selection_set["SET_NAME"],
                        "NODES": set(selection_set.get("NODES", [])),
                        "ELEMENTS": set(selection_set.get("ELEMENTS", [])),
                    }
                else:
                    # Update existing entry by merging nodes and elements
                    aggregated_data[set_id]["NODES"].update(selection_set.get("NODES", []))
                    aggregated_data[set_id]["ELEMENTS"].update(selection_set.get("ELEMENTS", []))

        # Convert sets to sorted lists for final output
        for set_id in aggregated_data:
            aggregated_data[set_id]["NODES"] = sorted(aggregated_data[set_id]["NODES"])
            aggregated_data[set_id]["ELEMENTS"] = sorted(aggregated_data[set_id]["ELEMENTS"])

        return aggregated_data


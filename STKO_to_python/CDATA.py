import numpy as np
import glob
from collections import defaultdict
import os

class CDATA:
    """This is a mixin class to parse through the info contained in a CDATA section."""
    
    def _get_cdata_info(self):

        file_list=self._get_file_list(extension='cdata', verbose=True)
        
        return file_list
    
    def list_CDATA_files(self):
        """
        Lists all base names (keys) of the CDATA files from the file list.

        Returns:
            None
        """
        
        file_list=self.get_file_list(extension='cdata', verbose=False)
        for file in file_list.keys():
            print(f'{file}')
    
    def _extract_selection_set_ids_for_file(self, file_path):
        """
        Extracts selection set IDs from the given file.

        Args:
            file_path (str): Path to the .cdata file.

        Returns:
            list: A list of dictionaries containing selection set IDs and their respective data.
        """
        selection_sets = []  # Store extracted data

        try:
            with open(file_path, 'r') as file:
                lines = file.readlines()

                # Iterate through lines to find *SELECTION_SET
                for i, line in enumerate(lines):
                    if line.strip() == "*SELECTION_SET":
                        # Get SET_ID info
                        set_id = int(lines[i + 1].strip())
                        # Debugging information
                        print(f"Found *SELECTION_SET at line {i + 1}: {line.strip()}")

                        # Further debugging or processing logic can go here
                        try:
                            set_id = int(lines[i + 1].strip())
                            print(f"SET_ID at line {i + 2}: {set_id}")
                        except (IndexError, ValueError) as e:
                            print(f"Error reading SET_ID at line {i + 2}: {e}")

        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
            return []

        return selection_sets

    def extract_all_selection_sets(self, CDATA_name:str):
        """
        Extracts all selection sets from all .cdata files by reusing the existing method.

        Returns:
            list: A list of dictionaries containing selection set data for all files.
        """
        # Get the CDATA file mapping
        file_mapping = self._get_file_list(extension='cdata', verbose=False)
        file_list=file_mapping[CDATA_name]
        all_selection_sets = []  # Store all selection sets across all files

        for file_info in file_list:
            file_path = file_info['file']
            try:
                # Use the existing method to extract selection sets for a single file
                selection_sets = self._extract_selection_set_ids_for_file(file_path)
                
                # Add the file's selection sets to the overall list
                all_selection_sets.extend(selection_sets)

            except Exception as e:
                print(f"Error processing file {file_path}: {e}")

        # Print a summary of extracted data
        for s_set in all_selection_sets:
            print(f"FILE: {s_set['FILE']}, SET_ID: {s_set['SET_ID']}, SET_NAME: {s_set['SET_NAME']}, "
                f"NNODES: {s_set['NNODES']}, NELEMENTS: {s_set['NELEMENTS']}")
            if s_set["NODES"]:
                print(f"NODES: {s_set['NODES'][:10]}... ({len(s_set['NODES'])} total)")
            if s_set["ELEMENTS"]:
                print(f"ELEMENTS: {s_set['ELEMENTS'][:10]}... ({len(s_set['ELEMENTS'])} total)")

        return all_selection_sets
    
    def _get_info_selectionSet(self, selection_set_id, CDATA_name:str):
        # Get the CDATA file mapping
        file_mapping=self._get_file_list(extension='cdata', verbose=False)
        file_list=file_mapping[CDATA_name]
        
        for file in file_list:
            file_path=file['file']
            
            selection_sets = []  # Store extracted data
            
            try:
                with open(file_path, 'r') as file:
                    lines = file.readlines()
                    
                    # Iterate through lines to find *SELECTION_SET
                    for i, line in enumerate(lines):
                        if line.strip() == "*SELECTION_SET":
                            # Get the ID on the next line
                            set_id = int(lines[i + 1].strip())  # Extract selection set ID
                            if set_id==selection_set_id:
                                set_name = lines[i + 2].strip()    # Extract name
                                n_nodes = int(lines[i + 3].strip())  # Extract NNODES
                                n_elements = int(lines[i + 4].strip())  # Extract NELEMENTS
                                
                                # Create a dictionary for this selection set
                                selection_set = {
                                    "SET_ID": set_id,
                                    "SET_NAME": set_name,
                                    "NNODES": n_nodes,
                                    "NELEMENTS": n_elements,
                                    "NODES": [],
                                    "ELEMENTS": []
                                }
                                
                                # Extract nodes and elements if they exist
                                if n_nodes > 0:
                                    selection_set["NODES"] = [
                                        int(node)
                                        for node in " ".join(lines[i + 5: i + 5 + (n_nodes + 9) // 10]).split()
                                    ]
                                
                                if n_elements > 0:
                                    selection_set["ELEMENTS"] = [
                                        int(element)
                                        for element in " ".join(lines[i + 5 + (n_nodes + 9) // 10:]).split()
                                    ]
                                
                                # Append this set's data
                                selection_sets.append(selection_set)
                
                # Print extracted data for verification
                for s_set in selection_sets:
                    print(f"SET_ID: {s_set['SET_ID']}, SET_NAME: {s_set['SET_NAME']}, "
                        f"NNODES: {s_set['NNODES']}, NELEMENTS: {s_set['NELEMENTS']}")
                    if s_set["NODES"]:
                        print(f"NODES: {s_set['NODES'][:10]}... ({len(s_set['NODES'])} total)")
                    if s_set["ELEMENTS"]:
                        print(f"ELEMENTS: {s_set['ELEMENTS'][:10]}... ({len(s_set['ELEMENTS'])} total)")

                return selection_sets

            except Exception as e:
                print(f"Error processing file: {e}")
                return []
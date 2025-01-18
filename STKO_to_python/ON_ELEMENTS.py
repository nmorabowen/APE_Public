import h5py
import numpy as np
import matplotlib.pyplot as plt
import re
import pandas as pd


class ON_ELEMENTS:
    """This is a mixin class to work with MPCO_VirtualDataset"""
    
    def _get_all_element_index(self, element_type=None, print_memory=False):
        """
        Fetch information for all elements of a given type in the partition files.
        If no element type is provided, fetch information for all element types.

        Args:
            element_type (str or None): The type of elements to fetch (e.g., 'ElasticBeam3d'). 
                                        If None, fetches all element types.
            print_memory (bool, optional): If True, prints memory usage for the structured array
                                        and DataFrame. Defaults to False.

        Returns:
            dict: A dictionary containing:
                - 'array': Structured NumPy array with element data.
                - 'dataframe': Pandas DataFrame with element data.
        """
        model_stages = self.get_model_stages()

        # Get all element types if none specified
        if element_type is None:
            element_types = self._get_all_types(model_stage=model_stages[0])
        else:
            element_types = [element_type]

        # List to store all elements' information
        elements_info = []

        # Loop through partition files
        for part_number, partition_path in self.results_partitions.items():
            with h5py.File(partition_path, 'r') as partition:
                # Loop through each element type
                for etype in element_types:
                    element_group = partition.get(self.MODEL_ELEMENTS_PATH.format(model_stage=model_stages[0], element_type=etype))

                    if element_group is None:
                        print(f"Warning: Element type '{etype}' not found in partition {partition_path}.")
                        continue

                    # Loop through each element in the group
                    for element in element_group.keys():
                        element_name = element.split('[')[0]
                        file_name = part_number

                        if element_name == etype:
                            dataset = element_group[element]
                            data = dataset[:]  # Load dataset

                            # Collect info for each element
                            for idx, element_data in enumerate(data):
                                elements_info.append({
                                    'element_id': element_data[0],
                                    'element_idx': idx,
                                    'node_list': element_data[1:].tolist(),  # Node connectivity data
                                    'file_name': file_name,
                                    'element_type': etype
                                })

        # Prepare structured array and DataFrame
        if elements_info:
            dtype = [
                ('element_id', int),
                ('element_idx', int),
                ('file_name', int),
                ('element_type', object),
                ('node_list', object)  # Include node_list as an object field
            ]
            structured_data = [
                (elem['element_id'], elem['element_idx'], elem['file_name'], elem['element_type'], elem['node_list'])
                for elem in elements_info
            ]
            results_array = np.array(structured_data, dtype=dtype)

            # Convert to a Pandas DataFrame
            df = pd.DataFrame.from_records(elements_info)

            # Optionally print memory usage
            if print_memory:
                array_memory = results_array.nbytes
                df_memory = df.memory_usage(deep=True).sum()
                print(f"Memory usage for structured array (ELEMENTS): {array_memory / 1024**2:.2f} MB")
                print(f"Memory usage for DataFrame (ELEMENTS): {df_memory / 1024**2:.2f} MB")

            return {
                'array': results_array,
                'dataframe': df
            }
        else:
            print("No elements found.")
            return {
                'array': np.array([], dtype=dtype),
                'dataframe': pd.DataFrame()
            }
        
    
    def get_element_results_for_type_and_id(self, result_name, element_type, element_id, model_stage):
        
        # Check for errors in result name and element type, and model stage
        self._model_stages_error(model_stage)
        self._element_results_name_error(result_name, model_stage)
        self._element_type_name_error(element_type, model_stage)
        
        # Get the element index and info, and check if the element exists
        element_info = self._get_element_index(element_type, element_id, model_stage)
        if element_info['element_id'] is None:
            raise ValueError(f"Element ID {element_id} not found in the virtual dataset.")

        # element_info is a dictionary with the element_id, element_idx, node_coordinates_list, and file_name
        
        results_data=[]
        
        # Fetch the elements results for the element index
        with h5py.File(self.virtual_data_set, 'r') as results:
            
            element_results = results.get(self.RESULTS_ON_ELEMENTS_PATH.format(model_stage=model_stage) + f"/{result_name}")
            
            if element_results is None:
                raise ValueError("Element results group not found in the virtual dataset.")
            
            # We have a list of element results, we need to get the results for the element type, for this we need to compare the propper file name without the info after '_'
            
            for ele_type in element_results.keys():
                
                ele_type_name=ele_type.split('[')[0]
                
                if ele_type_name == element_type:
                    
                    
                    # Navigate to the "DATA" folder
                    data_group_path = f"{self.RESULTS_ON_ELEMENTS_PATH.format(model_stage=model_stage)}/{result_name}/{ele_type}/DATA"
                    data_group = results.get(data_group_path)
                    
                    if data_group is None:
                        raise ValueError(f"The DATA group does not exist under the path'.")
                    
                    ####################################
                    # VER POR ACA
                    ####################################
                    
                    all_steps = list(data_group.keys())
                    # Filter steps ending with the specific file_id
                    relevant_steps = np.array([step for step in all_steps if step.endswith(element_info['file_name'])])
                    
                    if len(relevant_steps) == 0:
                        raise ValueError(f"No data found for Element ID {element_id} in file.")
                    
                    # Extract step numbers and sort using np.argsort
                    step_numbers = np.array([int(step.split("_")[1]) for step in relevant_steps])
                    sorted_indices = np.argsort(step_numbers)
                    sorted_steps = relevant_steps[sorted_indices]
                    num_steps = len(sorted_steps)
                    
                    # Determine the number of components from the first dataset
                    first_dataset = data_group[sorted_steps[0]]
                    num_components = first_dataset.shape[1] if len(first_dataset.shape) > 1 else 1
                    
                    # Preallocate NumPy array
                    results_data = np.zeros((num_steps, 1 + num_components))  # 1 column for step + components
                    
                    # Extract data for the node index
                    for i, step_name in enumerate(sorted_steps):
                        step_num = int(step_name.split("_")[1])
                        result_data = data_group[step_name][element_info['element_idx']]
                        results_data[i, 0] = step_num  # First column: step number
                        results_data[i, 1:] = result_data  # Remaining columns: result components
                        
                        
        return results_data
    
    def get_element_nodes(self, element_id):
        """
        Retrieve the node IDs for a specific element.

        Args:
            element_id (int): Element ID to query.

        Returns:
            list: A list of node IDs associated with the given element.
        """
        with h5py.File(self.virtual_data_set, 'r') as h5file:
            # Locate the element in the mapping
            element_mapping = h5file['/Mappings/Elements'][:]
            element_index = np.where(element_mapping['element_id'] == element_id)[0]
            if element_index.size == 0:
                raise ValueError(f"Element ID {element_id} not found.")
            
            element_file = element_mapping[element_index[0]]['file_name'].decode()

            # Fetch the connectivity information from the dataset
            element_group_path = f"/MODEL/ELEMENTS/{element_file}/CONNECTIVITY"
            connectivity_group = h5file.get(element_group_path)
            if connectivity_group is None:
                raise ValueError(f"Connectivity information not found for file {element_file}.")

            # Get the node connectivity for the specific element
            connectivity_data = connectivity_group[element_id]
            return connectivity_data.tolist()

    

                        

                    
                    
                    


                
                    

                            


                    

    
    
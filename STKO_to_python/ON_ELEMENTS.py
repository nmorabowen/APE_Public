import h5py
import numpy as np
import matplotlib.pyplot as plt


class ON_ELEMENTS:
    """This is a mixin class to work with MPCO_VirtualDataset"""
    
    def _get_all_element_index(self, element_type=None):
        """
        Fetch information for all elements of a given type in a specific model stage.
        If no element type is provided, fetch information for all element types.

        Args:
            element_type (str or None): The type of elements to fetch (e.g., 'ElasticBeam3d'). 
                                        If None, fetches all element types.
            model_stage (str): The model stage to query.

        Returns:
            list: A list of dictionaries containing element information.
        """
        model_stages = self.get_model_stages()

        # Get all element types if none specified
        if element_type is None:
            element_types = self._get_all_types(model_stage=model_stages[0])
        else:
            element_types = [element_type]

        # List to store all elements' information
        elements_info = []

        with h5py.File(self.virtual_data_set, 'r') as results:
            # Loop through each element type
            for etype in element_types:
                element_group = results.get(self.MODEL_ELEMENTS_PATH.format(model_stage=model_stages[0], element_type=etype))
                
                if element_group is None:
                    print(f"Warning: Element type '{etype}' not found in the virtual dataset.")
                    continue
                
                # Loop through each element in the group
                for element in element_group.keys():
                    element_name = element.split('[')[0]
                    file_name = element.split('file')[1]

                    if element_name == etype:
                        dataset = element_group[element]
                        data = dataset[:]  # Load dataset

                        # Collect info for each element
                        for idx, element_data in enumerate(data):
                            elements_info.append({
                                'element_id': element_data[0],
                                'element_idx': idx,
                                'node_coordinates_list': element_data[1:],  # Node data (e.g., coordinates or connectivity)
                                'file_name': '_file' + file_name,
                                'element_type': etype
                            })

        return elements_info


    
    def _get_element_index(self, element_type, element_id):
        
        model_stages = self.get_model_stages()
        # Check for errors in element type and model stage
        self._element_type_name_error(element_type, model_stages[0])
        
        # Get the element index
        with h5py.File(self.virtual_data_set, 'r') as results:
            
            # Get the element group names, ie. ElasticBeam3d, ASDShell, etc
            element_group = results.get(self.MODEL_ELEMENTS_PATH.format(model_stage=model_stages[0], element_type=element_type))
            
            if element_group is None:
                raise ValueError(f"Element type '{element_type}' not found in the virtual dataset.")
            
            # Go through the element partitions files fetching for the element id
            
            element_info = {
            'element_id': None,
            'element_idx': None,
            'node_coordinates_list': None,
            'file_name': None
            }
            
            for element in element_group.keys():
                # Get the relevant name data
                element_name = element.split('[')[0]
                file_name = element.split('_file')[1]

                if element_name == element_type:
                    dataset = element_group[element]
                    # The dataset is made up of an id and the nodes that make up the element, if it is a beam 2 nodes, a Q4 shell 4 nodes, etc
                    data = dataset[:]
                    
                    if element_id in data[:,0]:
                        node_idx = np.where(data[:,0]==element_id)[0][0]
                        element_info['element_id'] = element_id
                        element_info['element_idx'] = node_idx
                        element_info['node_coordinates_list'] = data[node_idx,1:]
                        element_info['file_name'] = '_file'+file_name
            
            return element_info
    
    def _get_multiple_elements_index(self, element_type, element_ids, model_stage):
        """
        Fetch information for multiple elements in a single pass.

        Args:
            element_type (str): The type of elements to fetch (e.g., 'ElasticBeam3d').
            element_ids (int, list, or np.ndarray): A single element ID, a list, or a NumPy array of element IDs to retrieve.
            model_stage (str): The model stage to query.

        Returns:
            list: A list of dictionaries containing element information for each element ID.
        """
        # Normalize element_ids to a list
        if isinstance(element_ids, (int, np.integer)):
            element_ids = [element_ids]
        elif isinstance(element_ids, np.ndarray):
            element_ids = element_ids.tolist()
        elif not isinstance(element_ids, list):
            raise TypeError("element_ids must be an int, list, or np.ndarray.")

        # Check for errors in element type and model stage
        self._model_stages_error(model_stage)
        self._element_type_name_error(element_type, model_stage)

        elements_info = []
        with h5py.File(self.virtual_data_set, 'r') as results:
            # Get the element group
            element_group = results.get(self.MODEL_ELEMENTS_PATH.format(model_stage=model_stage, element_type=element_type))
            
            if element_group is None:
                raise ValueError(f"Element type '{element_type}' not found in the virtual dataset.")
            
            for element in element_group.keys():
                element_name = element.split('[')[0]
                file_name = element.split('_file')[1]
                if element_name == element_type:
                    dataset = element_group[element]
                    data = dataset[:]
                    
                    # Process all requested element_ids in one go
                    for element_id in element_ids:
                        if element_id in data[:, 0]:
                            node_idx = np.where(data[:, 0] == element_id)[0][0]
                            elements_info.append({
                                'element_id': element_id,
                                'element_idx': node_idx,
                                'node_coordinates_list': data[node_idx, 1:],
                                'file_name': '_file' + file_name
                            })
        
        return elements_info
    
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

    

                        

                    
                    
                    


                
                    

                            


                    

    
    
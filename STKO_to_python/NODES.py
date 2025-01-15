import h5py
import numpy as np
import matplotlib.pyplot as plt



class NODES:
    """ 
    This is a mixin class to be used with the MCPO_VirtualDataset to handle the nodes of the dataset. 
    """
    
    def _get_all_nodes_ids(self):
        """
        Retrieve all node IDs, file names, indices, and coordinates for a given model stage.
        
        Args:
            model_stage (str): The model stage to query.
        
        Returns:
            np.ndarray: A structured array with node IDs, file names, indices, and individual coordinates (x, y, z).
        """
        
        model_stages= self.get_model_stages()
        
        with h5py.File(self.virtual_data_set, 'r') as results:
            nodes_group = results.get(self.MODEL_NODES_PATH.format(model_stage=model_stages[0]))
            if nodes_group is None:
                raise ValueError("Nodes group not found in the virtual dataset.")
            
            nodes = []
            file_ids = []
            indices = []
            xs, ys, zs = [], [], []
            for key in nodes_group.keys():
                if key.startswith("ID"):
                    file_id = key.replace("ID_", "")
                    node_ids = nodes_group[key][...]
                    coord_key = key.replace("ID", "COORDINATES")
                    if coord_key in nodes_group:
                        coords = nodes_group[coord_key][...]
                        nodes.extend(node_ids)
                        file_ids.extend([file_id] * len(node_ids))
                        indices.extend(range(len(node_ids)))
                        xs.extend(coords[:, 0])
                        ys.extend(coords[:, 1])
                        zs.extend(coords[:, 2])
            
            return np.array([nodes, file_ids, indices, xs, ys, zs])


    
    def get_node_id(self, model_stage, node_ids):
        """
        Get node information including indices, file locations, and coordinates for specified node IDs.
        Parameters
        ----------
        model_stage : str
            The model stage to retrieve node information from
        node_ids : int or list
            Single node ID or list of node IDs to look up
        Returns
        -------
        list
            List of dictionaries containing node information with keys:
            - 'node_id': Original node ID
            - 'index': Index position in the dataset
            - 'file': Source file name
            - 'coordinates': Node coordinates (if available)
        Notes
        -----
        Uses HDF5 file structure to efficiently lookup node information.
        Will convert single node_id input to list format internally.
        Verifies model stage before processing.
        """
        # Convert single node_id to list if necessary
        if not isinstance(node_ids, list):
            node_ids = [node_ids]
            
        # Verify the model stage
        self._model_stages_error(model_stage=model_stage)
        
        """Get nodes' indices, file locations, and coordinates."""
        with h5py.File(self.virtual_data_set, 'r') as h5file:
            # Get node index and file
            base_path = self.MODEL_NODES_PATH.format(model_stage=model_stage)
            nodes_group = h5file.get(base_path)
            
            nodes_info = []
            
            # Find node locations
            for node_id in node_ids:
                node_info = {'node_id': node_id}
                
                for dset_name in nodes_group.keys():
                    if dset_name.startswith("ID"):
                        dataset = nodes_group[dset_name]
                        data = dataset[:]  # Read data once
                        if node_id in data[:]:
                            node_info['index'] = np.where(data == node_id)[0][0]
                            node_info['file'] = dset_name.replace("ID_", "")
                            # Get coordinates
                            coord_key = dset_name.replace("ID", "COORDINATES")
                            if coord_key in nodes_group:
                                node_info['coordinates'] = nodes_group[coord_key][node_info['index']]
                            nodes_info.append(node_info)
                            break
                        
            return nodes_info
    
    def get_node_coordinates(self, model_stage, node_ids=None):
        """
        Retrieve the coordinates of specified node IDs for a given model stage.
        
        Args:
            model_stage (str): Name of the model stage.
            node_ids (list, optional): List of node IDs to retrieve. If None, retrieves all nodes.
        
        Returns:
            dict: Dictionary containing 'node list' and 'coordinates' as NumPy arrays.
        """
        if node_ids is not None and not isinstance(node_ids, list):
            raise TypeError("node_ids must be a list")
        
        with h5py.File(self.virtual_data_set, 'r') as results:
            nodes_group = results.get(self.MODEL_NODES_PATH.format(model_stage=model_stage))
            if nodes_group is None:
                raise ValueError("Nodes group not found in the virtual dataset.")
            
            coords, node_list = [], []
            for key in nodes_group.keys():
                if key.startswith("ID"):
                    ids = nodes_group[key][...]
                    coord_key = key.replace("ID", "COORDINATES")
                    if coord_key in nodes_group:
                        coordinates = nodes_group[coord_key][...]
                        if node_ids is None:
                            node_list.extend(ids)
                            coords.extend(coordinates)
                        else:
                            id_to_index = {id_: idx for idx, id_ in enumerate(ids)}
                            for node_id in node_ids:
                                if node_id in id_to_index:
                                    node_list.append(node_id)
                                    coords.append(coordinates[id_to_index[node_id]])
            return {'node list': np.array(node_list), 'coordinates': np.array(coords)}
            
    def _get_nodal_results(self, model_stage, node_id, results_name):
        """
        Retrieve nodal results (e.g., displacement, velocity) for a specific node.

        Args:
            model_stage (str): The model stage name.
            node_id (int): The ID of the node.
            results_name (str): The name of the results group.

        Returns:
            np.ndarray: A NumPy array with rows containing step number and result components.
        """
        # Validate the results name
        self._nodal_results_name_error(results_name, model_stage)
        
        # Get node location details
        node_results = self.get_node_id(model_stage, node_id)
        file_id = node_results[0]['file']
        node_index = node_results[0]['index']
        
        if file_id is None or node_index is None:
            raise ValueError(f"Node ID {node_id} not found in the dataset.")
        
        with h5py.File(self.virtual_data_set, 'r') as h5file:
            # Construct the base path and validate
            base_path = r'{model_stage}/RESULTS/ON_NODES/{results_name}'.format(
                model_stage=model_stage, results_name=results_name
            )

            nodes_group = h5file.get(base_path)
            if nodes_group is None:
                raise ValueError(f"The path '{base_path}' does not exist in the HDF5 file.")
            
            # Get all datasets under the DATA group
            data_group = nodes_group.get("DATA")
            if data_group is None:
                raise ValueError(f"The DATA group does not exist under the path '{base_path}'.")
            
            all_steps = list(data_group.keys())
            
            # Filter steps ending with the specific file_id
            relevant_steps = np.array([step for step in all_steps if step.endswith(file_id)])
            
            if len(relevant_steps) == 0:
                raise ValueError(f"No data found for Node ID {node_id} in file {file_id}.")
            
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
                result_data = data_group[step_name][node_index]
                results_data[i, 0] = step_num  # First column: step number
                results_data[i, 1:] = result_data  # Remaining columns: result components
                
        return results_data
    
    def get_nodal_results(self, model_stage, node_ids, results_name):
        """
        Get nodal results for a single node ID, list of node IDs, or NumPy array of node IDs.
        The results are stored in a 3D NumPy array.

        Args:
            model_stage (str): The model stage name.
            node_ids (int, list, or np.ndarray): Single node ID, a list, or a NumPy array of node IDs.
            results_name (str): The name of the result to retrieve.

        Returns:
            np.ndarray: A 3D NumPy array containing results for all requested nodes.
        """
        # Convert single integer to a NumPy array
        if isinstance(node_ids, int):
            node_ids = np.array([node_ids])
        # Convert list to NumPy array
        elif isinstance(node_ids, list):
            node_ids = np.array(node_ids)
        # Validate the input
        if not isinstance(node_ids, np.ndarray) or node_ids.size == 0:
            raise ValueError("node_ids must be a non-empty NumPy array, list, or a single integer")
        
        # Retrieve results for the first node to determine the shape
        node_zero = node_ids[0]
        results_data = self._get_nodal_results(model_stage, node_zero, results_name)
        matrix_shape = results_data.shape

        # Create a 3D array to store results
        shape = (len(node_ids),) + matrix_shape
        results_array = np.zeros(shape)
        results_array[0, :, :] = results_data

        # Retrieve results for the remaining nodes
        if len(node_ids) > 1:
            for i in range(1, len(node_ids)):
                node_id = node_ids[i]
                results_data = self._get_nodal_results(model_stage, node_id, results_name)
                results_array[i, :, :] = results_data

        return results_array
    
    def get_node_info(self, model_stage, node_id):
        """Get node's index, file location, and coordinates."""
        
        with h5py.File(self.virtual_data_set, 'r') as h5file:
            # Get node index and file
            base_path = f"{model_stage}/RESULTS/ON_NODES/DISPLACEMENT"
            nodes_group = h5file.get(base_path)
            
            node_info = {'node_id': node_id}
            
            # Find node location
            for dset_name in nodes_group.keys():
                if dset_name.startswith("ID"):
                    dataset = nodes_group[dset_name]
                    if node_id in dataset[:]:
                        node_info['index'] = np.where(dataset[:] == node_id)[0][0]
                        node_info['file'] = dset_name.replace("ID_", "")
                        break
            
            # Get coordinates
            coords = self.get_node_coordinates(model_stage, [node_id])
            if coords and len(coords['coordinates']) > 0:
                node_info['coordinates'] = coords['coordinates'][0]
                
            return node_info

    
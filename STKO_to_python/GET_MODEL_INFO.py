import h5py


class GetModelInfo:
    """This is a mixin class to be used with the MCPO_VirtualDataset retreive model information fom the dataset."""
    
    def get_model_stages(self, verbose=False):
        """
        Retrieve model stages from the virtual dataset.
        
        Args:
            verbose (bool, optional): If True, prints the model stages.
        
        Returns:
            list: List of model stage names.
        """
        with h5py.File(self.virtual_data_set, 'r') as results:
            model_stages = [key for key in results.keys() if key.startswith("MODEL_STAGE")]
            
            if not model_stages:
                raise ValueError("No model stages found in the virtual dataset.")
            
            if verbose:
                print(f'The model stages found are: {model_stages}')
            return model_stages
    
    def get_elements_results_names(self, model_stage, verbose=False):
        """
        Retrieve the names of element results for a given model stage.
        
        Args:
            model_stage (str): Name of the model stage.
            verbose (bool, optional): If True, prints the element results names.
        
        Returns:
            list: List of element results names.
        """
        # Check for model stage errors
        self._model_stages_error(model_stage)
        
        with h5py.File(self.virtual_data_set, 'r') as results:
            ele_results = results.get(self.RESULTS_ON_ELEMENTS_PATH.format(model_stage=model_stage))
            if ele_results is None:
                raise ValueError("Element results group not found in the virtual dataset.")
            
            element_results_names = list(ele_results.keys())
            if verbose:
                print(f'The element results names found are: {element_results_names}')
            return element_results_names
    
    def get_element_types(self, model_stage, results_name=None, verbose=False):
        """
        Retrieve the element types for a given result name and model stage.
        If results_name is None, get types for all results.
        
        Args:
            model_stage (str): Name of the model stage.
            results_name (str, optional): Name of the results group.
            verbose (bool, optional): If True, prints the element types.
        
        Returns:
            dict: Dictionary mapping results names to their element types.
        """
        # Check for model stage errors
        self._model_stages_error(model_stage)
        
        element_types_dict = {}
        with h5py.File(self.virtual_data_set, 'r') as results:
            if results_name is None:
                results_names = self.get_elements_results_names(model_stage)
            else:
                self._element_results_name_error(results_name, model_stage)
                results_names = [results_name]
            
            for name in results_names:
                ele_types = results.get(self.MODEL_ELEMENTS_PATH.format(model_stage=model_stage))
                if ele_types is None:
                    raise ValueError(f"Element types group not found for {name}")
                element_types_dict[name] = [key.split('[')[0] for key in ele_types.keys()]
            
            if verbose:
                print(f'The element types found are: {element_types_dict}')
            return element_types_dict
        
    def _get_all_types(self, model_stage):
        
        # Check for model stage errors
        self._model_stages_error(model_stage)
        
        element_types = set()
        
        with h5py.File(self.virtual_data_set, 'r') as results:
            results_names = self.get_elements_results_names(model_stage)
            for name in results_names:
                ele_types = results.get(self.RESULTS_ON_ELEMENTS_PATH.format(model_stage=model_stage) + f"/{name}")
                if ele_types is None:
                    raise ValueError(f"Element types group not found for {name}")
                element_types.update([key.split('[')[0] for key in ele_types.keys()])
            
            return sorted(list(element_types))
        
    def get_node_results_names(self, model_stage, verbose=False):
        """
        Retrieve the names of node results names for a given model stage.
        
        Args:
            model_stage (str): Name of the model stage.
            verbose (bool, optional): If True, prints the node results names.
        
        Returns:
            list: List of node results names.
        """
        
        #Check for model stage errors
        self._model_stages_error(model_stage)
        
        with h5py.File(self.virtual_data_set, 'r') as results:
            nodes_groups = results.get(self.RESULTS_ON_NODES_PATH.format(model_stage=model_stage))
            if nodes_groups is None:
                raise ValueError("Nodes results group not found in the virtual dataset.")
            
            nodes_results = list(nodes_groups.keys())
            if verbose:
                print(f'The node results found are: {nodes_results}')
            
            return nodes_results
        
    def get_nodes_by_z_coordinate(self, model_stage, z_value, tolerance=1e-6):
        """
        Retrieve all nodes at a specific z-coordinate value within a given tolerance.
        
        Args:
            model_stage (str): The model stage to query.
            z_value (float): The target z-coordinate value.
            tolerance (float, optional): The tolerance for comparing z-coordinates. Defaults to 1e-6.
        
        Returns:
            list: A list of node IDs at the specified z-coordinate.
        """
        with h5py.File(self.virtual_data_set, 'r') as h5file:
            # Access the nodes group
            nodes_group = h5file.get(self.MODEL_NODES_PATH.format(model_stage=model_stage))
            if nodes_group is None:
                raise ValueError(f"Nodes group not found for model stage '{model_stage}'.")

            node_ids = []
            for key in nodes_group.keys():
                if key.startswith("ID"):
                    ids = nodes_group[key][...]  # Node IDs
                    coord_key = key.replace("ID", "COORDINATES")
                    if coord_key in nodes_group:
                        coordinates = nodes_group[coord_key][...]  # Node coordinates
                        # Filter nodes by the z-coordinate
                        for i, coord in enumerate(coordinates):
                            if abs(coord[2] - z_value) <= tolerance:  # Compare z-coordinate
                                node_ids.append(int(ids[i]))
            return node_ids
        

                    

        

    
    
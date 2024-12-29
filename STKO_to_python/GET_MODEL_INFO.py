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
                print(f'The element results found are: {element_results_names}')
            return element_results_names
    
    def get_element_types(self, results_name, model_stage, verbose=False):
        """
        Retrieve the element types for a given result name and model stage.
        
        Args:
            results_name (str): Name of the results group.
            model_stage (str): Name of the model stage.
            verbose (bool, optional): If True, prints the element types.
        
        Returns:
            list: List of element types.
        """
        # Check for model stage errors
        self._model_stages_error(model_stage)
        #Check result name errors
        self._element_results_name_error(results_name, model_stage)
        
        with h5py.File(self.virtual_data_set, 'r') as results:
            ele_types = results.get(self.RESULTS_ON_ELEMENTS_PATH.format(model_stage=model_stage) + f"/{results_name}")
            if ele_types is None:
                raise ValueError("Element types group not found in the virtual dataset.")
            
            element_types = list(ele_types.keys())
            if verbose:
                print(f'The element types found are: {element_types}')
            return element_types
        
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

    
    

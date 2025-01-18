import h5py
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd



class NODES:
    """ 
    This is a mixin class to be used with the MCPO_VirtualDataset to handle the nodes of the dataset. 
    """
    
    def _get_all_nodes_ids(self, print_memory=False):
        """
        Retrieve all node IDs, file names, indices, and coordinates from the partition files.

        This method processes partition files, extracts node IDs and their corresponding coordinates, and returns 
        the results in both a structured NumPy array and a pandas DataFrame. It also provides an option to print 
        memory usage for both data representations.

        Args:
            print_memory (bool): If True, prints the memory usage of the structured array and DataFrame.

        Returns:
            dict: A dictionary containing:
                - 'array': A structured NumPy array with all node IDs, file names, indices, and coordinates (x, y, z).
                - 'dataframe': A pandas DataFrame with the same data.
        """
        import pandas as pd

        node_data = []

        for part_number, partition_path in self.results_partitions.items():
            with h5py.File(partition_path, 'r') as partition:
                nodes_group = partition.get(self.MODEL_NODES_PATH.format(model_stage=self.get_model_stages()[0]))
                if nodes_group is None:
                    continue  # Skip this partition if the nodes group is not found

                for key in nodes_group.keys():
                    if key.startswith("ID"):
                        file_id = part_number
                        node_ids = nodes_group[key][...]
                        coord_key = key.replace("ID", "COORDINATES")
                        if coord_key in nodes_group:
                            coords = nodes_group[coord_key][...]
                            for index, (node_id, coord) in enumerate(zip(node_ids, coords)):
                                node_data.append((node_id, file_id, index, coord[0], coord[1], coord[2]))

        # Convert the list to a structured NumPy array
        dtype = [
            ('node_id', 'i8'),
            ('file_id', 'i8'),
            ('index', 'i8'),
            ('x', 'f8'),
            ('y', 'f8'),
            ('z', 'f8')
        ]

        results_array = np.array(node_data, dtype=dtype)

        # Convert to a Pandas DataFrame
        columns = ['node_id', 'file_id', 'index', 'x', 'y', 'z']
        df = pd.DataFrame(node_data, columns=columns)

        results_dict = {
            'array': results_array, 
            'dataframe': df
        }

        if print_memory:
            array_memory = results_array.nbytes
            df_memory = df.memory_usage(deep=True).sum()
            print(f"Memory usage for structured array (NODES): {array_memory / 1024**2:.2f} MB")
            print(f"Memory usage for DataFrame (NODES): {df_memory / 1024**2:.2f} MB")

        return results_dict

    
    def _get_nodal_results_mapping(self, model_stage, results_name, node_ids=None, overwrite=False):
        """
        Map nodal results into an HDF5 group for efficient access. If `node_ids` is None, process all nodes.

        Args:
            model_stage (str): The model stage name.
            results_name (str): The name of the results group.
            node_ids (list, optional): List of node IDs to process. If None, process all nodes in the model stage.
            overwrite (bool): If True, overwrite existing data. Defaults to False.

        Returns:
            None
        """
        # Validate the results name
        self._nodal_results_name_error(results_name, model_stage)

        # Retrieve all nodes if `node_ids` is not provided
        if node_ids is None:
            nodes_info = self._get_all_nodes_ids()
            if nodes_info.size == 0:
                raise ValueError(f"No nodes found in the model stage '{model_stage}'.")
        else:
            # Retrieve node info: A list of tuples (node_id, file_id, index)
            nodes_info = self.get_node_files_and_indices(node_ids=node_ids)
            if nodes_info.size == 0:
                raise ValueError("No nodes found in the dataset.")

        # Open the HDF5 file
        with h5py.File(self.virtual_data_set, 'a') as h5file:
            # Create or retrieve the output group
            output_group_path = f"/TH/{model_stage}/{results_name}"
            if output_group_path in h5file:
                output_group = h5file[output_group_path]
            else:
                output_group = h5file.create_group(output_group_path)

            for node_id, file_id, node_index, _, _, _ in nodes_info:
                file_info=self.partition_files[file_id]
            
            # Loop over partition files
            for part_number, partition_path in self.results_partitions.items():
                with h5py.File(partition_path, 'r') as partition:
                    # Construct the base path
                    base_path = self.RESULTS_ON_NODES_PATH.format(
                        model_stage=model_stage
                    ) + f"/{results_name}"

                    nodes_group = partition.get(base_path)
                    if nodes_group is None:
                        print(f"The path '{base_path}' does not exist in the partition '{partition_path}'. Skipping.")
                        continue

                    # Access the "DATA" group
                    data_group = nodes_group.get("DATA")
                    if data_group is None:
                        raise ValueError(f"The DATA group does not exist under the path '{base_path}' in partition '{partition_path}'.")

                    # Loop over each node's information
                    for node_id, file_id, node_index, _, _, _ in nodes_info:
                        # Filter steps by file_id and sort them
                        all_steps = list(data_group.keys())
                        relevant_steps = [step for step in all_steps if step.endswith(str(file_id))]

                        if not relevant_steps:
                            print(f"No data found for Node ID {node_id} in partition '{partition_path}'.")
                            continue

                        step_numbers = [int(step.split("_")[1]) for step in relevant_steps]
                        sorted_indices = np.argsort(step_numbers)
                        sorted_steps = [relevant_steps[i] for i in sorted_indices]

                        # Check if the dataset for this node already exists
                        node_dataset_path = f"{output_group_path}/Node_{node_id}"
                        if node_dataset_path in h5file:
                            if overwrite:
                                del h5file[node_dataset_path]  # Remove if overwrite is True
                            else:
                                print(f"Dataset for Node {node_id} already exists in '{output_group_path}'. Skipping.")
                                continue

                        num_steps = len(sorted_steps)
                        first_dataset = data_group[sorted_steps[0]]
                        num_components = first_dataset.shape[1] if len(first_dataset.shape) > 1 else 1

                        # Preallocate the dataset in the output group
                        node_dataset = output_group.create_dataset(
                            f"Node_{node_id}",
                            shape=(num_steps, 1 + num_components),
                            dtype=first_dataset.dtype
                        )

                        # Populate the dataset
                        for i, step_name in enumerate(sorted_steps):
                            step_num = int(step_name.split("_")[1])
                            result_data = data_group[step_name][node_index]
                            node_dataset[i, 0] = step_num  # Step number
                            node_dataset[i, 1:] = result_data  # Result components

            print(f"Results for nodes {'all' if node_ids is None else node_ids} mapped to group '{output_group_path}' in the HDF5 file.")

        
        
            
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
    
    def get_nodal_results(self, model_stage=None, results_name=None, node_ids=None, selection_set_id=None):
        """
        Get nodal results optimized for numerical operations.
        Returns results as a structured NumPy array or DataFrame for efficient computation.
        
        Args:
            model_stage (str, optional): The model stage name. If None, gets results for all stages.
            results_name (str): The name of the result to retrieve.
            node_ids (int, list, or np.ndarray, optional): Single node ID, a list, or a NumPy array of node IDs.
            selection_set_id (int, optional): The ID of the selection set to retrieve nodes from.

        Returns:
            pd.DataFrame: A DataFrame with MultiIndex (stage, node_id, step) if model_stage is None,
                        or Index (node_id, step) if model_stage is specified.
                        Columns represent the components of the results.
        """
        # Input validation
        node_ids = self._validate_and_prepare_inputs(
            model_stage, results_name, node_ids, selection_set_id
        )

        # If no specific model stage is given, process all stages
        if model_stage is None:
            all_results = []
            for stage in self.model_stages:
                try:
                    stage_results = self._get_stage_results(
                        stage, results_name, node_ids
                    )
                    stage_results['stage'] = stage  # Add stage information
                    all_results.append(stage_results)
                except Exception as e:
                    print(f"Warning: Could not retrieve results for stage {stage}: {str(e)}")
                    continue
            
            if not all_results:
                raise ValueError("No results found for any stage")
            
            # Combine all stages into a single DataFrame
            return pd.concat(all_results, axis=0)
        
        # If specific model stage is given, process just that stage
        return self._get_stage_results(model_stage, results_name, node_ids)

    def _get_stage_results(self, model_stage, results_name, node_ids):
        """Helper function to get results for a specific model stage."""
        # Get node files and indices information
        nodes_info = self.get_node_files_and_indices(node_ids=node_ids)
        
        # Group nodes by file_id for batch processing
        file_groups = nodes_info.groupby('file_id')
        
        # Base path for results
        base_path = f"{model_stage}/RESULTS/ON_NODES/{results_name}/DATA"
        
        # List to store all results before combining
        all_results = []
        
        # Process each file only once, reading multiple nodes
        for file_id, group in file_groups:
            with h5py.File(self.results_partitions[int(file_id)], 'r') as results:
                data_group = results.get(base_path)
                if data_group is None:
                    raise ValueError(f"DATA group not found in path '{base_path}'.")
                
                # Get all step names once
                step_names = list(data_group.keys())
                
                # Pre-fetch all node indices for this file
                node_indices = group['index'].values
                file_node_ids = group['node_id'].values
                
                # Process all steps
                for step_idx, step_name in enumerate(step_names):
                    dataset = data_group[step_name]
                    # Read all required indices at once
                    step_data = dataset[node_indices]
                    
                    # Create DataFrame for this step
                    step_df = pd.DataFrame(
                        step_data,
                        index=file_node_ids,
                        columns=[f'component_{i}' for i in range(step_data.shape[1])]
                    )
                    step_df['step'] = step_idx
                    step_df['step_name'] = step_name
                    step_df['node_id'] = file_node_ids
                    
                    all_results.append(step_df)
        
        if not all_results:
            raise ValueError(f"No results found for stage {model_stage}")
        
        # Combine all results into a single DataFrame
        combined_results = pd.concat(all_results, axis=0)
        
        # Set up MultiIndex
        combined_results.set_index(['node_id', 'step'], inplace=True)
        combined_results.sort_index(inplace=True)
        
        return combined_results

    def _validate_and_prepare_inputs(self, model_stage, results_name, node_ids, selection_set_id):
        """Helper function to validate inputs and prepare node_ids."""
        # Input validation
        if node_ids is not None and selection_set_id is not None:
            raise ValueError("Only one of 'node_ids' or 'selection_set_id' can be provided.")
        if node_ids is None and selection_set_id is None:
            raise ValueError("Either 'node_ids' or 'selection_set_id' must be provided.")
        
        # Results name validation
        if results_name not in self.node_results_names:
            raise ValueError(f"Results name '{results_name}' not found in the dataset.")
        
        # Model stage validation (only if specified)
        if model_stage is not None and model_stage not in self.model_stages:
            raise ValueError(f"Model stage '{model_stage}' not found in the dataset.")
        
        # Handle selection set
        if selection_set_id is not None:
            selection_set = self.selection_set[selection_set_id]
            if not selection_set or "NODES" not in selection_set:
                raise ValueError(f"Selection set ID '{selection_set_id}' does not contain nodes.")
            return np.array(selection_set["NODES"])
        
        # Handle node_ids
        if isinstance(node_ids, int):
            return np.array([node_ids])
        elif isinstance(node_ids, list):
            return np.array(node_ids)
        elif isinstance(node_ids, np.ndarray) and node_ids.size > 0:
            return node_ids
        else:
            raise ValueError("node_ids must be a non-empty NumPy array, list, or a single integer.")
                
                



    


    
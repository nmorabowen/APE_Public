import h5py
import numpy as np

class MAPPING:
    def build_and_store_mappings(self):
        """
        Build and store node and element mappings for efficient lookups.
        Includes node lists for each element with -1 padding for consistency.
        """
        node_mappings = []
        element_mappings = []

        with h5py.File(self.virtual_data_set, 'r+') as h5file:
            # Use the first model stage to fetch node coordinates since they are consistent across stages
            model_stages = self.get_model_stages()
            if not model_stages:
                raise ValueError("No model stages found in the dataset.")

            # Fetch and map nodes from the first stage only
            first_stage = model_stages[0]
            nodes_data = self.get_node_coordinates(model_stage=first_stage)
            node_ids = nodes_data['node list']
            coordinates = nodes_data['coordinates']

            for node_id, (x, y, z) in zip(node_ids, coordinates):
                node_mappings.append((node_id, x, y, z))

            # Fetch and map elements for all model stages
            element_data = self._get_all_element_index()
            
            # Calculate maximum number of nodes per element for array sizing
            max_nodes = max(len(elem['node_coordinates_list']) for elem in element_data)
            
            # Create element mappings including the node list
            for elem in element_data:
                # Create a fixed-size array of node IDs, padded with -1
                node_list = np.full(max_nodes, -1, dtype=np.int64)
                # Fill in the actual node IDs
                node_list[:len(elem['node_coordinates_list'])] = elem['node_coordinates_list']
                
                element_mappings.append((elem['element_id'], elem['file_name'], elem['element_idx'], elem['element_type'], node_list))

            # Save to HDF5 with compression and chunking
            mapping_group = h5file.require_group("/Mappings")
            
            # Convert to structured numpy arrays for better query efficiency
            node_dtype = np.dtype([
                ('node_id', 'i8'),
                ('x', 'f8'),
                ('y', 'f8'),
                ('z', 'f8')
            ])
            
            element_dtype = np.dtype([
                ('element_id', 'i8'),
                ('file_name', 'S50'),
                ('element_idx', 'i8'),
                ('element_type', 'S50'),
                ('node_list', f'i8', (max_nodes,))  # Fixed-size array for node IDs
            ])

            # Save datasets with compression and chunking
            mapping_group.create_dataset(
                "Nodes", 
                data=np.array(node_mappings, dtype=node_dtype),
                chunks=True,
                compression="gzip"
            )
            mapping_group.create_dataset(
                "Elements",
                data=np.array(element_mappings, dtype=element_dtype),
                chunks=True,
                compression="gzip"
            )
            
            # Store metadata
            mapping_group.attrs['max_nodes'] = max_nodes

    def get_node_files_and_indices(self, node_ids):
        """
        Retrieve files and indices for a list of nodes.

        Args:
            node_ids (list[int]): List of Node IDs to lookup.

        Returns:
            list: A list of tuples (file_name, index) for each node found.
        """
        if not isinstance(node_ids, list):
            raise ValueError("node_ids should be a list of integers")
        
        with h5py.File(self.virtual_data_set, 'r') as h5file:
            nodes_mapping = h5file['/Mappings/Nodes'][:]
            results = []
            for node_id in node_ids:
                indices = np.where(nodes_mapping['node_id'] == node_id)[0]
                if indices.size > 0:
                    idx = indices[0]
                    results.append((nodes_mapping[idx]['file_name'].decode(), nodes_mapping[idx]['index']))
            return results

    def get_element_files_and_indices(self, element_ids):
        """
        Retrieve files and indices for a list of elements.

        Args:
            element_ids (list[int]): List of Element IDs to lookup.

        Returns:
            list: A list of tuples (file_name, index) for each element found.
        """
        if not isinstance(element_ids, list):
            raise ValueError("element_ids should be a list of integers")
        
        with h5py.File(self.virtual_data_set, 'r') as h5file:
            element_mapping = h5file['/Mappings/Elements'][:]
            results = []
            for element_id in element_ids:
                indices = np.where(element_mapping['element_id'] == element_id)[0]
                if indices.size > 0:
                    idx = indices[0]
                    results.append((element_mapping[idx]['file_name'].decode(), element_mapping[idx]['element_idx']))
            return results
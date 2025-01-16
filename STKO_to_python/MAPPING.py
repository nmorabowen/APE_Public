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

            for part_number, partition_path in self.results_partitions.items():
                with h5py.File(partition_path, 'r') as partition:
                    nodes_info = self._get_all_nodes_ids()

                    for node_id, file_id, index, x, y, z in nodes_info:
                        node_mappings.append((node_id, file_id, index, x, y, z))

                    # Fetch and map elements for this partition
                    element_data = self._get_all_element_index()

                    # Calculate maximum number of nodes per element for array sizing
                    max_nodes = max(len(elem['node_list']) for elem in element_data)

                    # Create element mappings including the node list
                    for elem in element_data:
                        # Create a fixed-size array of node IDs, padded with -1
                        node_list = np.full(max_nodes, -1, dtype=np.int64)
                        # Fill in the actual node IDs
                        node_list[:len(elem['node_list'])] = elem['node_list']

                        element_mappings.append((
                            elem['element_id'],
                            elem['file_name'],
                            elem['element_idx'],
                            elem['element_type'],
                            node_list
                        ))

            # Save to HDF5 with compression and chunking
            mapping_group = h5file.require_group("/Mappings")

            # Convert to structured numpy arrays for better query efficiency
            node_dtype = np.dtype([
                ('node_id', 'i8'),
                ('file_id', 'S50'),
                ('index', 'i8'),
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

            # Write the mappings
            node_array = np.array(node_mappings, dtype=node_dtype)
            element_array = np.array(element_mappings, dtype=element_dtype)

            # Save datasets with compression
            mapping_group.create_dataset("Nodes", data=node_array, compression="gzip", compression_opts=9)
            mapping_group.create_dataset("Elements", data=element_array, compression="gzip", compression_opts=9)

    def get_node_files_and_indices(self, node_ids):
        """
        Retrieve files, indices, and coordinates for a list of nodes using vectorized NumPy operations.

        Args:
            node_ids (list[int]): List of Node IDs to lookup.

        Returns:
            np.ndarray: A structured array with dtype containing 'node_id', 'file_id', 'index', 'x', 'y', 'z'.
        """
        if not isinstance(node_ids, list) or not all(isinstance(id, int) for id in node_ids):
            raise ValueError("node_ids should be a list of integers")

        with h5py.File(self.virtual_data_set, 'r') as h5file:
            # Load mappings as structured array
            nodes_mapping = h5file['/Mappings/Nodes'][:]
            node_ids_array = np.array(node_ids)

            # Find indices of matches using vectorized operations
            sorter = np.argsort(nodes_mapping['node_id'])
            sorted_ids = nodes_mapping['node_id'][sorter]
            indices = sorter[np.searchsorted(sorted_ids, node_ids_array)]

            # Create mask for valid matches
            mask = (indices < len(nodes_mapping)) & (nodes_mapping['node_id'][indices] == node_ids_array)

            # Initialize results with default dtype
            dtype = np.dtype([
                ('node_id', 'i8'),
                ('file_id', 'U50'),
                ('index', 'i8'),
                ('x', 'f8'),
                ('y', 'f8'),
                ('z', 'f8')
            ])
            results = np.zeros(len(node_ids), dtype=dtype)

            # Fill results with default values for unmatched nodes
            results['node_id'] = node_ids_array
            results['file_id'] = np.full(len(node_ids), '', dtype='U50')
            results['index'] = -1
            results['x'] = np.nan
            results['y'] = np.nan
            results['z'] = np.nan

            # Fill in matched values
            if np.any(mask):
                results['file_id'][mask] = np.char.decode(nodes_mapping['file_id'][indices[mask]], 'utf-8')
                results['index'][mask] = nodes_mapping['index'][indices[mask]]
                results['x'][mask] = nodes_mapping['x'][indices[mask]]
                results['y'][mask] = nodes_mapping['y'][indices[mask]]
                results['z'][mask] = nodes_mapping['z'][indices[mask]]

            return results


    def get_element_files_and_indices(self, element_ids):
        """
        Store and query elements based on their IDs, efficiently handling large datasets.

        Args:
            element_ids (list[int]): List of Element IDs to lookup.

        Returns:
            np.ndarray: Structured array containing elements with fields:
                - element_id
                - file_name
                - element_idx
                - element_type
                - node_list
        """
        if not isinstance(element_ids, list):
            raise ValueError("element_ids should be a list of integers")

        with h5py.File(self.virtual_data_set, 'r') as h5file:
            # Load the element mappings
            element_mapping = h5file['/Mappings/Elements'][:]

            # Convert to NumPy for efficient processing
            element_ids_array = np.array(element_ids, dtype=np.int64)

            # Sort and find matches using vectorized operations
            sorter = np.argsort(element_mapping['element_id'])
            sorted_ids = element_mapping['element_id'][sorter]
            indices = sorter[np.searchsorted(sorted_ids, element_ids_array)]

            # Create mask for valid matches
            mask = (indices < len(element_mapping)) & (element_mapping['element_id'][indices] == element_ids_array)

            # Prepare structured array for results
            dtype = np.dtype([
                ('element_id', 'i8'),
                ('file_name', 'U50'),
                ('element_idx', 'i8'),
                ('element_type', 'U50'),
                ('node_list', 'i8', (element_mapping['node_list'].shape[1],))  # Fixed-size array for nodes
            ])

            results = np.zeros(len(element_ids_array), dtype=dtype)

            # Fill results with valid matches
            if np.any(mask):
                matched_elements = element_mapping[indices[mask]]
                results['element_id'][mask] = matched_elements['element_id']
                results['file_name'][mask] = matched_elements['file_name']
                results['element_idx'][mask] = matched_elements['element_idx']
                results['element_type'][mask] = matched_elements['element_type']
                results['node_list'][mask] = matched_elements['node_list']

            # Mark unmatched elements as None
            unmatched_mask = ~mask
            results['element_id'][unmatched_mask] = element_ids_array[unmatched_mask]
            results['file_name'][unmatched_mask] = "None"
            results['element_idx'][unmatched_mask] = -1
            results['element_type'][unmatched_mask] = "None"
            results['node_list'][unmatched_mask] = -1

            return results

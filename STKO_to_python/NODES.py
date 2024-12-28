import h5py
import numpy as np
import glob  # Import the glob module
import os
import yaml
import matplotlib.pyplot as plt





class NODES:
    """ 
    This is a mixin class to be used with the MCPO_VirtualDataset to handle the nodes of the dataset. 
    """
    
    # Common path templates
    MODEL_NODES_PATH = "/{model_stage}/MODEL/NODES"
    RESULTS_ON_ELEMENTS_PATH = "/{model_stage}/RESULTS/ON_ELEMENTS"
    RESULTS_ON_NODES_PATH = "/{model_stage}/RESULTS/ON_NODES"
    
    def get_node_id(self, model_stage, node_id):
        
        # Verify the model stage
        self._model_stages_error(model_stage=model_stage)
        
        """Get node's index, file location, and coordinates."""
        with h5py.File(self.virtual_data_set, 'r') as h5file:
            # Get node index and file
            base_path = self.MODEL_NODES_PATH.format(model_stage=model_stage)
            nodes_group = h5file.get(base_path)
            
            node_info = {'node_id': node_id}
            
            # Find node location
            for dset_name in nodes_group.keys():
                if dset_name.startswith("ID"):
                    dataset = nodes_group[dset_name]
                    data = dataset[:]  # Read data once
                    if node_id in data:
                        node_info['index'] = np.where(data == node_id)[0][0]
                        node_info['file'] = dset_name.replace("ID_", "")
                        break
                
            return node_info
    
    def get_node_coordinates_optimized(self, model_stage, node_ids=None):
        
        with h5py.File(self.virtual_data_set, 'r') as results:
            nodes_group = results.get(self.MODEL_NODES_PATH.format(model_stage=model_stage))
            
            if nodes_group is None:
                raise ValueError("Nodes group not found")
                
            if node_ids is not None:
                # For specific nodes - get direct mapping
                coords = []
                found_ids = []
                for node_id in node_ids:
                    for key in nodes_group.keys():
                        if key.startswith("ID"):
                            dataset = nodes_group[key]
                            if node_id in dataset[:]:
                                idx = np.where(dataset[:] == node_id)[0][0]
                                coord_key = key.replace("ID", "COORDINATES")
                                coords.append(nodes_group[coord_key][idx])
                                found_ids.append(node_id)
                                break
                return {'node list': np.array(found_ids), 'coordinates': np.array(coords)}
            else:
                # For all nodes - use virtual datasets directly
                all_coords = []
                all_ids = []
                for key in nodes_group.keys():
                    if key.startswith("ID"):
                        coord_key = key.replace("ID", "COORDINATES")
                        if coord_key in nodes_group:
                            all_ids.extend(nodes_group[key][:])
                            all_coords.extend(nodes_group[coord_key][:])
                return {'node list': np.array(all_ids), 'coordinates': np.array(all_coords)}
    
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
    
    
    def find_node_data_links(self, model_stage, node_id):
        """Find all data links for a specific node."""
        
        with h5py.File(self.virtual_data_set, 'r') as h5file:
            base_path = self.MODEL_NODES_PATH.format(model_stage=model_stage)
            nodes_group = h5file.get(base_path)
            
            links = {
                'node_location': None,
                'step_data': []
            }
            
            # Find node's file and index
            for dset_name in nodes_group.keys():
                if dset_name.startswith("ID"):
                    dataset = nodes_group[dset_name]
                    if node_id in dataset[:]:
                        node_index = np.where(dataset[:] == node_id)[0][0]
                        file_id = dset_name.replace("ID_", "")
                        links['node_location'] = {
                            'file': file_id,
                            'dataset': dset_name,
                            'index': node_index
                        }
                        break
            
            # Find associated step data
            if links['node_location']:
                data_group = nodes_group.get("DATA")
                for step_name in data_group.keys():
                    if step_name.endswith(file_id):
                        step_num = int(step_name.split("_")[1])
                        links['step_data'].append({
                            'step': step_num,
                            'dataset': step_name,
                            'virtual': data_group[step_name].is_virtual
                        })
                        
            return links
    
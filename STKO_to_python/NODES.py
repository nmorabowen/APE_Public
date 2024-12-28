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
                        if node_id in data:
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
    
    
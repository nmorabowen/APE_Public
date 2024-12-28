import h5py
import numpy as np
import glob  # Import the glob module
import os
import yaml
import matplotlib.pyplot as plt

from .NODES import NODES
from .ERRORS import errorChecks
from .GET_MODEL_INFO import GetModelInfo

class MCPO_VirtualDataset(NODES, errorChecks, GetModelInfo):
    
    # Common path templates
    MODEL_NODES_PATH = "/{model_stage}/MODEL/NODES"
    RESULTS_ON_ELEMENTS_PATH = "/{model_stage}/RESULTS/ON_ELEMENTS"
    RESULTS_ON_NODES_PATH = "/{model_stage}/RESULTS/ON_NODES"

    def __init__(self, results_directory, results_directory_name='results_h5', results_filename='results.h5', file_extension='*.mpco'):
        """
        Initialize the MCPO_VirtualDataset instance.
        
        Args:
            results_directory (str): Path to the directory containing the source partition files.
            results_directory_name (str): Name of the directory to store the virtual dataset.
            results_filename (str): Name of the virtual dataset file.
            file_extension (str): File extension to identify partition files.
        """
        self.results_directory = results_directory
        self.file_extension = file_extension
        
        # Get the results partitions
        self.results_partitions = self._get_results_partitions()
        
        # Define the database path and directory
        self._define_virtual_paths(results_directory_name=results_directory_name, results_filename=results_filename)
        
        # Create the virtual dataset directory
        self._create_results_directory()
        
        # Create the virtual dataset
        self.create_virtual_dataset()
        
        # Cache information
        self._node_location_cache={}
        self._coordinates_cache={}
    
    
    
    def _define_virtual_paths(self, results_directory_name, results_filename):
        """
        Define paths for the virtual dataset and its directory.
        """
        self.virtual_data_set_directory = os.path.join(self.results_directory, results_directory_name)
        self.virtual_data_set = os.path.join(self.virtual_data_set_directory, results_filename)
    
    def _create_results_directory(self):
        """
        Ensure that the directory for the virtual dataset exists.
        """
        os.makedirs(self.virtual_data_set_directory, exist_ok=True)
    
    def _get_results_partitions(self):
        """
        Retrieve a sorted list of partition files matching the specified file extension.
        
        Returns:
            list: Sorted list of partition file paths.
        """
        results_partitions = sorted(glob.glob(f"{self.results_directory}/{self.file_extension}"))
        return results_partitions
    
    def create_virtual_dataset(self):
        """
        Create the virtual dataset by linking datasets from source partition files.
        """
        if os.path.exists(self.virtual_data_set):
            os.remove(self.virtual_data_set)

        def copy_structure(source_group, target_group, file_name, file_index):
            """
            Recursively copy groups and datasets from source files into the virtual dataset.
            """
            for key in source_group.keys():
                item = source_group[key]
                if isinstance(item, h5py.Group):
                    # Recursively handle groups
                    new_group = target_group.require_group(key)
                    copy_structure(item, new_group, file_name, file_index)
                elif isinstance(item, h5py.Dataset):
                    # Create a virtual layout for the dataset
                    dataset_name = f"{key}_file{file_index}"
                    if dataset_name not in target_group:
                        vsource = h5py.VirtualSource(file_name, item.name, shape=item.shape)
                        layout = h5py.VirtualLayout(shape=item.shape, dtype=item.dtype)
                        layout[:] = vsource
                        target_group.create_virtual_dataset(dataset_name, layout)

        # Create the virtual dataset
        with h5py.File(self.virtual_data_set, 'w') as results:
            for i, file in enumerate(self.results_partitions):
                with h5py.File(file, 'r') as source:
                    for group_name in source.keys():
                        copy_structure(source[group_name], results.require_group(group_name), file, i)

        print(f"Virtual dataset created successfully at {self.virtual_data_set}")

    def read_virtual_dataset(self):
        """
        Read the structure of the virtual dataset and print it in YAML format.
        """
        def traverse_datasets(h5_obj, tree):
            """
            Recursively traverse groups and datasets to build the structure.
            """
            for key in h5_obj.keys():
                if isinstance(h5_obj[key], h5py.Group):
                    tree[key] = {}
                    traverse_datasets(h5_obj[key], tree[key])
                elif isinstance(h5_obj[key], h5py.Dataset):
                    tree[key] = {'shape': h5_obj[key].shape, 'dtype': str(h5_obj[key].dtype)}
        
        file_structure = {}
        with h5py.File(self.virtual_data_set, 'r') as results:
            traverse_datasets(results, file_structure)
        
        yaml_output = yaml.dump(file_structure, default_flow_style=False)
        print("HDF5 File Structure in YAML Format:")
        print(yaml_output)
    
    def get_node_results(self, model_stage, verbose=False):
        """
        Retrieve the names of node results for a given model stage.
        
        Args:
            model_stage (str): Name of the model stage.
            verbose (bool, optional): If True, prints the node results names.
        
        Returns:
            list: List of node results names.
        """
        with h5py.File(self.virtual_data_set, 'r') as results:
            nodes_groups = results.get(self.RESULTS_ON_NODES_PATH.format(model_stage=model_stage))
            if nodes_groups is None:
                raise ValueError("Nodes results group not found in the virtual dataset.")
            
            nodes_results = list(nodes_groups.keys())
            if verbose:
                print(f'The node results found are: {nodes_results}')
            return nodes_results
        
    def get_node_response_history(self, model_stage, node_id, plot=False):
        
        with h5py.File(self.virtual_data_set, 'r') as results:
            base_path = f"{model_stage}/RESULTS/ON_NODES/DISPLACEMENT"
            nodes_group = results.get(base_path)
            
            for key in nodes_group.keys():
                if key.startswith("ID"):
                    dataset = nodes_group[key]
                    if node_id in dataset[:]:
                        node_index = np.where(dataset[:] == node_id)[0][0]
                        file_num = key.replace("ID_file", "")
                        
                        disp_group = results.get(f"{base_path}/DATA")
                        steps_data = []
                        
                        for step_key in disp_group.keys():
                            if f"_file{file_num}" in step_key:
                                step_num = int(step_key.split('_')[1])
                                data = disp_group[step_key][node_index].tolist()
                                steps_data.append([step_num] + data)
                        
                        response_history = np.array(sorted(steps_data))
                        
                        if plot and len(response_history) > 0:
                            plt.figure(figsize=(10, 6))
                            
                            # Plot each component (x,y,z)
                            components = ['X', 'Y', 'Z']
                            for i in range(1, 4):
                                plt.plot(response_history[:, 0], response_history[:, i], 
                                        label=f'{components[i-1]}-displacement')
                            
                            plt.xlabel('Step Number')
                            plt.ylabel('Displacement')
                            plt.title(f'Node {node_id} Displacement History')
                            plt.legend()
                            plt.grid(True)
                            plt.show()
                            
                        return response_history
                        
        raise ValueError(f"Node ID {node_id} not found")
    
    
    def _find_node_location(self, h5file, model_stage, node_id):
        
        with h5py.File(self.virtual_data_set, 'r') as results:

            base_path = f"{model_stage}/RESULTS/ON_NODES/DISPLACEMENT"
            nodes_group = results.get(base_path)
            
            for key in nodes_group.keys():
                if key.startswith("ID"):
                    dataset = nodes_group[key]
                    if node_id in dataset[:]:
                        node_index = np.where(dataset[:] == node_id)[0][0]
                        file_name = key.replace("ID_file", "")
                        return file_name, node_index
            raise ValueError(f"Node ID {node_id} not found")
        
    def check_node_displacement_links(self, model_stage, node_id):
        """Check HDF5 references/links between node IDs and displacement data."""
        with h5py.File(self.virtual_data_set, 'r') as h5file:
            base_path = f"{model_stage}/RESULTS/ON_NODES/DISPLACEMENT"
            nodes_group = h5file.get(base_path)
            
            # Check for HDF5 references or external links
            for name, obj in nodes_group.items():
                if isinstance(obj, h5py.Dataset):
                    if obj.attrs.get("TARGET") or hasattr(obj, "is_virtual"):
                        return True
                
            return False
        
    def get_node_response_history_via_links(self, model_stage, node_id):
        with h5py.File(self.virtual_data_set, 'r') as h5file:
            base_path = f"{model_stage}/RESULTS/ON_NODES/DISPLACEMENT"
            nodes_group = h5file.get(base_path)
            
            # Get direct reference
            if hasattr(nodes_group, "refs") and node_id in nodes_group.refs:
                displacement_data = nodes_group.refs[node_id]
                return displacement_data[:]
                
            # Alternative: check virtual mappings
            if hasattr(nodes_group, "virtual_sources"):
                for source in nodes_group.virtual_sources():
                    if source.target_name == str(node_id):
                        return source.get_target()[:]
    
    def inspect_displacement_structure(self, model_stage):
        with h5py.File(self.virtual_data_set, 'r') as h5file:
            base_path = f"{model_stage}/RESULTS/ON_NODES/DISPLACEMENT"
            nodes_group = h5file.get(base_path)
            
            print("Group structure:")
            def print_structure(name, obj):
                print(f"{name}:")
                if isinstance(obj, h5py.Dataset):
                    print(f"  Type: Dataset")
                    print(f"  Shape: {obj.shape}")
                    print(f"  Virtual: {obj.is_virtual}")
                elif isinstance(obj, h5py.Group):
                    print(f"  Type: Group")
                    
            nodes_group.visititems(print_structure)
            
    def get_node_displacement_direct(self, model_stage, node_id):
        with h5py.File(self.virtual_data_set, 'r') as h5file:
            base_path = f"{model_stage}/RESULTS/ON_NODES/DISPLACEMENT"
            nodes_group = h5file.get(base_path)
            
            # Get file mapping from ID datasets
            file_id = None
            for dset_name in nodes_group.keys():
                if dset_name.startswith("ID"):
                    dataset = nodes_group[dset_name]
                    if node_id in dataset[:]:
                        node_index = np.where(dataset[:] == node_id)[0][0]
                        file_id = dset_name.replace("ID_", "")
                        break
                        
            if file_id is None:
                return None
            
            print(file_id)
            
            # Get direct data from matching file in DATA group
            data_group = nodes_group.get("DATA")
            steps_data = []
            
            for step_name in data_group.keys():
                if step_name.endswith(file_id):
                    step_num = int(step_name.split("_")[1])
                    disp_data = data_group[step_name][node_index].tolist()
                    steps_data.append([step_num] + disp_data)
                    
            return np.array(sorted(steps_data))
    
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
    
        
    def find_node_references(self, node_id):
        """Find all references to a node ID across the database."""
        references = {}
        
        with h5py.File(self.virtual_data_set, 'r') as h5file:
            def check_node(name, obj):
                if isinstance(obj, h5py.Dataset):
                    try:
                        if obj.shape[0] > 0:  # Only check non-empty datasets
                            if node_id in obj[:]:
                                node_index = np.where(obj[:] == node_id)[0][0]
                                references[name] = {
                                    'type': 'Dataset',
                                    'shape': obj.shape,
                                    'virtual': obj.is_virtual,
                                    'index': node_index
                                }
                    except Exception as e:
                        pass  # Skip datasets that can't be checked
                        
            h5file.visititems(check_node)
            
        return references
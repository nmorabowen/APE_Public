import h5py
import numpy as np
import glob  # Import the glob module
import os
import yaml
import matplotlib.pyplot as plt

from .NODES import NODES
from .ERRORS import errorChecks
from .GET_MODEL_INFO import GetModelInfo
from .PLOTTER import plotter

class MCPO_VirtualDataset(NODES, errorChecks, GetModelInfo, plotter):
    
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
    


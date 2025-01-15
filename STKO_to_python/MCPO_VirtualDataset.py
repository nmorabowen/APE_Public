import h5py
import numpy as np
import glob  # Import the glob module
import os
import yaml
import matplotlib.pyplot as plt
from collections import defaultdict
import concurrent.futures

from .NODES import NODES
from .ON_ELEMENTS import ON_ELEMENTS
from .ERRORS import errorChecks
from .GET_MODEL_INFO import GetModelInfo
from .PLOTTER import plotter
from .CDATA import CDATA
from .MAPPING import MAPPING

class MCPO_VirtualDataset(NODES, 
                          ON_ELEMENTS, 
                          errorChecks, 
                          GetModelInfo, 
                          plotter,
                          CDATA,
                          MAPPING):
    
    # Common path templates
    MODEL_NODES_PATH = "/{model_stage}/MODEL/NODES"
    MODEL_ELEMENTS_PATH = "/{model_stage}/MODEL/ELEMENTS"
    RESULTS_ON_ELEMENTS_PATH = "/{model_stage}/RESULTS/ON_ELEMENTS"
    RESULTS_ON_NODES_PATH = "/{model_stage}/RESULTS/ON_NODES"

    def __init__(self, results_directory, recorder_name=None, results_directory_name='results_h5', results_filename='results.h5', file_extension='*.mpco'):
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
        self.recorder_name = recorder_name
        
        # Get the results partitions
        self.results_partitions = self._get_results_partitions()
        
        # Define the database path and directory
        self._define_virtual_paths(results_directory_name=results_directory_name, results_filename=results_filename)
        
        # Create the virtual dataset directory
        self._create_results_directory()
        
        # Create the virtual dataset
        self.create_virtual_dataset()
        self.build_and_store_mappings()
        

    
    
    
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
        print("Using the following partition files:")
        for partition in results_partitions:
            print(f"  - {partition}")
        return results_partitions
    
    def create_virtual_dataset(self):
        """
        Create or update the virtual dataset by linking datasets from source partition files.
        """
        if os.path.exists(self.virtual_data_set):
            print(f"Virtual dataset already exists at {self.virtual_data_set}. File will be overwritten.")
            os.remove(self.virtual_data_set)
        else:
            print(f"Creating virtual dataset at {self.virtual_data_set}...")

        def copy_structure(source_group, target_group, file_name, file_index):
            """
            Recursively copy groups, datasets, and attributes from source files into the virtual dataset.
            """
            for key in source_group.keys():
                item = source_group[key]
                if isinstance(item, h5py.Group):
                    # Require group (create only if not exists)
                    new_group = target_group.require_group(key)

                    # Copy attributes (if not already present)
                    for attr_name, attr_value in item.attrs.items():
                        if attr_name not in new_group.attrs:
                            new_group.attrs[attr_name] = attr_value

                    # Recursive call
                    copy_structure(item, new_group, file_name, file_index)
                elif isinstance(item, h5py.Dataset):
                    # Create or update virtual dataset
                    dataset_name = f"{key}_file{file_index}"
                    if dataset_name not in target_group:
                        vsource = h5py.VirtualSource(file_name, item.name, shape=item.shape)
                        layout = h5py.VirtualLayout(shape=item.shape, dtype=item.dtype)
                        layout[:] = vsource
                        virtual_dataset = target_group.create_virtual_dataset(dataset_name, layout)

                        # Copy attributes
                        for attr_name, attr_value in item.attrs.items():
                            virtual_dataset.attrs[attr_name] = attr_value

        # Open the virtual dataset file
        with h5py.File(self.virtual_data_set, 'a') as results:
            for i, file in enumerate(self.results_partitions):
                with h5py.File(file, 'r') as source:
                    for group_name in source.keys():
                        copy_structure(source[group_name], results.require_group(group_name), file, i)

        print(f"Virtual dataset successfully created or updated at {self.virtual_data_set}")
        
    def create_reduced_hdf5(self, node_ids, element_ids, output_file):
        """
        Create a reduced HDF5 file containing only the specified nodes and elements with their attributes.

        Args:
            node_ids (list): List of node IDs to include in the reduced file.
            element_ids (list): List of element IDs to include in the reduced file.
            output_file (str): Path to save the reduced HDF5 file.
        """
        with h5py.File(self.virtual_data_set, 'r') as original_file, h5py.File(output_file, 'w') as reduced_file:
            # Copy relevant nodes
            for model_stage in self.get_model_stages():
                nodes_path = self.MODEL_NODES_PATH.format(model_stage=model_stage)
                if nodes_path in original_file:
                    node_info = self.get_node_coordinates(model_stage, node_ids)
                    node_group = reduced_file.require_group(nodes_path)
                    node_list = node_info['node list']
                    coords = node_info['coordinates']

                    node_group.create_dataset('node_ids', data=node_list)
                    node_group.create_dataset('coordinates', data=coords)

                # Copy relevant elements
                elements_path = self.RESULTS_ON_ELEMENTS_PATH.format(model_stage=model_stage)
                if elements_path in original_file:
                    element_group = original_file[elements_path]
                    reduced_element_group = reduced_file.require_group(elements_path)
                    for element_id in element_ids:
                        if f"ID_{element_id}" in element_group:
                            original_data = element_group[f"ID_{element_id}"]
                            reduced_element_group.create_dataset(f"ID_{element_id}", data=original_data)

            print(f"Reduced HDF5 file created at {output_file}")
    


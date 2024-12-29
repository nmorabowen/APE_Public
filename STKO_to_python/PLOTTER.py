import numpy as np
import matplotlib.pyplot as plt

class plotter:
    """This is a mixin class for plotting results for the MCPO_VirtualDataset class"""
    
    def plot_nodal_results(self, model_stage, results_name, node_ids, directions=None, ax=None, figsize=None, normalize=False, scaling_factor=1.0, linewidth=0.75, color='#000077', marker=None, plot_width_mm=210, aspect_ratio=0.5, dpi=150, z_sorted=False):
        """
        Plot nodal results for multiple nodes and directions, optionally sorting by Z-coordinate.
        
        Args:
            model_stage (str): The model stage name.
            results_name (str): The name of the results to plot (e.g., 'DISPLACEMENT').
            node_ids (list): List of node IDs to plot.
            directions (list or str, optional): List of directions (e.g., ['x', 'y']) or a single direction (e.g., 'y').
            ax (matplotlib axes, optional): Pre-created axes for plotting.
            figsize (tuple, optional): Figure size if axes are not provided.
            normalize (bool, optional): If True, make all axes limits uniform based on the global max value.
            scaling_factor (float, optional): Global scaling factor to apply to all `y` values. Defaults to 1.0.
            linewidth (float, optional): Line width for the plot. Defaults to 0.75.
            color (str, optional): Line color for the plot. Defaults to '#000077'.
            marker (str, optional): Marker for the plot. Defaults to None.
            plot_width_mm (float, optional): Width of the plot in millimeters. Defaults to 210 (A4 width).
            aspect_ratio (float, optional): Aspect ratio (height-to-width ratio for each plot). Defaults to 0.5.
            dpi (int, optional): Dots per inch for the figure. Defaults to 300.
            z_sorted (bool, optional): If True, sort nodes by Z-coordinate. Defaults to False.
        
        Returns:
            matplotlib.axes: Axes with the plots.
        """
        # Retrieve node coordinates if z_sorted is enabled
        if z_sorted:
            node_infos = [self.get_node_info(model_stage, node_id) for node_id in node_ids]
            z_coords = [info['coordinates'][2] for info in node_infos]  # Extract Z-coordinates
            sorted_indices = np.argsort(z_coords)[::-1]  # Sort by Z-coordinate (high to low)
            node_ids = np.array(node_ids)[sorted_indices]  # Reorder node IDs
            z_coords = np.array(z_coords)[sorted_indices]  # Reorder Z-coordinates

        number_of_nodes = len(node_ids)
        valid_directions = ['x', 'y', 'z']

        # Normalize and validate directions
        if directions is None:
            directions = valid_directions  # Default: All directions
        elif isinstance(directions, str):
            directions = [directions.lower()]  # Single direction
        elif isinstance(directions, list):
            directions = [d.lower() for d in directions]  # Normalize list
        else:
            raise ValueError("Directions must be a list or a single string.")

        # Validate directions
        for dir in directions:
            if dir not in valid_directions:
                raise ValueError(f"Invalid direction: {dir}. Valid directions are {valid_directions}.")

        number_of_directions = len(directions)

        # Dynamically calculate figsize if not provided
        if figsize is None:
            width_in_inches = plot_width_mm / 25.4  # Convert mm to inches
            height_in_inches = number_of_nodes * number_of_directions * aspect_ratio
            figsize = (width_in_inches, height_in_inches)

        # Ensure axes are created or provided
        fig, ax = plt.subplots(number_of_nodes, number_of_directions, figsize=figsize, squeeze=False, dpi=dpi)

        # Get the nodal results
        results = self.get_nodal_results(model_stage=model_stage, results_name=results_name, node_ids=node_ids)

        # Compute global min and max for uniform axes if normalize is enabled
        global_min, global_max = None, None
        if normalize:
            global_min = np.min(results[:, :, 1:] * scaling_factor)
            global_max = np.max(results[:, :, 1:] * scaling_factor)

        # Plot data
        for j, node in enumerate(node_ids):  # Loop over nodes
            for i, dir in enumerate(directions):  # Loop over directions
                col_idx = valid_directions.index(dir) + 1  # Map direction to column index
                y_values = results[j, :, col_idx] * scaling_factor  # Apply scaling factor

                label = f'Node {node}'
                if z_sorted:
                    label += f' (Z={z_coords[j]:.0f})'  # Append Z-coordinate to label

                ax[j, i].plot(results[j, :, 0], y_values, label=label, linewidth=linewidth, color=color, marker=marker)
                ax[j, i].set_xlabel('Time', fontsize=6)
                ylabel = f'{results_name} ({dir})'
                if normalize:
                    ylabel += " (Normalized)"
                ax[j, i].set_ylabel(ylabel, fontsize=6)
                ax[j, i].legend(fontsize=8, edgecolor='none', loc=3)
                ax[j, i].tick_params(axis='both', which='major', labelsize=8)
                ax[j, i].grid(True)

                # Set uniform axes limits if normalize is enabled
                if normalize:
                    ax[j, i].set_ylim(global_min, global_max)

        # Adjust layout
        plt.tight_layout()
        
        return ax
    
    def plot_single_node_result(self, model_stage, results_name, node_id, direction, ax=None, figsize=(10, 6), scaling_factor=1.0, linewidth='0.75', color='#000077', marker=None):
        """
        Plot a specific result for a single node in a given direction.

        Args:
            model_stage (str): The model stage name.
            results_name (str): The name of the result to plot (e.g., 'DISPLACEMENT').
            node_id (int): The ID of the node to plot.
            direction (str): The direction ('x', 'y', or 'z') to plot.
            ax (matplotlib axes, optional): Pre-created axes for plotting.
            figsize (tuple, optional): Size of the figure. Defaults to (8, 6).
            scaling_factor (float, optional): Scaling factor for the results. Defaults to 1.0.
            linewidth (float, optional): Line width for the plot. Defaults to 1.0.
            color (str, optional): Line color for the plot. Defaults to 'b'.
            marker (str, optional): Marker style for the plot. Defaults to None.

        Returns:
            matplotlib.axes: Axes with the plot.
        """
        # Validate the direction
        valid_directions = ['x', 'y', 'z']
        if direction.lower() not in valid_directions:
            raise ValueError(f"Invalid direction: {direction}. Must be one of {valid_directions}.")

        # Retrieve the results
        results = self.get_nodal_results(model_stage=model_stage, node_ids=node_id, results_name=results_name)

        # Map direction to column index
        col_idx = valid_directions.index(direction.lower()) + 1
        time_steps = results[0,:, 0]  # First column is time steps
        values = results[0,:, col_idx] * scaling_factor  # Apply scaling factor to results

        # Create axes if not provided
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)

        # Plot the data
        ax.plot(time_steps, values, label=f"Node {node_id} ({direction.upper()})", linewidth=linewidth, color=color, marker=marker)

        # Customize the plot
        ax.set_xlabel('Time')
        ax.set_ylabel(f'{results_name} ({direction.upper()})')
        ax.legend()
        ax.grid(True)

        return ax

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
    
    def plot_node(
        self,
        model_stage,
        results_name_verticalAxis=None,
        node_ids_verticalAxis=None,
        selection_set_id_verticalAxis=None,
        direction_verticalAxis=None,
        values_operation_verticalAxis='Sum',
        scaling_factor_verticalAxis=1.0,
        results_name_horizontalAxis=None,
        node_ids_horizontalAxis=None,
        selection_set_id_horizontallAxis=None,
        direction_horizontalAxis=None,
        values_operation_horizontalAxis='Sum',
        scaling_factor_horizontalAxis=1.0,
        ax=None,
        figsize=(10, 6),
        color='k',
        linetype='-',
        linewidth=0.75,
        label=None,
    ):
        """
        Plots nodal results with specified parameters.
        
        Args:
            model_stage (str): Name of the model stage to retrieve data.
            results_name_verticalAxis (str): Name of the vertical axis results.
            node_ids_verticalAxis (list): List of node IDs for vertical axis results.
            selection_set_id_verticalAxis (int): Selection set ID for vertical axis.
            direction_verticalAxis (str): Direction of vertical axis results ('x', 'y', or 'z').
            values_operation_verticalAxis (str): Aggregation operation for vertical axis ('Sum', 'Mean', etc.).
            results_name_horizontalAxis (str): Name of the horizontal axis results.
            node_ids_horizontalAxis (list): List of node IDs for horizontal axis results.
            selection_set_id_horizontallAxis (int): Selection set ID for horizontal axis.
            direction_horizontalAxis (str): Direction of horizontal axis results ('x', 'y', or 'z').
            values_operation_horizontalAxis (str): Aggregation operation for horizontal axis.
            ax (matplotlib.axes.Axes): Pre-existing axes to plot on. If None, a new figure is created.
            figsize (tuple): Size of the figure if no `ax` is provided.
            color (str): Line color for the plot.
            linetype (str): Line style for the plot.
            linewidth (float): Line width for the plot.
            label (str): Label for the legend.
            
        Returns:
            matplotlib.axes.Axes: Axes object containing the plot.
        """
        

        self._nodal_results_name_error(results_name_verticalAxis, model_stage)
        self._nodal_results_name_error(results_name_horizontalAxis, model_stage)
        

        # Retrieve results for vertical and horizontal axes
        vertical_results_df = self.get_nodal_results(
            model_stage=model_stage,
            results_name=results_name_verticalAxis,
            node_ids=node_ids_verticalAxis,
            selection_set_id=selection_set_id_verticalAxis,
        )
        horizontal_results_df = self.get_nodal_results(
            model_stage=model_stage,
            results_name=results_name_horizontalAxis,
            node_ids=node_ids_horizontalAxis,
            selection_set_id=selection_set_id_horizontallAxis,
        )

        # Aggregate results
        x_array = plotter._aggregate_results(horizontal_results_df, direction_horizontalAxis, values_operation_horizontalAxis) * scaling_factor_horizontalAxis
        y_array = plotter._aggregate_results(vertical_results_df, direction_verticalAxis, values_operation_verticalAxis) * scaling_factor_verticalAxis

        if len(x_array) != len(y_array):
            raise ValueError("Mismatch in lengths of horizontal and vertical data arrays.")

        # Create figure and axes if not provided
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)

        # Plot the data
        ax.plot(x_array, y_array, color=color, linestyle=linetype, linewidth=linewidth, label=label)
        if label:
            ax.legend()
        ax.set_xlabel(results_name_horizontalAxis or "Horizontal Axis")
        ax.set_ylabel(results_name_verticalAxis or "Vertical Axis")
        ax.grid(True)
        
        return ax
    
        
    @staticmethod
    def _aggregate_results(results_df, direction, operation):
        """
        Aggregates the results DataFrame based on the specified operation for the given direction.

        Args:
            results_df (pd.DataFrame): DataFrame with results, including a 'step' index and direction columns.
            direction (str): Direction to aggregate ('x', 'y', 'z').
            operation (str): Operation to perform ('Sum', 'Mean', 'Max', 'Min').

        Returns:
            np.ndarray: Aggregated values for the specified direction.
        """
        # Validate direction
        if direction not in results_df.columns:
            raise KeyError(f"Direction '{direction}' not found in DataFrame columns: {results_df.columns}. The available directions are: {results_df.columns}")
        
        # Group and aggregate only the relevant column
        if operation == 'Sum':
            aggregated_values = results_df.groupby('step')[direction].sum()
        elif operation == 'Mean':
            aggregated_values = results_df.groupby('step')[direction].mean()
        elif operation == 'Max':
            aggregated_values = results_df.groupby('step')[direction].max()
        elif operation == 'Min':
            aggregated_values = results_df.groupby('step')[direction].min()
        else:
            raise ValueError(f"Invalid operation: {operation}")
        
        return aggregated_values.values
        
        
    

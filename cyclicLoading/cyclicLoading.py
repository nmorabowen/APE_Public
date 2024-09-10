import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FixedLocator
import pandas as pd
import os  # Import os to help with path handling

class cyclicLoading:
    def __init__(self, loading_array=None, verbose=False, ax=None, plot_figure=False, dpi=150):
        # Initialize the loading array and check validity
        self.loading_array = loading_array
        self.verbose = verbose
        self.ax = ax
        self.plot_figure = plot_figure
        self.dpi = dpi  # Set the DPI

        # Check if loading_array is a NumPy ndarray and has two columns
        if loading_array is not None:
            if not isinstance(loading_array, np.ndarray):
                raise TypeError("loading_array must be a NumPy ndarray")
            if loading_array.ndim != 2 or loading_array.shape[1] != 2:
                raise ValueError("loading_array must be a 2D NumPy array with exactly two columns")

        # If loading_array is None, set the default loading array
        self._set_loading_array()
        
        # Create the cyclic information
        self.cycles, self.loading, self.time = self.create_cyclic_loads()
        
        # If plot is enabled, call the plot method
        if self.plot_figure:
            self.plot()

    def _set_loading_array(self):
        # Set a default loading array if none is provided
        if self.loading_array is None:
            self.loading_array = np.array([[0.25 / 12, 3],
                                           [0.50 / 12, 3],
                                           [0.75 / 12, 3],
                                           [1.00 / 12, 3],
                                           [1.50 / 12, 3],
                                           [2.00 / 12, 3],
                                           [3.00 / 12, 3],
                                           [4.00 / 12, 2],
                                           [6.00 / 12, 2],
                                           [8.00 / 12, 2],
                                           [10.00 / 12, 2],
                                           [12.00 / 12, 2]])

    def create_cyclic_loads(self):
        # Access the instance's loading_array
        loading_array = self.loading_array

        # Calculate total number of cycles
        total_cycles = np.sum(loading_array[:, 1])

        # Create the values of the cycles based on the array
        loading = np.zeros(int(2 * total_cycles))
        values = np.repeat(loading_array[:, 0], loading_array[:, 1].astype(int))
        loading[0::2] = values
        loading[1::2] = -values
        loading = np.insert(loading, 0, 0)
        loading = np.append(loading, 0)

        # Create the array of cycles
        cycles = np.arange(0, total_cycles + 1, 0.50)

        # Create the time array, scaled such that the last time value is 1
        time=cycles/total_cycles
        
        # Verbose output if required
        if self.verbose:
            print(f'The total number of cycles is: {total_cycles:.2f}')
            print(f'Each cycle magnitude is: {np.round(values, 2)}')
            print(f'Time array: {np.round(time, 2)}')
            
        return cycles, loading, time
    
    def plot(self):
        # Generate cyclic loads
        cycles, loading, time = self.cycles, self.loading, self.time

        # Use the provided ax if available, otherwise create a new one
        if self.ax is None:
            fig, ax1 = plt.subplots(figsize=(10, 5), dpi=self.dpi)  # Create the primary axis for time
        else:
            ax1 = self.ax  # Use the provided ax

        # Create a second x-axis that shares the same y-axis
        ax2 = ax1.twiny()  # Create a secondary x-axis for cycles

        # Create the x-locator array for cycles
        x_locator_array = self.loading_array[:, 1]
        x_locator_array = [sum(x_locator_array[:i+1]) for i in range(len(x_locator_array))]

        # Create the y-tick values (reduce the number of ticks by picking fewer)
        y_min, y_max = np.min(self.loading_array[:, 0]), np.max(self.loading_array[:, 0])
        y_ticks = np.linspace(-y_max, y_max, num=7)  # Set fewer y-ticks (adjust 'num' as needed)

        # Plot the cyclic loading against time on ax1
        ax1.plot(time, loading, color='#000077', label='Loading Protocol #1')
        ax1.set_xlim(0, 1)  # Time is between 0 and 1
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Rotation %')
        ax1.set_title('Loading Protocol with Dual X-Axis (Time and Cycles)')
        ax1.grid(True, which='both')
        ax1.legend(loc='upper left')

        # Plot the cyclic loading against cycles on ax2
        ax2.set_xlim(0, np.max(cycles))  # Set x-limits for cycles
        ax2.set_xlabel('Cycles')
        ax2.set_xticks(x_locator_array)  # Set custom x-ticks for cycles
        ax2.xaxis.grid(True, which='both', linestyle='--', linewidth=0.5)

        # Set y-axis ticks to correspond to the reduced loading values
        ax1.yaxis.set_major_locator(FixedLocator(y_ticks))

        # If no axis was provided, display the plot
        if self.ax is None:
            plt.show()
            
    def export_to_excel(self, filename="cyclic_loading.xlsx", export_path="."):
        # Combine the export path and filename to form the full file path
        full_path = os.path.join(export_path, filename)

        # Create a DataFrame with cycles and loading values
        df = pd.DataFrame({
            'Cycles': self.cycles,
            'Loading': self.loading,
            'Time':self.time
        })

        # Export to Excel
        df.to_excel(full_path, index=False)

        if self.verbose:
            print(f"Data exported to {full_path}")

# Only execute if running as the main module
if __name__ == "__main__":
    # Create an example case using a custom loading array
    custom_loading_array = np.array([[0.2, 3],  # 0.2 for 3 cycles
                                     [0.4, 2],  # 0.4 for 2 cycles
                                     [0.6, 4]])  # 0.6 for 4 cycles

    # Initialize the cyclicLoading object with the custom loading array and verbose mode
    cyclic_test = cyclicLoading(loading_array=custom_loading_array, verbose=True, plot_figure=True)

    # Example with default ATC loading protocol
    ATC_cyclic = cyclicLoading(verbose=True, plot_figure=True)
    

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator
import pandas as pd
import os

class cyclicLoading:
    def __init__(self, Delta=1, loading_array=None, unitTime=True, unitCycle=True, verbose=False, ax=None, plot_figure=False, dpi=150):
        # Initialize the loading array and other attributes
        self.Delta=Delta
        self.loading_array = loading_array
        self.verbose = verbose
        self.ax = ax
        self.plot_figure = plot_figure
        self.dpi = dpi
        self.unitTime=unitTime
        self.unitCycle=unitCycle

        # Check if loading_array is a NumPy ndarray and has two columns
        if loading_array is not None:
            if not isinstance(loading_array, np.ndarray):
                raise TypeError("loading_array must be a NumPy ndarray")
            if loading_array.ndim != 2 or loading_array.shape[1] != 2:
                raise ValueError("loading_array must be a 2D NumPy array with exactly two columns")

        # If loading_array is None, set the default loading array
        self._set_loading_array()
        
        # Perform calculations
        self.H = np.max(self.loading_array)
        self.m = (4 * self.H) / self.Delta
        self.cycles, self.loading, self.time = self.create_cyclic_loads()
        
        # If plot is enabled, call the plot method
        if self.plot_figure:
            self.plot()

    def _set_loading_array(self):
        # Set a default loading array if none is provided
        if self.loading_array is None:
            self.loading_array = np.array([
                [0.25, 3], [0.50, 3], [0.75, 3], [1.00, 3], [1.50, 3],
                [2.00, 3], [3.00, 3], [4.00, 2], [6.00, 2], [8.00, 2],
                [10.00, 2], [12.00, 2]
            ])

    def create_cyclic_loads(self):
        # Calculations
        xi = self.loading_array[:, 0] / self.m
        repetitions = self.loading_array[:, 1].astype(int)

        # Generate the loading protocol
        delta_array = np.repeat(xi, repetitions, axis=None)
        delta_array = np.repeat(delta_array, 4)
        delta_array = np.insert(delta_array, 0, 0)
        
        # Create the cumulative time array
        time = np.cumsum(delta_array)
        
        if self.unitTime is True:
            time=time/np.max(time)

        # Initialize cycles array and populate values at every 4th index
        cycles=self.loading_array[:,0]
        
        # Unitarize time and cycle if needed
        if self.unitCycle is True:
            cycles=cycles/np.max(cycles)
        
        cycles=np.repeat(cycles, repetitions, axis=None)
        cycles_array = np.zeros(len(time))
        cycles_array[1::4]=cycles
        cycles_array[3::4]=-cycles
        
        # Verbose output
        if self.verbose:
            if self.loading_array is not None:
                print(f'The default loading array is used')
            print(f'The maximum value from the loading array is: {self.H}')
            print(f'The slope of the loading cycles is {np.round(self.m, 3)}')

        
            
        
        return np.arange(len(cycles_array)), cycles_array, time
    
    def plot(self):
        # Use the provided ax if available, otherwise create a new one
        if self.ax is None:
            fig, ax = plt.subplots(figsize=(10, 6), dpi=self.dpi)
        else:
            ax = self.ax

        # Plot the cyclic loading
        ax.plot(self.time, self.loading, color='black', linewidth=1.5)
        ax.axhline(0, color='gray', linestyle='--', linewidth=0.5)
        ax.axvline(0, color='gray', linestyle='--', linewidth=0.5)

        # Adding labels and title
        ax.set_xlabel('Time (cumulative)')
        ax.set_ylabel('Cycles')
        ax.set_title('Loading Protocol Plot')

        # Adding grid
        ax.grid(True, linestyle='--', alpha=0.7)

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
            'Time': self.time
        })

        # Export to Excel
        df.to_excel(full_path, index=False)

        if self.verbose:
            print(f"Data exported to {full_path}")

# Example usage
if __name__ == "__main__":
    custom_loading_array = np.array([
        [0.25, 3], [0.50, 3], [0.75, 3], [1.00, 3], [1.50, 3],
        [2.00, 3], [3.00, 3], [4.00, 2], [6.00, 2], [8.00, 2],
        [10.00, 2], [12.00, 2]
    ])

    # Initialize the cyclicLoading object with the custom loading array and verbose mode
    cyclic_test = cyclicLoading(loading_array=custom_loading_array, verbose=True, plot_figure=True)
    
    # Initialize the cyclicLoading object with the custom loading array and verbose mode
    ATC_cyclic_test = cyclicLoading(unitTime=True, unitCycle=False, verbose=True, plot_figure=True)

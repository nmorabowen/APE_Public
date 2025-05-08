import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.integrate import cumulative_trapezoid


class Earthquake:
    def __init__(self, dt:float, filepath:str=None, data_array:np.ndarray=None, units:dict=None):
        
        # Error control for input data
        if filepath is None and data_array is None:
            raise ValueError("Either 'filepath' or 'data_array' must be provided.")
        
        # Default unit system
        default_units = {
            'length': {'factor': 1, 'label': None},
            'time': {'factor': 1, 'label': None},
            'acceleration': {'factor': 1, 'label': None},
        }
        
        # Merge user-defined units with defaults
        self.units = {**default_units, **(units or {})}

        self.filepath = filepath
        self.dt = dt
        self.data_array = data_array
        
        self.record_results = self.read_data()
        
    def read_data(self):
        if self.filepath is not None:
            # Load the acceleration data from file
            acceleration = np.loadtxt(self.filepath, skiprows=0)
        else:
            acceleration=np.asarray(self.data_array) #enforce array type
            
        # Determine the number of data points
        N = len(acceleration)
        
        # Create a uniform time array
        time = np.linspace(0, self.dt * (N - 1), N)

        # Extract unit scaling factors
        length_factor = self.units['length']['factor']
        time_factor = self.units['time']['factor']
        
        # Compute velocity and displacement using trapezoidal integration
        acceleration = acceleration * (length_factor / time_factor**2)
        # Integrate acceleration → velocity
        velocity = cumulative_trapezoid(acceleration, dx=self.dt, initial=0)
        # Integrate velocity → displacement
        displacement = cumulative_trapezoid(velocity, dx=self.dt, initial=0)

        # Find indices for max and min values for each signal
        accel_max_index = np.argmax(acceleration)
        accel_min_index = np.argmin(acceleration)
        velocity_max_index = np.argmax(velocity)
        velocity_min_index = np.argmin(velocity)
        displacement_max_index = np.argmax(displacement)
        displacement_min_index = np.argmin(displacement)
        
        return {
            'time': time,
            'acceleration': acceleration,
            'velocity': velocity,
            'displacement': displacement,
            'accel_max_index': accel_max_index,
            'accel_min_index': accel_min_index,
            'velocity_max_index': velocity_max_index,
            'velocity_min_index': velocity_min_index,
            'displacement_max_index': displacement_max_index,
            'displacement_min_index': displacement_min_index
        }
        
    def newmark_sdof(self, T, zeta=0.05, m=1.0, beta=0.25, gamma=0.5, plot=False):
        """
        Compute the displacement, velocity, and acceleration responses of an SDOF oscillator 
        using the Newmark method.

        The equation of motion is:
            m*u''(t) + c*u'(t) + k*u(t) = - m*ag(t)
        where ag(t) is the ground acceleration.

        For a unit mass (m=1) and a specified period T, the stiffness is computed as:
            k = 4*pi^2/T^2

        Parameters:
            ag    : Ground acceleration array (m/s²)
            dt    : Time step (s)
            T     : Oscillator period (s)
            zeta  : Damping ratio (default: 0.05)
            m     : Mass (default: 1.0)
            beta  : Newmark parameter beta (default: 0.25)
            gamma : Newmark parameter gamma (default: 0.5)
        
        Returns:
            u     : Relative displacement response (array, m)
            udot  : Relative velocity response (array, m/s)
            uacc  : Relative acceleration response (array, m/s²)
            time  : Time array corresponding to the responses.
        """
        # Number of time steps and time vector
        ag = self.record_results['acceleration']
        dt= self.dt
        
        N = len(ag)
        time = np.linspace(0, dt * (N - 1), N)
        
        # Define SDOF parameters
        k = 4 * np.pi**2 / T**2   # stiffness from period: T = 2*pi*sqrt(m/k)
        omega_n = np.sqrt(k/m)    # natural circular frequency
        c = 2 * m * zeta * omega_n  # damping coefficient

        # Newmark method constants (using the average acceleration method)
        a0 = 1.0 / (beta * dt**2)
        a1 = gamma / (beta * dt)
        a2 = 1.0 / (beta * dt)
        a3 = 1.0 / (2 * beta) - 1.0
        a4 = gamma / beta - 1.0
        a5 = dt * (gamma / (2 * beta) - 1.0)
        
        # Effective stiffness of the system
        K_eff = k + a0 * m + a1 * c
        
        # Initialize response arrays
        u = np.zeros(N)      # displacement (m)
        udot = np.zeros(N)   # velocity (m/s)
        uacc = np.zeros(N)   # acceleration (m/s²)
        
        # Initial conditions (assume zero displacement and velocity)
        u[0] = 0.0
        udot[0] = 0.0
        # Initial acceleration is computed from the equation of motion at t=0
        uacc[0] = -ag[0]
        
        # Time integration loop using Newmark's method
        for i in range(N - 1):
            # Compute the effective load at the next time step (including inertia and damping)
            P_eff = (- m * ag[i + 1] +
                    m * (a0 * u[i] + a2 * udot[i] + a3 * uacc[i]) +
                    c * (a1 * u[i] + a4 * udot[i] + a5 * uacc[i]))
            
            # Solve for displacement at time step i+1
            u[i + 1] = P_eff / K_eff
            
            # Compute acceleration at time step i+1
            uacc[i + 1] = a0 * (u[i + 1] - u[i]) - a2 * udot[i] - a3 * uacc[i]
            
            # Update velocity at tim
            # e step i+1
            udot[i + 1] = udot[i] + dt * ((1 - gamma) * uacc[i] + gamma * uacc[i + 1])
        
        if plot is True:
            self._plot_sdof_response(u, udot, uacc, time)
        
        results={'displacement': u, 
                 'velocity': udot, 
                 'acceleration': uacc, 
                 'time': time}
        
        return results
    
    def compute_response_spectrum(self, T_range, zeta=0.05, m=1.0, beta=0.25, gamma=0.5, plot=False):
        """
        Compute the response spectrum for a range of periods T with unit scaling.
        
        Parameters
        ----------
        T_range : array-like
            Array of natural periods (s) for which the response spectrum is computed.
        zeta : float, optional
            Damping ratio (default is 0.05 for 5% damping).
        m : float, optional
            Mass of the SDOF system (default is 1.0, as it cancels out in spectral ratios).
        beta : float, optional
            Newmark-beta method parameter (default is 0.25).
        gamma : float, optional
            Newmark method parameter (default is 0.5).
        plot : bool, optional
            If True, plot the computed spectra.

        Returns
        -------
        response_spectrum : dict
            A dictionary containing:
                - 'T': Natural periods
                - 'Sa': Spectral acceleration (adjusted for units)
                - 'Sv': Spectral velocity (adjusted for units)
                - 'Sd': Spectral displacement (adjusted for units)
                - 'units': Dictionary of unit labels
        """

        # Unit labels
        length_unit = self.units['length']['label']
        time_unit = self.units['time']['label']
        
        # Initialize arrays for spectral values
        Sa = np.zeros(len(T_range))
        Sv = np.zeros(len(T_range))
        Sd = np.zeros(len(T_range))
        
        # Get the ground acceleration from the stored earthquake record
        ag = self.record_results['acceleration']
        
        # Loop over each natural period
        for i, T in enumerate(T_range):
            
            # Compute time-domain SDOF response using Newmark
            response = self.newmark_sdof(T, zeta, m, beta, gamma, plot=False)
            
            # Maximum absolute relative displacement and velocity
            Sd[i] = np.max(np.abs(response['displacement']))
            Sv[i] = np.max(np.abs(response['velocity']))
            
            # ====== KEY FIX FOR SPECTRAL ACCELERATION ======
            # total_acc = (relative_acceleration + ground_acceleration)
            total_acc = response['acceleration'] + ag
            Sa[i] = np.max(np.abs(total_acc))
            # ===============================================

        # Organize results into a dictionary
        results = {
            'T': T_range,
            'Sa': Sa,
            'Sv': Sv,
            'Sd': Sd,
            'units': {
                'Sa': f'{length_unit}/{time_unit}²',
                'Sv': f'{length_unit}/{time_unit}',
                'Sd': f'{length_unit}',
                'T': f'{time_unit}'
            }
        }
        
        # Optionally, plot the response spectra
        if plot is True:
            self.plot_response_spectrum(results)
        
        return results

    def plot_response_spectrum(self, spectrum_results, ax=None, figsize=(5, 10), linewidth=0.75, linestyle='-'):
        """
        Plot the response spectrum using three subplots with correct unit labels.
        """
        
        # Extract unit labels from computed spectrum
        units = spectrum_results['units']
        
        if ax is None:
            fig, ax = plt.subplots(nrows=3, ncols=1, figsize=figsize, sharex=True)
        
        ax[0].plot(spectrum_results['T'], spectrum_results['Sa'], 
                label=f'Spectral Acceleration $S_a$ ({units["Sa"]})', 
                color='r', linewidth=linewidth, linestyle=linestyle)
        
        ax[1].plot(spectrum_results['T'], spectrum_results['Sv'], 
                label=f'Spectral Velocity $S_v$ ({units["Sv"]})', 
                color='b', linewidth=linewidth, linestyle=linestyle)
        
        ax[2].plot(spectrum_results['T'], spectrum_results['Sd'], 
                label=f'Spectral Displacement $S_d$ ({units["Sd"]})', 
                color='g', linewidth=linewidth, linestyle=linestyle)
        
        ax[0].set_ylabel(f'Acceleration ({units["Sa"]})')
        ax[1].set_ylabel(f'Velocity ({units["Sv"]})')
        ax[2].set_ylabel(f'Displacement ({units["Sd"]})')
        ax[2].set_xlabel(f'Period ({units["T"]})')
        
        ax[0].set_title('Spectral Acceleration')
        ax[1].set_title('Spectral Velocity')
        ax[2].set_title('Spectral Displacement')
        
        ax[-1].set_xlim(0, max(spectrum_results['T']))

        # Set y-axis limits starting from 0
        ax[0].set_ylim(0, max(spectrum_results['Sa']) * 1.1)  # Add a margin of 10%
        ax[1].set_ylim(0, max(spectrum_results['Sv']) * 1.1)
        ax[2].set_ylim(0, max(spectrum_results['Sd']) * 1.1)
        
        for a in ax:
            a.legend()
            a.grid(True, linestyle='--', linewidth=0.5)
            a.tick_params(labelbottom=True)  # Ensure x-axis tick labels are shown for all subplots
            
        plt.tight_layout()
        plt.show()
        return ax
   
    def _plot_sdof_response(self, u, udot, uacc, time, ax=None, figsize=(10,6), linewidth=0.75, linestyle='-', color=['k', 'b', 'r']):
        """Plots SDOF responses (displacement, velocity, acceleration) with correct unit labels."""
        
        # Get unit labels
        length_unit = self.units['length']['label']
        time_unit = self.units['time']['label']
        unit_labels = {
            'displacement': f'{length_unit}',
            'velocity': f'{length_unit}/{time_unit}',
            'acceleration': f'{length_unit}/{time_unit}²'
        }
        
        # Define plot configurations
        plot_data = {
            'acceleration': (uacc, 'Aceleración', 'b'),
            'velocity': (udot, 'Velocidad', 'b'),
            'displacement': (u, 'Desplazamiento', 'r'),
        }
        
        
        if ax is None:
            # Create subplots dynamically
            fig, ax = plt.subplots(nrows=3, ncols=1, figsize=figsize, sharex=True)
        
        # Loop through each response and plot it
        for i, (key, (y_data, title, plot_color)) in enumerate(plot_data.items()):
            max_idx = np.argmax(y_data)
            min_idx = np.argmin(y_data)
            unit = unit_labels[key]
            
            ax[i].plot(time, y_data, label=title, color=color[i % len(color)], linewidth=linewidth, linestyle=linestyle)
            ax[i].plot(time[max_idx], y_data[max_idx], 'ro', label=f'Máx: {y_data[max_idx]:.2f} {unit}')
            ax[i].plot(time[min_idx], y_data[min_idx], 'go', label=f'Mín: {y_data[min_idx]:.2f} {unit}')
            ax[i].set_ylabel(f'{title} ({unit})')
            ax[i].set_title(f'{title} vs. Tiempo')
            ax[i].legend()
            ax[i].grid(True)
        
        ax[-1].set_xlabel(f'Tiempo ({time_unit})')  # Set x-axis label for the last plot
        plt.tight_layout()
        plt.show()
        
        return fig, ax

    def plot_earthquake_record(self, ax=None, figsize=(10, 6), linewidth=0.75, linestyle='-', 
                               quantities=['acceleration', 'velocity', 'displacement'], color=['k', 'b', 'r']):
        """Plots selected earthquake records with correct unit labels."""
        
        # Get unit factors and labels
        length_unit = self.units['length']['label']
        time_unit = self.units['time']['label']
        accel_units = self.units['acceleration']['label']

        # Define unit labels for each quantity
        unit_labels = {
            'acceleration': accel_units['label'],
            'velocity': f'{length_unit}/{time_unit}',
            'displacement': f'{length_unit}'
        }

        # Determine which plots to show
        valid_quantities = {'acceleration', 'velocity', 'displacement'}
        selected_quantities = [q for q in quantities if q in valid_quantities]
        num_plots = len(selected_quantities)

        if num_plots == 0:
            print("No valid quantities selected for plotting.")
            return None, None

        # Create subplots dynamically
        if ax is None:
            fig, ax = plt.subplots(nrows=num_plots, ncols=1, figsize=figsize, sharex=True)
            if num_plots == 1:
                ax = [ax]  # Convert single subplot to list
        else:
            fig = ax.figure

        # Extract data
        time = self.record_results['time']
        acceleration = self.record_results['acceleration']
        velocity = self.record_results['velocity']
        displacement = self.record_results['displacement']
        accel_max_index = self.record_results['accel_max_index']
        accel_min_index = self.record_results['accel_min_index']
        velocity_max_index = self.record_results['velocity_max_index']
        velocity_min_index = self.record_results['velocity_min_index']
        displacement_max_index = self.record_results['displacement_max_index']
        displacement_min_index = self.record_results['displacement_min_index']

        # Define plot configurations
        plot_data = {
            'acceleration': (acceleration, accel_max_index, accel_min_index, 'Aceleración', 'blue'),
            'velocity': (velocity, velocity_max_index, velocity_min_index, 'Velocidad', 'green'),
            'displacement': (displacement, displacement_max_index, displacement_min_index, 'Desplazamiento', 'red')
        }

        # Loop through selected quantities and plot each one
        for i, quantity in enumerate(selected_quantities):
            y_data, max_idx, min_idx, title, plot_color = plot_data[quantity]
            unit = unit_labels[quantity]  # Get the correct unit for the y-axis
            
            ax[i].plot(time, y_data, label=title, color=color[i % len(color)], linewidth=linewidth, linestyle=linestyle)
            ax[i].plot(time[max_idx], y_data[max_idx], 'ro', label=f'Máx: {y_data[max_idx]:.2f} {unit}')
            ax[i].plot(time[min_idx], y_data[min_idx], 'go', label=f'Mín: {y_data[min_idx]:.2f} {unit}')
            ax[i].set_ylabel(f'{title} ({unit})')
            ax[i].set_title(f'{title} vs. Tiempo')
            ax[i].legend()
            ax[i].grid(True)

        ax[-1].set_xlabel(f'Tiempo ({time_unit})')  # Set x-axis label for the last plot
        plt.tight_layout()
        plt.show()

        return fig, ax
    


from .Earthquake import Earthquake
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


class Earthquake2D:
    def __init__(self, eq_x, eq_y):
        """
        Initialize with two Earthquake objects: one for X and one for Y direction.
        """
        if eq_x.dt != eq_y.dt:
            raise ValueError("The time step (dt) of both components must be the same.")
        if len(eq_x.record_results['acceleration']) != len(eq_y.record_results['acceleration']):
            raise ValueError("X and Y acceleration arrays must be the same length.")
        
        self.eq_x = eq_x
        self.eq_y = eq_y
        self.dt = eq_x.dt
        self.units = eq_x.units  # assume same units
        self.time = eq_x.record_results['time']

    def rotate(self, theta_deg: float) -> np.ndarray:
        """
        Rotate the 2D ground motion by an angle θ (degrees).
        Returns the rotated acceleration array.
        """
        ax = self.eq_x.record_results['acceleration']
        ay = self.eq_y.record_results['acceleration']
        theta_rad = np.deg2rad(theta_deg)
        return ax * np.cos(theta_rad) + ay * np.sin(theta_rad)

    def plot_xy_acceleration(self, label_x='X-dir', label_y='Y-dir', color=('k', 'r'), linewidth=0.75):
        """
        Plot both X and Y acceleration time histories.
        """
        ax = self.eq_x.record_results['acceleration']
        ay = self.eq_y.record_results['acceleration']
        time = self.time
        unit = self.units['acceleration']['label']
        
        fig, axis = plt.subplots(figsize=(10, 4))
        axis.plot(time, ax, label=label_x, color=color[0], linewidth=linewidth)
        axis.plot(time, ay, label=label_y, color=color[1], linewidth=linewidth)
        
        axis.set_title("Acceleration Time Histories")
        axis.set_xlabel(f"Time ({self.units['time']['label']})")
        axis.set_ylabel(f"Acceleration ({unit})")
        axis.grid(True)
        axis.legend()
        plt.tight_layout()
        plt.show()
        return fig, axis

    def plot_lissajous(self, color='#000077', linewidth=0.75):
        """
        Plot the Lissajous figure (Ay vs Ax).
        """
        ax = self.eq_x.record_results['acceleration']
        ay = self.eq_y.record_results['acceleration']
        unit = self.units['acceleration']['label']
        
        fig, axis = plt.subplots(figsize=(5, 5))
        axis.plot(ax, ay, color=color, linewidth=linewidth, linestyle='-')
        axis.set_title("Lissajous Curve (Ay vs Ax)")
        axis.set_xlabel(f"X Acceleration ({unit})")
        axis.set_ylabel(f"Y Acceleration ({unit})")
        axis.axis('equal')
        axis.grid(True)
        plt.tight_layout()
        plt.show()
        return fig, axis

    def compute_rotd_spectrum(self, T_range: np.ndarray, angles=np.arange(0, 180, 1)) -> dict:
        """
        Compute RotD50 and RotD100 spectra from rotated 2D motions.

        Parameters:
        - T_range: array of periods (s)
        - angles: angles in degrees to rotate through (default: every 1° from 0 to 179)

        Returns:
        - dict with 'T', 'RotD50', 'RotD100', and 'units'
        """
        Sa_all = []

        for theta in angles:
            a_rot = self.rotate(theta)
            eq_rot = self.eq_x.__class__(dt=self.dt, data_array=a_rot, units=self.units)
            spectrum = eq_rot.compute_response_spectrum(T_range)
            Sa_all.append(spectrum['Sa'])

        Sa_matrix = np.stack(Sa_all, axis=0)  # shape: (n_angles, n_T)
        rotd50 = np.percentile(Sa_matrix, 50, axis=0)
        rotd100 = np.max(Sa_matrix, axis=0)

        return {
            'T': T_range,
            'RotD50': rotd50,
            'RotD100': rotd100,
            'units': self.units['acceleration']['label']
        }

    def plot_rotd_spectra(self, rotd_result: dict, figsize=(6, 4), linewidth=1.0):
        """
        Plot RotD50 and RotD100 spectra.
        """
        T = rotd_result['T']
        Sa50 = rotd_result['RotD50']
        Sa100 = rotd_result['RotD100']
        unit = rotd_result['units']

        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(T, Sa50, label='RotD50', linewidth=linewidth)
        ax.plot(T, Sa100, label='RotD100', linewidth=linewidth, linestyle='--')
        ax.set_xlabel("Period (s)")
        ax.set_ylabel(f"Spectral Acceleration ({unit})")
        ax.set_title("RotD50 and RotD100 Spectra")
        ax.grid(True)
        ax.legend()
        plt.tight_layout()
        plt.show()
        return fig, ax    

    def estimate_rotd100_angle(self) -> float:
        """
        Estimate the RotD100 angle (in degrees) from the principal axis of [ax, ay] using PCA.

        Returns:
        --------
        float: estimated angle in degrees ∈ [0, 180)
        """
        ax = self.eq_x.record_results['acceleration']
        ay = self.eq_y.record_results['acceleration']
        
        # Stack and fit PCA
        data = np.vstack((ax, ay)).T  # shape: (N, 2)
        pca = PCA(n_components=2)
        pca.fit(data)
        direction = pca.components_[0]  # first principal axis
        
        # Angle of principal axis
        angle_rad = np.arctan2(direction[1], direction[0])
        angle_deg = np.rad2deg(angle_rad) % 180  # ensure in [0, 180)
        return angle_deg

    def compute_rotmax_spectrum(self, T_range=np.linspace(0.01, 4.0, 100), zeta=0.05, m=1.0, beta=0.25, gamma=0.5, plot=False) -> dict:
        """
        Estimate the strongest shaking direction (RotD100 angle) using PCA
        and compute the response spectrum at that rotation.

        Parameters:
        -----------
        T_range : np.ndarray
            Array of periods (s) for the spectrum.
        Other parameters are passed to the Newmark method.

        Returns:
        --------
        dict: spectrum with 'T', 'Sa', 'Sv', 'Sd', 'units', 'angle'
        """
        ax = self.eq_x.record_results['acceleration']
        ay = self.eq_y.record_results['acceleration']
        
        # Estimate principal direction using PCA
        data = np.vstack((ax, ay)).T
        pca = PCA(n_components=2)
        pca.fit(data)
        principal_axis = pca.components_[0]
        
        angle_rad = np.arctan2(principal_axis[1], principal_axis[0])
        angle_deg = np.rad2deg(angle_rad) % 180
        
        # Rotate the motion to that angle
        a_rot = self.rotate(angle_deg)
        eq_rot = Earthquake(dt=self.dt, data_array=a_rot, units=self.units)
        spectrum = eq_rot.compute_response_spectrum(T_range, zeta, m, beta, gamma, plot=plot)
        
        # Attach angle info
        spectrum['angle'] = angle_deg
        
        if plot:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(spectrum['T'], spectrum['Sa'], label=f'RotMax (θ={angle_deg:.1f}°)', color='darkorange')
            ax.set_xlabel("Period (s)")
            ax.set_ylabel(f"Spectral Acceleration ({spectrum['units']['Sa']})")
            ax.set_title("Response Spectrum at Estimated RotD100 Direction")
            ax.grid(True)
            ax.legend()
            plt.tight_layout()
            plt.show()

        return spectrum

    def analyze_pca(self, scale=4.0, plot_components=True):
        """
        Analyze the PCA of the 2D ground motion and plot:
        - Lissajous curve with PCA directions
        - Time series in principal axes
        - Explained variance (energy) distribution
        - Orthogonality check

        Parameters:
        -----------
        scale : float
            Scaling factor for visualizing PCA axes
        plot_components : bool
            Whether to plot the time history projections onto PCA axes
        """
        ax_data = self.eq_x.record_results['acceleration']
        ay_data = self.eq_y.record_results['acceleration']
        time = self.time
        unit = self.units['acceleration']['label']

        # Perform PCA
        data = np.vstack((ax_data, ay_data)).T
        pca = PCA(n_components=2)
        pca.fit(data)
        transformed = pca.transform(data)  # projection to PCA axes

        v1, v2 = pca.components_
        dot = np.dot(v1, v2)
        angle_rad = np.arctan2(v1[1], v1[0])
        angle_deg = np.rad2deg(angle_rad) % 180

        print(f"🔎 Estimated RotD100 angle: {angle_deg:.2f}°")
        print(f"🟰 Orthogonality check (v1 ⋅ v2): {dot:.2e} (should be ~0)")
        print(f"📊 Variance explained: PC1 = {pca.explained_variance_ratio_[0]*100:.1f}%, PC2 = {pca.explained_variance_ratio_[1]*100:.1f}%")

        # === Plot 1: Lissajous + PCA vectors
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(ax_data, ay_data, label='Original motion', color='gray', alpha=0.4)

        # PCA vectors scaled by sqrt(variance)
        for i, vec in enumerate(pca.components_):
            var = np.sqrt(pca.explained_variance_[i])
            color = 'r' if i == 0 else 'b'
            ax.arrow(0, 0, scale*var*vec[0], scale*var*vec[1], 
                    width=0.001, head_width=0.02, color=color, label=f"PC{i+1}")
        
        ax.set_aspect('equal')
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.set_xlabel(f"X Accel ({unit})")
        ax.set_ylabel(f"Y Accel ({unit})")
        ax.set_title("Lissajous Curve with PCA Axes")
        ax.legend()
        plt.tight_layout()
        plt.show()

        # === Plot 2: Time Series in PCA directions
        if plot_components:
            p1, p2 = transformed[:, 0], transformed[:, 1]
            plt.figure(figsize=(10, 4))
            plt.plot(time, p1, label='PC1 (main)', color='r')
            plt.plot(time, p2, label='PC2 (orthogonal)', color='b')
            plt.xlabel(f"Time ({self.units['time']['label']})")
            plt.ylabel(f"Acceleration ({unit})")
            plt.title("Acceleration in PCA-Aligned Directions")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.show()

        # === Plot 3: Explained Variance
        plt.figure(figsize=(4, 4))
        plt.bar(['PC1', 'PC2'], pca.explained_variance_ratio_ * 100, color=['r', 'b'])
        plt.ylabel("Explained Variance (%)")
        plt.title("Energy Distribution in PCA Axes")
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.show()




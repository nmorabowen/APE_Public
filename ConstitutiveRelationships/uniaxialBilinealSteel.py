# ===============================================
# Importamos dependencias
# ===============================================

import numpy as np
import matplotlib.pyplot as plt

from plotApeConfig import set_default_plot_params, blueAPE
set_default_plot_params()

# ===============================================
# Class Definition
# ===============================================

class uniaxialBilinealSteel:
    def __init__(self, name, fy, fsu, esh=0.008, esu=0.012, Es=200000, Esh=7000, plot=False, delta=15, color='k', marker=None):
        """
        Initialize the UniaxialAceroBilineal class.

        Parameters:
        name (str): Name of the steel type.
        fy (float): Yield strength.
        fsu (float): Ultimate strength.
        esh (float): Strain at the start of hardening (default 0.02).
        esu (float): Ultimate strain (default 0.12).
        Es (float): Young's modulus (default 200000).
        Esh (float): Hardening modulus (default 7000).
        plot (bool): Whether to plot the constitutive law (default False).
        delta (int): Number of points for interpolation (default 15).
        color (str): Color of the plot (default 'k').
        """
        try:
            # Instance values
            self.name = str(name)
            self.fy = float(fy)
            self.fsu = float(fsu)
            self.esh = float(esh)
            self.esu = float(esu)
            self.Es = float(Es)
            self.Esh = float(Esh)
            self.color = str(color)
            self.marker=marker

            # Input validation
            if self.fy <= 0 or self.fsu <= 0 or self.Es <= 0 or self.Esh <= 0:
                raise ValueError("Strength and modulus values must be positive.")
            if not (0 < self.esh < self.esu):
                raise ValueError("Strain at the start of hardening (esh) must be less than ultimate strain (esu).")
            if delta <= 0:
                raise ValueError("Delta must be a positive integer.")

            # Calculated values
            self.ey = self.fy / self.Es

            _, _, self.constitutiveLaw = self.relacionesConstitutivas(delta)
            
            if plot:
                self.plot()
        
        except Exception as e:
            print(f"Error initializing UniaxialAceroBilineal: {e}")

    def __str__(self):
        return f'{self.name}'
    
    def __repr__(self) -> str:
        return f'{self.name}'
    
    def relacionesConstitutivas(self, delta):
        """
        Calculate the constitutive relations.

        Parameters:
        delta (int): Number of points for interpolation.

        Returns:
        tuple: Strain array, stress array, and constitutive law matrix.
        """
        try:
            fs_array = np.array([0, self.fy, self.fy])
            es_array = np.array([0, self.ey, self.esh])
            
            es_sh_array = np.linspace(self.esh, self.esu, delta)
            P = self.Esh * ((self.esu - self.esh) / (self.fsu - self.fy))
            
            calculo_fs = lambda fsu, fy, esu, es, esh, P: fsu + (fy - fsu) * ((esu - es) / (esu - esh)) ** P
            fs_sh_array = calculo_fs(self.fsu, self.fy, self.esu, es_sh_array, self.esh, P)
            
            es_array = np.append(es_array, es_sh_array[1:])
            fs_array = np.append(fs_array, fs_sh_array[1:])
            
            constitutiveLaw = np.vstack((es_array, fs_array))
            
            return es_array, fs_array, constitutiveLaw
        
        except Exception as e:
            print(f"Error calculating constitutive relations: {e}")
            return np.array([]), np.array([]), np.array([[], []])

    def plot(self, ax=None):
        """
        Plot the constitutive law with LaTeX labels, customized grid lines, and an optional label.

        Parameters:
        ax (matplotlib.axes.Axes): Existing axes to plot on. If None, a new figure and axes are created.
        label (str): Label for the plot line (default None).
        """
        try:
            if ax is None:
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.set_title('Constitutive Law of Uniaxial Bilinear Steel')
                
            figureLabel = f'Acero Uniaxial: {self.name}'
            ax.plot(self.constitutiveLaw[0], self.constitutiveLaw[1], color=self.color, linewidth=1.50, label=figureLabel, marker=self.marker)
            ax.legend()
            ax.set_xlabel('Strain')
            ax.set_ylabel('Stress')
            
            if ax is None:
                plt.show()
        
        except Exception as e:
            print(f"Error plotting constitutive law: {e}")
            
            
if __name__ == "__main__":
    from baseUnits import ksi
    A36=uniaxialBilinealSteel('A36',36*ksi, 1.5*36*ksi, plot=True, color=blueAPE)
    A572=uniaxialBilinealSteel('A572',50*ksi, 1.25*50*ksi, plot=True, color='k')
    
    plt.show()
    
    fig, ax = plt.subplots(figsize=(10, 5))
    A36.plot(ax=ax)
    A572.plot(ax=ax)
    plt.show()
    
    
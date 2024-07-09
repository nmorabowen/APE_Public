# ===============================================
# Import Dependencies
# ===============================================

import numpy as np
import matplotlib.pylab as plt
from baseUnits import MPa
from plotApeConfig import set_default_plot_params, blueAPE

# Set default plotting parameters for consistency
set_default_plot_params()

# ===============================================
# Class Definition
# ===============================================

class uniaxialUnconfinedConcrete:
    """
    Class to model the behavior of unconfined concrete under uniaxial stress conditions.

    Attributes:
        name (str): Name of the concrete instance.
        fco (float): Compressive strength of the concrete.
        eco (float): Strain corresponding to the compressive strength.
        ec_sprall (float): Maximum strain before failure.
        color (str): Color for plotting the constitutive law.
        ecc (float): Maximum strain at the compressive strength.
        Esec (float): Secant modulus.
        Ec (float): Elastic modulus of the concrete.
        r (float): Ratio between the elastic modulus and the secant modulus.
        constitutiveLaw (np.ndarray): Calculated constitutive law of the concrete.
    """
    
    def __init__(self, name, fco, eco=0.002, ec_sprall=0.006, delta=15, plot=False, color='k', marker=None):
        """
        Initializes the uniaxialUnconfinedConcrete class instance.

        Parameters:
            name (str): Name of the concrete instance.
            fco (float): Compressive strength of the concrete.
            eco (float): Strain corresponding to the compressive strength. Default is 0.002.
            ec_sprall (float): Maximum strain before failure. Default is 0.006.
            delta (int): Number of data points for strain array. Default is 15.
            plot (bool): Whether to plot the constitutive law upon initialization. Default is False.
            color (str): Color for plotting. Default is 'k' (black).
        """
        self.name = name
        self.fco = fco
        self.eco = eco
        self.ec_sprall = ec_sprall
        self.color = color
        self.marker=marker

        try:
            # Calculated values
            self.ecc = self.eco
            self.Esec = fco / eco
            self.Ec = (5000 * (fco / MPa)**0.5) * MPa
            self.r = self.Ec / (self.Ec - self.Esec)

            _, _, self.constitutiveLaw = self.relacionesConstitutivas(delta, plot)

            if plot:
                self.plot()
        except ZeroDivisionError:
            print("Error: Division by zero encountered during initialization.")
        except Exception as e:
            print(f"Error during initialization: {e}")
        
    def __str__(self):
        return f'{self.name}'
    
    def __repr__(self) -> str:
        return f'{self.name}'
    
    def relacionesConstitutivas(self, delta, plot):
        """
        Calculate the constitutive relationship of the concrete.

        Parameters:
            delta (int): Number of data points for strain array.
            plot (bool): Whether to plot the constitutive law.

        Returns:
            es_array (np.ndarray): Strain array.
            fs_array (np.ndarray): Stress array.
            constitutiveLaw (np.ndarray): Constitutive law array.
        """
        try:
            es_array = np.linspace(0, self.ec_sprall, delta)
            
            def calculo_fs(es, eco, ec_sprall, r, fco):
                """
                Calculate the stress based on the given strain.

                Parameters:
                    es (float): Strain value.
                    eco (float): Strain corresponding to the compressive strength.
                    ec_sprall (float): Maximum strain before failure.
                    r (float): Ratio between the elastic modulus and the secant modulus.
                    fco (float): Compressive strength of the concrete.

                Returns:
                    fs (float): Calculated stress.
                """
                try:
                    x = es / eco
                    if es <= 2 * eco:
                        fs = (fco * x * r) / (r - 1 + x**r)
                    else:
                        x_eco = 2 * eco / eco
                        fc_eco = (fco * x_eco * r) / (r - 1 + x_eco**r)
                        m = (fc_eco - 0) / (2 * eco - ec_sprall)
                        fs = m * (es - ec_sprall)
                except ZeroDivisionError:
                    print("Error: Division by zero encountered in stress calculation.")
                    fs = 0
                except Exception as e:
                    print(f"Error in stress calculation: {e}")
                    fs = 0
                    
                return fs
            
            fs_array = np.zeros(len(es_array))
            for i, es in enumerate(es_array):
                fs_array[i] = calculo_fs(es, self.eco, self.ec_sprall, self.r, self.fco)  
            
            constitutiveLaw = np.vstack((es_array, fs_array))        
            
            return es_array, fs_array, constitutiveLaw
        except Exception as e:
            print(f"Error in calculating constitutive relationships: {e}")
            return np.array([]), np.array([]), np.array([[], []])
    
    def plot(self, ax=None):
        """
        Plot the constitutive law with LaTeX labels, customized grid lines, and an optional label.

        Parameters:
            ax (matplotlib.axes.Axes): Existing axes to plot on. If None, a new figure and axes are created.
        """
        try:
            if ax is None:
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.set_title('Unconfined Concrete - Mandel')
                
            figureLabel = f'Unconfined Concrete: {self.name}'
            ax.plot(self.constitutiveLaw[0], self.constitutiveLaw[1], marker=self.marker, color=self.color, linewidth=1.50, label=figureLabel)
            ax.legend()
            ax.set_xlabel('Strain')
            ax.set_ylabel('Stress')
            
            if ax is None:
                plt.show()
        
        except Exception as e:
            print(f"Error plotting constitutive law: {e}")
            
if __name__ == "__main__":
    from baseUnits import kgf, cm

    try:
        # Instantiate and plot different concrete instances
        fc210 = uniaxialUnconfinedConcrete('fc210', 210 * kgf / cm**2, plot=True)
        plt.show()
        
        fc210 = uniaxialUnconfinedConcrete('fc210', 210 * kgf / cm**2)
        fc240 = uniaxialUnconfinedConcrete('fc240', 240 * kgf / cm**2, color=blueAPE)
        fc280 = uniaxialUnconfinedConcrete('fc280', 280 * kgf / cm**2, color='red')
        fc350 = uniaxialUnconfinedConcrete('fc350', 350 * kgf / cm**2, color='grey')
        fc400 = uniaxialUnconfinedConcrete('fc400', 400 * kgf / cm**2, color='darkolivegreen')
        
        fig, ax = plt.subplots(figsize=(10, 5))
        fc210.plot(ax=ax)
        fc240.plot(ax=ax)
        fc280.plot(ax=ax)
        fc350.plot(ax=ax)
        fc400.plot(ax=ax)
        
        plt.show()
    except Exception as e:
        print(f"Error in main execution: {e}")

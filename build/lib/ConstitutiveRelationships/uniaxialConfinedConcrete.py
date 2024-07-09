# ===============================================
# Import Dependencies
# ===============================================

import numpy as np
import matplotlib.pyplot as plt
from math import pi
from plotApeConfig import set_default_plot_params, blueAPE
set_default_plot_params()
from baseUnits import MPa

# ===============================================
# Class Definition
# ===============================================

class uniaxialConfinedConcrete:
    """
    Class to model the behavior of confined concrete under uniaxial stress conditions.

    Attributes:
        name (str): Name of the concrete instance.
        fco (float): Compressive strength of the concrete.
        eco (float): Strain corresponding to the compressive strength.
        b (float): Width of the concrete section.
        h (float): Height of the concrete section.
        rec (float): Concrete cover.
        num_var_b (int): Number of longitudinal bars in width.
        num_var_h (int): Number of longitudinal bars in height.
        phi_longitudinal (float): Diameter of longitudinal bars.
        num_est_perpendicular_b (int): Number of stirrups perpendicular to width.
        num_est_perpendicular_h (int): Number of stirrups perpendicular to height.
        phi_estribo (float): Diameter of stirrups.
        s (float): Spacing of stirrups.
        fye (float): Yield strength of stirrups.
        esu_estribo (float): Ultimate strain of stirrups.
        color (str): Color for plotting the constitutive law.
    """
    
    def __init__(self, name, fco, eco, b, h, rec, num_var_b, num_var_h, phi_longitudinal, num_est_perpendicular_b, num_est_perpendicular_h, phi_estribo, s, fye, esu_estribo, aprox=False, delta=50, plot=False, color='k', marker=None):
        """
        Initializes the uniaxialConfinedConcrete class instance.

        Parameters:
            name (str): Name of the concrete instance.
            fco (float): Compressive strength of the concrete.
            eco (float): Strain corresponding to the compressive strength.
            b (float): Width of the concrete section.
            h (float): Height of the concrete section.
            rec (float): Concrete cover.
            num_var_b (int): Number of longitudinal bars in width.
            num_var_h (int): Number of longitudinal bars in height.
            phi_longitudinal (float): Diameter of longitudinal bars.
            num_est_perpendicular_b (int): Number of stirrups perpendicular to width.
            num_est_perpendicular_h (int): Number of stirrups perpendicular to height.
            phi_estribo (float): Diameter of stirrups.
            s (float): Spacing of stirrups.
            fye (float): Yield strength of stirrups.
            esu_estribo (float): Ultimate strain of stirrups.
            aprox (bool): Use approximate method for compression resistance calculation. Default is False.
            delta (int): Number of data points for strain array. Default is 50.
            plot (bool): Whether to plot the constitutive law upon initialization. Default is False.
            color (str): Color for plotting. Default is 'k' (black).
        """
        self.name = name
        self.fco = fco
        self.eco = eco
        self.b = b
        self.h = h
        self.rec = rec
        self.num_var_b = num_var_b
        self.num_var_h = num_var_h
        self.phi_longitudinal = phi_longitudinal
        self.num_est_perpendicular_b = num_est_perpendicular_b
        self.num_est_perpendicular_h = num_est_perpendicular_h
        self.phi_estribo = phi_estribo
        self.s = s
        self.fye = fye
        self.esu_estribo = esu_estribo
        self.color = color
        self.marker=marker
        
        try:
            # Calculated values
            self.bc = self.b - 2 * self.rec - self.phi_estribo
            self.hc = self.h - 2 * self.rec - self.phi_estribo
            self.Ac = self.bc * self.hc
            self.num_var_long = self.num_var_b * 2 + (self.num_var_h - 2) * 2
            self.As = self.num_var_long * pi * self.phi_longitudinal**2 / 4
            self.rho_confinado = self.As / self.Ac
            self.Ec = (5000 * (fco / MPa)**0.5) * MPa

            self.rho_estribo_perp_b, self.rho_estribo_perp_h, self.ke, self.fl_perpendicular_b_efectivo, self.fl_perpendicular_h_efectivo = self.calculosConfinamiento()
            self.fcc_ratio, self.fcc, self.ecc_ratio, self.ecc = self.calculosResistenciasCompresion(aprox)
            self.ecu, self.ecu_ratio = self.calculoUltimateStrain()

            self.Esec = self.fcc / self.ecc
            self.r = self.Ec / (self.Ec - self.Esec)

            _, _, self.constitutiveLaw = self.relacionesConstitutivas(delta, plot)

            if plot:
                self.plot()
        except ZeroDivisionError:
            print("Error: Division by zero encountered during initialization.")
        except Exception as e:
            print(f"Error during initialization: {e}")
        
    def __str__(self) -> str:
        return f'{self.name}'
    
    def __repr__(self) -> str:
        return f'{self.name}'
        
    def calculosConfinamiento(self):
        """
        Calculate confinement parameters for the concrete.

        Returns:
            rho_estribo_perp_b (float): Transverse reinforcement ratio perpendicular to width.
            rho_estribo_perp_h (float): Transverse reinforcement ratio perpendicular to height.
            ke (float): Effectiveness coefficient.
            fl_perpendicular_b_efectivo (float): Effective transverse pressure perpendicular to width.
            fl_perpendicular_h_efectivo (float): Effective transverse pressure perpendicular to height.
        """
        try:
            num_w_en_b = self.num_var_b - 1
            num_w_en_h = self.num_var_h - 1
            w_libre_b = (self.b - 2 * self.rec - 2 * self.phi_estribo - self.phi_longitudinal * self.num_var_b) / num_w_en_b
            w_libre_h = (self.h - 2 * self.rec - 2 * self.phi_estribo - self.phi_longitudinal * self.num_var_h) / num_w_en_h

            Ai = 2 * num_w_en_b * (w_libre_b**2 / 6) + 2 * num_w_en_h * (w_libre_h**2 / 6)
            Acc = self.Ac * (1 - self.rho_confinado)
            s_libre = self.s - self.phi_estribo
            Ae = (self.Ac - Ai) * (1 - (s_libre) / (2 * self.bc)) * (1 - (s_libre) / (2 * self.hc))
            ke = Ae / Acc

            As_estribo_perp_b = self.num_est_perpendicular_b * pi * self.phi_estribo**2 / 4
            As_estribo_perp_h = self.num_est_perpendicular_h * pi * self.phi_estribo**2 / 4

            rho_estribo_perp_b = As_estribo_perp_b / (self.s * self.bc)
            rho_estribo_perp_h = As_estribo_perp_h / (self.s * self.hc)

            fl_perpendicular_b = rho_estribo_perp_b * self.fye
            fl_perpendicular_h = rho_estribo_perp_h * self.fye

            fl_perpendicular_b_efectivo = ke * fl_perpendicular_b
            fl_perpendicular_h_efectivo = ke * fl_perpendicular_h

            return rho_estribo_perp_b, rho_estribo_perp_h, ke, fl_perpendicular_b_efectivo, fl_perpendicular_h_efectivo
        except Exception as e:
            print(f"Error in confinement calculations: {e}")
            return 0, 0, 0, 0, 0
            
    def calculosResistenciasCompresion(self, aprox=False):
        """
        Calculate compression resistances for the concrete.

        Parameters:
            aprox (bool): Use approximate method for compression resistance calculation. Default is False.

        Returns:
            fcc_ratio (float): Ratio of confined to unconfined compressive strength.
            fcc (float): Confined compressive strength.
            ecc_ratio (float): Ratio of confined to unconfined strain.
            ecc (float): Confined strain.
        """
        try:
            fl1_ratio = self.fl_perpendicular_b_efectivo / self.fco
            fl2_ratio = self.fl_perpendicular_h_efectivo / self.fco
            rho_transversal = (self.rho_estribo_perp_b + self.rho_estribo_perp_h) * 0.50
            fpl = rho_transversal * self.fye * self.ke
            fcc_ratio_apox = (-1.254 + 2.254 * (1 + ((7.94 * fpl) / (self.fco)))**0.5 - 2 * (fpl / self.fco))
            
            def calculoRatiofcc(fl1_ratio, fl2_ratio):
                """
                Calculate the ratio of confined to unconfined compressive strength based on tabulated values.

                Parameters:
                    fl1_ratio (float): Ratio of effective transverse pressure to unconfined compressive strength (width).
                    fl2_ratio (float): Ratio of effective transverse pressure to unconfined compressive strength (height).

                Returns:
                    fcc_ratio (float): Ratio of confined to unconfined compressive strength.
                """
                Table = np.array([[1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00],
                                  [1.05, 1.13, 1.13, 1.13, 1.13, 1.13, 1.13, 1.13, 1.13, 1.13, 1.13, 1.13, 1.13, 1.13, 1.13, 1.13],
                                  [1.09, 1.19, 1.26, 1.26, 1.26, 1.26, 1.26, 1.26, 1.26, 1.26, 1.26, 1.26, 1.26, 1.26, 1.26, 1.26],
                                  [1.13, 1.23, 1.31, 1.37, 1.37, 1.37, 1.37, 1.37, 1.37, 1.37, 1.37, 1.37, 1.37, 1.37, 1.37, 1.37],
                                  [1.15, 1.26, 1.34, 1.41, 1.47, 1.47, 1.47, 1.47, 1.47, 1.47, 1.47, 1.47, 1.47, 1.47, 1.47, 1.47],
                                  [1.18, 1.28, 1.38, 1.45, 1.51, 1.58, 1.58, 1.58, 1.58, 1.58, 1.58, 1.58, 1.58, 1.58, 1.58, 1.58],
                                  [1.20, 1.30, 1.40, 1.48, 1.55, 1.62, 1.67, 1.67, 1.67, 1.67, 1.67, 1.67, 1.67, 1.67, 1.67, 1.67],
                                  [1.22, 1.32, 1.43, 1.51, 1.58, 1.65, 1.71, 1.76, 1.76, 1.76, 1.76, 1.76, 1.76, 1.76, 1.76, 1.76],
                                  [1.23, 1.35, 1.45, 1.53, 1.62, 1.68, 1.74, 1.81, 1.85, 1.85, 1.85, 1.85, 1.85, 1.85, 1.85, 1.85],
                                  [1.25, 1.36, 1.47, 1.55, 1.64, 1.71, 1.77, 1.84, 1.88, 1.93, 1.93, 1.93, 1.93, 1.93, 1.93, 1.93],
                                  [1.26, 1.38, 1.48, 1.57, 1.66, 1.73, 1.80, 1.86, 1.92, 1.96, 2.00, 2.00, 2.00, 2.00, 2.00, 2.00],
                                  [1.27, 1.39, 1.50, 1.59, 1.68, 1.75, 1.83, 1.88, 1.94, 1.98, 2.02, 2.07, 2.07, 2.07, 2.07, 2.07],
                                  [1.28, 1.40, 1.52, 1.61, 1.70, 1.77, 1.85, 1.90, 1.96, 2.01, 2.05, 2.09, 2.14, 2.14, 2.14, 2.14],
                                  [1.29, 1.41, 1.52, 1.63, 1.71, 1.80, 1.87, 1.92, 1.99, 2.04, 2.08, 2.13, 2.16, 2.20, 2.20, 2.20],
                                  [1.30, 1.42, 1.54, 1.64, 1.73, 1.81, 1.88, 1.95, 2.01, 2.06, 2.11, 2.16, 2.19, 2.23, 2.26, 2.26],
                                  [1.30, 1.43, 1.55, 1.65, 1.74, 1.83, 1.89, 1.96, 2.03, 2.08, 2.13, 2.18, 2.22, 2.25, 2.29, 2.30]])

                header = np.array([0, 0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2, 0.22, 0.24, 0.26, 0.28, 0.3])
                target_value = fl2_ratio
                abs_diff = np.abs(header - target_value)
                sorted_indices = np.argsort(abs_diff)
                x_range = np.array([header[sorted_indices[0]], header[sorted_indices[1]]])
                range_inf = Table[:, sorted_indices[0]]
                range_sup = Table[:, sorted_indices[1]]
                range_values = np.zeros(len(range_inf))
                
                for i, (inf, sup) in enumerate(zip(range_inf, range_sup)):
                    value = np.interp(fl2_ratio, x_range, np.array([inf, sup]))
                    range_values[i] = value
                
                fcc_ratio = np.interp(fl1_ratio, header, range_values)
                
                return fcc_ratio
            
            fcc_ratio_interpolado = calculoRatiofcc(fl1_ratio, fl2_ratio)
            
            if aprox is False:
                fcc_ratio = fcc_ratio_interpolado
            else:
                fcc_ratio = fcc_ratio_apox
            
            fcc = self.fco * fcc_ratio
            ecc_ratio = 1 + 5 * (fcc / self.fco - 1)
            ecc = ecc_ratio * self.eco
            
            return fcc_ratio, fcc, ecc_ratio, ecc
        except Exception as e:
            print(f"Error in compression resistance calculations: {e}")
            return 0, 0, 0, 0
    
    def calculoUltimateStrain(self):
        """
        Calculate the ultimate strain for the concrete.

        Returns:
            ecu (float): Ultimate strain.
            ecu_ratio (float): Ratio of ultimate strain to unconfined strain.
        """
        try:
            ecu = 1.50 * (0.004 + 1.40 * ((self.rho_estribo_perp_b + self.rho_estribo_perp_h) * self.fye * self.esu_estribo) / (self.fcc))
            ecu_ratio = ecu / self.eco
            return ecu, ecu_ratio
        except Exception as e:
            print(f"Error in ultimate strain calculation: {e}")
            return 0, 0
    
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
            
            es_linspace = np.linspace(0, self.eco, delta // 4)
            es_geomspace = np.geomspace(self.eco, self.ecu, 3* delta // 4)
            es_array = np.unique(np.concatenate((es_linspace, es_geomspace)))
            
            def calculo_fs(es, ecc, r, fcc):
                """
                Calculate the stress based on the given strain.

                Parameters:
                    es (float): Strain value.
                    ecc (float): Confined strain.
                    r (float): Ratio between elastic and secant modulus.
                    fcc (float): Confined compressive strength.

                Returns:
                    fs (float): Calculated stress.
                """
                try:
                    x = es / ecc
                    fs = (fcc * x * r) / (r - 1 + x**r)
                    return fs
                except ZeroDivisionError:
                    print("Error: Division by zero encountered in stress calculation.")
                    return 0
                except Exception as e:
                    print(f"Error in stress calculation: {e}")
                    return 0
            
            fs_array = np.zeros(len(es_array))
            for i, es in enumerate(es_array):
                fs_array[i] = calculo_fs(es, self.ecc, self.r, self.fcc)  
            
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
                ax.set_title('Constitutive Law of Confined Concrete')
                
            figureLabel = f'Confined Concrete: {self.name}'
            ax.plot(self.constitutiveLaw[0], self.constitutiveLaw[1], marker=self.marker, color=self.color, linewidth=1.50, label=figureLabel)
            ax.legend()
            ax.set_xlabel('Strain')
            ax.set_ylabel('Stress')
            
            if ax is None:
                plt.show()
        
        except Exception as e:
            print(f"Error plotting constitutive law: {e}")

if __name__ == "__main__":
    from baseUnits import kgf, cm, mm
    
    try:
        # Instantiate and plot the concrete instance
        fc210c = uniaxialConfinedConcrete('fc210c',
                                          fco=210 * kgf / cm**2,
                                          eco=0.003,
                                          b=30 * cm,
                                          h=40 * cm,
                                          rec=3 * cm,
                                          num_var_b=3,
                                          num_var_h=4,
                                          phi_longitudinal=16 * mm,
                                          num_est_perpendicular_b=3,
                                          num_est_perpendicular_h=4,
                                          phi_estribo=10 * mm,
                                          esu_estribo=0.12,
                                          s=10 * cm,
                                          fye=4200 * kgf / cm**2,
                                          plot=True,
                                          color=blueAPE)
        plt.show()
    except Exception as e:
        print(f"Error in main execution: {e}")

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
    
    def __init__(self, name, fco, eco, b, h, rec, num_var_b, num_var_h, phi_longitudinal, num_est_perpendicular_b, num_est_perpendicular_h, phi_estribo, s, fye, esu_estribo, delta=50, plot=False, color='k', marker=None):
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
        self.result={}
        
        try:
            # Calculated values
            self.bc = self.b - 2 * self.rec - self.phi_estribo
            self.hc = self.h - 2 * self.rec - self.phi_estribo
            self.Ac = self.bc * self.hc
            self.num_var_long = self.num_var_b * 2 + (self.num_var_h - 2) * 2
            self.As = self.num_var_long * pi * self.phi_longitudinal**2 / 4
            self.rho_confinado = self.As / self.Ac
            self.Ec = (5000 * (fco / MPa)**0.5) * MPa
            
            self.addResult('Ac', self.Ac)
            self.addResult('As', self.As)
            
            self._setTable()

            self.rho_estribo_perp_b, self.rho_estribo_perp_h, self.ke, self.fl_perpendicular_b_efectivo, self.fl_perpendicular_h_efectivo = self.calculosConfinamiento()
            self.fcc_ratio, self.fcc, self.ecc_ratio, self.ecc = self.calculosResistenciasCompresion()
            self.ecu, self.ecu_ratio = self.calculoUltimateStrain()

            self.Esec = self.fcc / self.ecc
            self.r = self.Ec / (self.Ec - self.Esec)

            self.addResult('r', self.r)
            
            _, _, self.constitutiveLaw = self.relacionesConstitutivas(delta)
            
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

    def addResult(self, key, value):
        self.result[key]=value
        
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

            self.addResult('wi # in b', num_w_en_b)
            self.addResult('wi # in h', num_w_en_h)
            self.addResult('wi Spacing in b', w_libre_b)
            self.addResult('wi Spacing in h', w_libre_h)
            self.addResult('Ai', Ai)
            self.addResult('Acc', Acc)
            self.addResult('Acc', s_libre)
            self.addResult('Ae', Ae)
            self.addResult('As Perpendicular to b', As_estribo_perp_b)
            self.addResult('As Perpendicular to h', As_estribo_perp_h)
            self.addResult('Rho Perpendicular to b', rho_estribo_perp_b)
            self.addResult('Rho Perpendicular to h', rho_estribo_perp_h)
            self.addResult('Lateral stress perpendicular to b', fl_perpendicular_b)
            self.addResult('Lateral stress perpendicular to h', fl_perpendicular_h)
            self.addResult('Effective lateral stress perpendicular to b', fl_perpendicular_b_efectivo)
            self.addResult('Effective lateral stress perpendicular to h', fl_perpendicular_h_efectivo)
            
            return rho_estribo_perp_b, rho_estribo_perp_h, ke, fl_perpendicular_b_efectivo, fl_perpendicular_h_efectivo
        
        except Exception as e:
            print(f"Error in confinement calculations: {e}")
            return 0, 0, 0, 0, 0
    
    def _setTable(self):
        # Function to set the multiaxial strenght values
        self.Table = np.array([
                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1.062750334, 1.14552737, 1.14552737, 1.14552737, 1.14552737, 1.14552737, 1.14552737, 1.14552737, 1.14552737, 1.14552737, 1.14552737, 1.14552737, 1.14552737, 1.14552737, 1.14552737, 1.14552737],
                            [1.102803738, 1.196261682, 1.26835781, 1.26835781, 1.26835781, 1.26835781, 1.26835781, 1.26835781, 1.263017356, 1.263017356, 1.263017356, 1.263017356, 1.263017356, 1.263017356, 1.263017356, 1.263017356],
                            [1.134846462, 1.23364486, 1.316421896, 1.364485981, 1.364485981, 1.364485981, 1.364485981, 1.364485981, 1.367156208, 1.367156208, 1.367156208, 1.367156208, 1.367156208, 1.367156208, 1.367156208, 1.367156208],
                            [1.164218959, 1.263017356, 1.351134846, 1.417890521, 1.465954606, 1.465954606, 1.465954606, 1.465954606, 1.473965287, 1.473965287, 1.473965287, 1.473965287, 1.473965287, 1.473965287, 1.473965287, 1.473965287],
                            [1.185580774, 1.284379172, 1.375166889, 1.452603471, 1.516688919, 1.567423231, 1.567423231, 1.567423231, 1.567423231, 1.567423231, 1.567423231, 1.567423231, 1.567423231, 1.567423231, 1.567423231, 1.567423231],
                            [1.204272363, 1.308411215, 1.407209613, 1.479305741, 1.551401869, 1.612817089, 1.660881175, 1.660881175, 1.660881175, 1.660881175, 1.660881175, 1.660881175, 1.660881175, 1.660881175, 1.660881175, 1.660881175],
                            [1.217623498, 1.329773031, 1.428571429, 1.508678238, 1.580774366, 1.64753004, 1.703604806, 1.748998665, 1.746328438, 1.746328438, 1.746328438, 1.746328438, 1.746328438, 1.746328438, 1.746328438, 1.746328438],
                            [1.238985314, 1.348464619, 1.449933244, 1.530040053, 1.610146862, 1.676902537, 1.738317757, 1.791722296, 1.834445928, 1.834445928, 1.834445928, 1.834445928, 1.834445928, 1.834445928, 1.834445928, 1.834445928],
                            [1.252336449, 1.367156208, 1.468624833, 1.551401869, 1.634178905, 1.703604806, 1.765020027, 1.818424566, 1.869158879, 1.909212283, 1.909212283, 1.909212283, 1.909212283, 1.909212283, 1.909212283, 1.909212283],
                            [1.265687583, 1.377837116, 1.481975968, 1.570093458, 1.655540721, 1.727636849, 1.794392523, 1.845126836, 1.898531375, 1.935914553, 1.970627503, 1.970627503, 1.970627503, 1.970627503, 1.970627503, 1.970627503],
                            [1.273698264, 1.393858478, 1.503337784, 1.58611482, 1.67423231, 1.748998665, 1.818424566, 1.866488652, 1.925233645, 1.962616822, 2.005340454, 2.040053405, 2.040053405, 2.040053405, 2.040053405, 2.040053405],
                            [1.281708945, 1.407209613, 1.514018692, 1.604806409, 1.690253672, 1.767690254, 1.837116155, 1.88518024, 1.946595461, 1.989319092, 2.032042724, 2.072096128, 2.109479306, 2.109479306, 2.109479306, 2.109479306],
                            [1.292389853, 1.417890521, 1.530040053, 1.62082777, 1.706275033, 1.783711615, 1.855807744, 1.909212283, 1.970627503, 2.013351135, 2.06141522, 2.106809079, 2.136181575, 2.173564753, 2.173564753, 2.173564753],
                            [1.300400534, 1.425901202, 1.540720961, 1.631508678, 1.719626168, 1.799732977, 1.871829105, 1.927903872, 1.989319092, 2.034712951, 2.082777036, 2.130841121, 2.160213618, 2.200267023, 2.234979973, 2.234979973],
                            [1.308411215, 1.433911883, 1.554072096, 1.644859813, 1.732977303, 1.815754339, 1.887850467, 1.941255007, 2.008010681, 2.053404539, 2.106809079, 2.152202937, 2.189586115, 2.224299065, 2.259012016, 2.3]
                        ])

        self.header = np.array([0, 0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2, 0.22, 0.24, 0.26, 0.28, 0.3])
    
    def plotMultiaxialStrenght2D(self):
        Table=self.Table
        header=self.header
        # Create the figure with 2 subplots side by side
        fig, axs = plt.subplots(1, 2, figsize=(20, 10))

        # Subplot 1: Rows
        for i, row in enumerate(Table):
            axs[0].plot(header, row, label=f'$f\'_{{cc}}/f\'_{{co}}$ {header[i]}', color=blueAPE, linewidth=1.5)
            axs[0].text(header[-1], row[-1], f'$f\'_{{l2}}/f\'_{{co}}$ = {np.round(header[i],2)}', fontsize=8, color='k', ha='left')

        axs[0].set_xlabel('Smallest Confining Stress Ratio $f\'_{l1}/f\'_{co}$')
        axs[0].set_xlim([0, 0.35])
        xTicks = np.linspace(0, 0.3, 31)
        axs[0].set_xticks(xTicks)
        axs[0].set_xticklabels(xTicks, rotation=90)
        axs[0].set_ylabel('Confined Strength Ratio $f\'_{cc}/f\'_{co}$')
        axs[0].set_ylim([0.8, 2.5])
        yTicks = np.linspace(1, 2.30, 27)
        axs[0].set_yticks(yTicks)
        axs[0].set_title('Confined Strength Ratio $f\'_{cc}/f\'_{co}$ vs. Smallest Confining Stress Ratio $f\'_{l1}/f\'_{co}$')
        axs[0].grid(True)

        # Subplot 2: Columns
        for i in range(Table.shape[1]):
            col = Table[:, i]
            axs[1].plot(header, col, label=f'$f\'_{{cc}}/f\'_{header[i]}$', color=blueAPE, linewidth=1.5)
            axs[1].text(header[-1], col[-1], f'$f\'_{{l1}}/f\'_{{co}}$ = {np.round(header[i],2)}', fontsize=8, color='k', ha='left')

        axs[1].set_xlabel('Largest Confining Stress Ratio $f\'_{l2}/f\'_{co}$')
        axs[1].set_xlim([0, 0.35])
        axs[1].set_xticks(xTicks)
        axs[1].set_xticklabels(xTicks, rotation=90)
        axs[1].set_ylabel('Confined Strength Ratio $f\'_{cc}/f\'_{co}$')
        axs[1].set_ylim([0.8, 2.5])
        axs[1].set_yticks(yTicks)
        axs[1].set_title('Confined Strength Ratio $f\'_{cc}/f\'_{co}$ vs. Largest Confining Stress Ratio $f\'_{l2}/f\'_{co}$')
        axs[1].grid(True)


        plt.tight_layout()
        plt.show()
    
    def plotMultiaxialStrenght3D(self):
        Table=self.Table
        header=self.header
        # Create meshgrid for plots
        smallest, largest = np.meshgrid(header, header)
        Z = Table

        # Create the figure
        fig = plt.figure(figsize=(20, 10))

        # Subplot 1: Contour plot
        ax1 = fig.add_subplot(1, 2, 1)
        contour = ax1.contourf(smallest, largest, Z, cmap='viridis', levels=50)
        contour_lines = ax1.contour(smallest, largest, Z, colors='black', linewidths=0.5, levels=25)
        ax1.clabel(contour_lines, fmt='%2.2f', colors='black', fontsize=8)
        fig.colorbar(contour, ax=ax1, label='Confined Strength Ratio $f\'_{cc}/f\'_{co}$')

        ax1.set_xlabel('Smallest Confining Stress Ratio $f\'_{l1}/f\'_{co}$')
        ax1.set_ylabel('Largest Confining Stress Ratio $f\'_{l2}/f\'_{co}$')
        ax1.set_title('Contour Plot of Confined Strength Ratio $f\'_{cc}/f\'_{co}$')
        ax1.grid(True)

        # Subplot 2: 3D surface plot
        ax2 = fig.add_subplot(1, 2, 2, projection='3d')
        surf = ax2.plot_surface(smallest, largest, Z, cmap='viridis')
        # Add color bar which maps values to colors
        fig.colorbar(surf, ax=ax2, shrink=0.5, aspect=5, label='Confined Strength Ratio $f\'_{cc}/f\'_{co}$')
        ax2.set_xlabel('Smallest Confining Stress Ratio $f\'_{l1}/f\'_{co}$')
        ax2.set_ylabel('Largest Confining Stress Ratio $f\'_{l2}/f\'_{co}$')
        ax2.set_zlabel('Confined Strength Ratio $f\'_{cc}/f\'_{co}$')
        ax2.set_title('Multiaxial Strength Ratio $f\'_{cc}/f\'_{co}$')

        plt.tight_layout()
        plt.show()
              
    def calculosResistenciasCompresion(self):
        """
        Calculate compression resistances for the concrete.

        Returns:
            fcc_ratio (float): Ratio of confined to unconfined compressive strength.
            fcc (float): Confined compressive strength.
            ecc_ratio (float): Ratio of confined to unconfined strain.
            ecc (float): Confined strain.
        """
        try:
            
            fl_b=self.fl_perpendicular_b_efectivo / self.fco
            fl_h=self.fl_perpendicular_h_efectivo / self.fco
            
            fl1_ratio = np.minimum(fl_b, fl_h)
            fl2_ratio = np.maximum(fl_b, fl_h)

            Table = self.Table
            header = self.header
            
            def calculoRatiofcc(fl1_ratio, fl2_ratio, Table, header):
                """
                Calculate the ratio of confined to unconfined compressive strength based on tabulated values.

                Parameters:
                    fl1_ratio (float): Ratio of effective transverse pressure to unconfined compressive strength (width).
                    fl2_ratio (float): Ratio of effective transverse pressure to unconfined compressive strength (height).

                Returns:
                    fcc_ratio (float): Ratio of confined to unconfined compressive strength.
                """
                # smallest values for fl1/fco in rows
                # largest values of fl1/fco in cols
                
                
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
            
            fcc_ratio = calculoRatiofcc(fl1_ratio, fl2_ratio, Table, header)
            fcc = self.fco * fcc_ratio
            ecc_ratio = 1 + 5 * (fcc / self.fco - 1)
            ecc = ecc_ratio * self.eco
            
            self.addResult('fcc ratio',fcc_ratio)
            self.addResult('fcc',fcc)
            self.addResult('ecc ratio',ecc_ratio)
            self.addResult('ecc ratio',ecc)
            
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
            self.addResult('ecu',ecu)
            return ecu, ecu_ratio
        except Exception as e:
            print(f"Error in ultimate strain calculation: {e}")
            return 0, 0
    
    def relacionesConstitutivas(self, delta):
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
            
    def print_results(self):
        """
        Print the results of the calculations.
        """
        try:
            print(f"Results for {self.name}:")
            for key, value in self.result.items():
                print(f"{key}: {value}")
        except Exception as e:
            print(f"Error printing results: {e}")

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
        fc210c.plotMultiaxialStrenght2D()
        fc210c.plotMultiaxialStrenght3D()
        print(fc210c.result)
    except Exception as e:
        print(f"Error in main execution: {e}")

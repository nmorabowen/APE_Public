import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d

from plotApeConfig import blueAPE, set_default_plot_params
set_default_plot_params()

class NCh433:
    """
    A class to represent the NCh433 code parameters for seismic design.

    Attributes:
    zonaSismica : str
        The seismic zone (e.g., 'Zona 1', 'Zona 2', 'Zona 3')
    categoriaDiseño : str
        The design category (e.g., 'Tipo I', 'Tipo II', 'Tipo III', 'Tipo IV')
    tipoSuelo : str
        The type of soil (e.g., 'A', 'B', 'C', 'D', 'E')
    Ao : float
        Seismic zone coefficient
    I : float
        Importance factor
    S : float
        Soil amplification factor
    To : float
        Characteristic period of the soil
    T_star : float
        Second characteristic period of the soil
    n : float
        Soil exponent
    p : float
        Damping correction factor
    """

    def __init__(self, zonaSismica, categoriaDiseño, tipoSuelo):
        """
        Constructs all the necessary attributes for the NCh433 object.

        Parameters:
        zonaSismica : str
            The seismic zone (e.g., 'Zona 1', 'Zona 2', 'Zona 3')
        categoriaDiseño : str
            The design category (e.g., 'Tipo I', 'Tipo II', 'Tipo III', 'Tipo IV')
        tipoSuelo : str
            The type of soil (e.g., 'A', 'B', 'C', 'D', 'E')
        """
        # Load code parameters
        self.codeParams = self._loadDefaultParams()

        # Validate inputs
        self.zonaSismica = self._validate_input(zonaSismica, self.codeParams['zona'], 'zonaSismica')
        self.categoriaDiseño = self._validate_input(categoriaDiseño, self.codeParams['categoria'], 'categoriaDiseño')
        self.tipoSuelo = self._validate_input(tipoSuelo, self.codeParams['suelo'], 'tipoSuelo')
        
        # Retrieve specific parameters
        self.Ao = self.codeParams['zona'][self.zonaSismica]['Ao']
        self.I = self.codeParams['categoria'][self.categoriaDiseño]['I']
        self.S = self.codeParams['suelo'][self.tipoSuelo]['S']
        self.To = self.codeParams['suelo'][self.tipoSuelo]['To']
        self.T_star = self.codeParams['suelo'][self.tipoSuelo]['T*']
        self.n = self.codeParams['suelo'][self.tipoSuelo]['n']
        self.p = self.codeParams['suelo'][self.tipoSuelo]['p']
        
        # Calulations
        self.elastic_spectra=self.calculate_elastic_spectral_acceleration()
    
    def __str__(self):
        """
        Returns a string representation of the NCh433 object.
        """
        return (f"NCh433(zonaSismica={self.zonaSismica}, categoriaDiseño={self.categoriaDiseño}, "
                f"tipoSuelo={self.tipoSuelo}, Ao={self.Ao}, I={self.I}, S={self.S}, "
                f"To={self.To}, T*={self.T_star}, n={self.n}, p={self.p})")
    
    def __repr__(self):
        """
        Returns a string representation of the NCh433 object for debugging.
        """
        return (f"NCh433(zonaSismica={self.zonaSismica}, categoriaDiseño={self.categoriaDiseño}, "
                f"tipoSuelo={self.tipoSuelo}, Ao={self.Ao}, I={self.I}, S={self.S}, "
                f"To={self.To}, T*={self.T_star}, n={self.n}, p={self.p})")
    
    def _loadDefaultParams(self):
        """
        Loads the default parameters for the NCh433 code.

        Returns:
        dict: A dictionary containing the default parameters for the seismic zone,
              design category, and soil type.
        """
        zona = {
            'Zona 1': {'Ao': 0.20},
            'Zona 2': {'Ao': 0.30},
            'Zona 3': {'Ao': 0.40},
        }   
        categoria = {
            'Tipo I': {'I': 0.60},
            'Tipo II': {'I': 1.00},
            'Tipo III': {'I': 1.20},
            'Tipo IV': {'I': 1.20},
        }
        suelo = {
            'A': {
                'Descripcion': 'Roca, suelo cementado',
                'S': 0.90,
                'To': 0.15,
                'T*': 0.20,
                'n': 1.00,
                'p': 2.00
            },
            'B': {
                'Descripcion': 'Roca blanda o fracturada, suelo muy denso o muy firme',
                'S': 1.00,
                'To': 0.30,
                'T*': 0.35,
                'n': 1.33,
                'p': 1.50
            },
            'C': {
                'Descripcion': 'Suelo denso o firme',
                'S': 1.05,
                'To': 0.40,
                'T*': 0.45,
                'n': 1.40,
                'p': 1.60
            },
            'D': {
                'Descripcion': 'Suelo medianamente denso, o firme',
                'S': 1.20,
                'To': 0.75,
                'T*': 0.85,
                'n': 1.80,
                'p': 1.00
            },
            'E': {
                'Descripcion': 'Suelo de compacidad, o consistencia mediana',
                'S': 1.30,
                'To': 1.20,
                'T*': 1.35,
                'n': 1.80,
                'p': 1.00
            }
        }
        codeParams = {
            'zona': zona,
            'categoria': categoria,
            'suelo': suelo
        }
        
        return codeParams
    
    def _validate_input(self, input_value, valid_options, input_name):
        """
        Validates the input parameters to ensure they are within the valid options.

        Parameters:
        input_value : str
            The value to be validated.
        valid_options : dict
            A dictionary of valid options.
        input_name : str
            The name of the input parameter being validated.

        Returns:
        str: The validated input value.

        Raises:
        ValueError: If the input value is not valid.
        """
        if input_value not in valid_options:
            raise ValueError(f"Invalid value for {input_name}: {input_value}. Valid options are: {list(valid_options.keys())}")
        return input_value
    
    def printLog(self):
        """
        Prints the log of the NCh433 object attributes.
        """
        print(f'La zona sísmica es {self.zonaSismica} donde Ao = {self.Ao:.2f}')
        print(f'La categoría de diseño es {self.categoriaDiseño} donde I = {self.I:.2f}')
        print(f'El tipo de suelo es {self.tipoSuelo} correspondiente a: {self.codeParams["suelo"][self.tipoSuelo]["Descripcion"]}')
        print(f'El coeficiente S es {self.S:.2f}')
        print(f'El coeficiente To es {self.To:.2f}')
        print(f'El coeficiente T* es {self.T_star:.2f}')
        print(f'El coeficiente n es {self.n:.2f}')
        print(f'El coeficiente p es {self.p:.2f}')
    
    @staticmethod
    def Ro_function(T, Ro, To):
            return 1 + (T) / (0.10 * To + T / Ro)
    
    @staticmethod
    def R_ratio(T, Ro, To):
            return (1 + (T) / (0.10 * To + T / Ro))/Ro
        
    def calcular_R_mod(self, Tx, Ty, Ro_min=2, Ro_max=11, num_div_Ro_array=10):
        
        To=self.To

        Ro_array=np.linspace(Ro_min, Ro_max, num_div_Ro_array)
        R_mod_x=self.Ro_function(Tx,Ro_array,To)
        R_mod_y=self.Ro_function(Ty,Ro_array,To)
        ratio_x=self.R_ratio(Tx,Ro_array,To)
        ratio_y=self.R_ratio(Ty,Ro_array,To)

        R_results={
            'Tx':Tx,
            'Ty':Ty,
            'Ro_array':Ro_array,
            'R_mod_x':R_mod_x,
            'R_mod_y':R_mod_y,
            'ratio_x':ratio_x,
            'ratio_y':ratio_y
        }
        
        return R_results

    
    def plot_R_mod(self, R_results):
        
        Tx, Ty = R_results['Tx'], R_results['Ty']
        Ro_array=R_results['Ro_array']
        R_mod_x=R_results['R_mod_x']
        R_mod_y=R_results['R_mod_y']
        ratio_x=R_results['ratio_x']
        ratio_y=R_results['ratio_y']
        Ro_min=np.min(Ro_array)
        Ro_max=np.max(Ro_array)
        
        fig, ax = plt.subplots(2,2, figsize=(10,10))
        
        # Find intersection
        interp_funct=interp1d(ratio_x, Ro_array, kind='linear', fill_value='extrapolate')
        Ro_intersection=interp_funct(1.0)
        
        ax[0,0].plot(Ro_array, R_mod_x, color='k', linewidth=1.5, marker='o', label=f'R* for Tx={np.round(Tx,2)}')
        ax[0,0].plot([Ro_min, Ro_max],[Ro_min, Ro_max], color=blueAPE, linewidth=1, linestyle='-', label='Ro')
        ax[0,0].axvline(x=Ro_intersection, color='red', linestyle='--', linewidth=1.0)
        ax[0,0].annotate(f'Ro = {np.round(Ro_intersection,2)}', 
                        (Ro_intersection-0.4,Ro_min), 
                        ha='left',
                        rotation=90)
        ax[0,0].grid(True)
        ax[0,0].set_xlabel('Ro')
        ax[0,0].set_ylabel('R*')
        ax[0,0].set_title('Direccion X')
        ax[0,0].legend()
        
        ax[0,1].plot(Ro_array, ratio_x, color='k', linewidth=1.5, marker='o', label=f'R*/Ro for Tx={np.round(Tx,2)}')
        ax[0,1].axhline(y=1, color='red', linestyle='--', linewidth=1.0, label='limit')
        ax[0,1].axvline(x=Ro_intersection, color='red', linestyle='--', linewidth=1.0)
        ax[0,1].annotate(f'Ro = {np.round(Ro_intersection,2)}', 
                        (Ro_intersection-0.4,np.min(ratio_x)), 
                        ha='left',
                        rotation=90)
        ax[0,1].grid(True)
        ax[0,1].set_xlabel('Ro')
        ax[0,1].set_ylabel('Ratio R*/Ro')
        ax[0,1].set_title('Direccion X')
        ax[0,1].legend()
        
        # Find intersection
        interp_funct=interp1d(ratio_y, Ro_array, kind='linear', fill_value='extrapolate')
        Ro_intersection=interp_funct(1.0)
        
        ax[1,0].plot(Ro_array, R_mod_y, color='k', linewidth=1.5, marker='o', label=f'R* for Ty={np.round(Ty,2)}')
        ax[1,0].axvline(x=Ro_intersection, color='red', linestyle='--', linewidth=1.0)
        ax[1,0].plot(Ro_array,Ro_array, color=blueAPE, linewidth=1, linestyle='-', label='Ro')
        ax[1,0].annotate(f'Ro = {np.round(Ro_intersection,2)}', 
                         (Ro_intersection-0.4,Ro_min), 
                         ha='left',
                         rotation=90)
        ax[1,0].grid(True)
        ax[1,0].set_xlabel('Ro')
        ax[1,0].set_ylabel('R*')
        ax[1,0].set_title('Direccion Y')
        ax[1,0].legend()
        
        ax[1,1].plot(Ro_array, ratio_y, color='k', linewidth=1.5, marker='o', label=f'R*/Ro for Ty={np.round(Ty,2)}')
        ax[1,1].axhline(y=1, color='red', linestyle='-', linewidth=1.5, label='limit')
        ax[1,1].axvline(x=Ro_intersection, color='red', linestyle='--', linewidth=1.0)
        ax[1,1].annotate(f'Ro = {np.round(Ro_intersection,2)}', 
                         (Ro_intersection-0.4,np.min(ratio_y)), 
                         ha='left',
                         rotation=90)
        ax[1,1].grid(True)
        ax[1,1].set_xlabel('Ro')
        ax[1,1].set_ylabel('Ratio R*/Ro')
        ax[1,1].set_title('Direccion Y')
        ax[1,1].legend()
    
    def plot_R_mod_3D(self, R_results):
        To = self.To
        Tx, Ty = R_results['Tx'], R_results['Ty']
        Ro_array = R_results['Ro_array']
        T_array = self.elastic_spectra['T']
        
        T_mesh, Ro_mesh = np.meshgrid(T_array, Ro_array)

        R_mod = self.Ro_function(T_mesh, Ro_mesh, To)
        ratio_mod = self.R_ratio(T_mesh, Ro_mesh, To)

        fig = plt.figure(figsize=(5,5))
        ax = fig.add_subplot(111, projection='3d')

        ax.plot_surface(T_mesh, Ro_mesh, ratio_mod, cmap='viridis')
        ax.view_init(elev=45, azim=315)

        ax.set_xlabel('T*')
        ax.set_ylabel('Ro')
        ax.set_zlabel('R*/Ro')
        ax.set_title('Surface Plot of R*/Ro')

        plt.show()

        plt.figure()
        contour = plt.contourf(T_mesh, Ro_mesh, ratio_mod, cmap='viridis')
        plt.colorbar(contour)
        plt.plot([R_results['Tx'], R_results['Ty']],[R_results['R_mod_x'], R_results['R_mod_y']], marker='o')
        plt.xlabel('T*')
        plt.ylabel('Ro')
        plt.title('Contour Plot of R*/Ro')

        plt.show()
        
        
    def calculate_elastic_spectral_acceleration(self, T_lim=4, num_Tn_array=100):
        Tn_array = np.linspace(0, T_lim, num_Tn_array)

        def alpha_function(Tn, To, p):
            return (1 + 4.50 * (Tn / To)**p) / (1 + (Tn / To)**3)

        def Sa_function(S, Ao, alpha, I):
            return S * Ao * alpha * I

        alpha_array = alpha_function(Tn_array, self.To, self.p)
        Sa = Sa_function(self.S, self.Ao, alpha_array, self.I)
        
        elastic_spectra={
            'Sa':Sa,
            'T':Tn_array
        }
        
        return elastic_spectra
    
    def plot_spectral_acceleration(self, ax=None, marker=None):

        T=self.elastic_spectra['T']
        Sa=self.elastic_spectra['Sa']
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(10,5))

        ax.plot(T, Sa, color=blueAPE, linewidth=1.5, marker=marker)
        ax.grid(True)
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        ax.set_xlabel('Period (T)')
        ax.set_ylabel('Spectral Acceleration (Sa)')
        plt.show()

# Example usage:
try:
    n = NCh433('Zona 1', 'Tipo II', 'B')
    print(n)
    n.printLog()
    R_results=n.calcular_R_mod(Tx=1.0, Ty=1.5)
    n.plot_R_mod(R_results)
    n.plot_R_mod_3D(R_results)
    direccion_X = {'T': np.linspace(0, 4, 100), 'Ro': np.random.uniform(1, 3, 100)}
    direccion_Y = {'T': np.linspace(0, 4, 100), 'Ro': np.random.uniform(1, 3, 100)}
    n.plot_spectral_acceleration(marker='o')
except ValueError as e:
    print(e)
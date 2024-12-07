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
        Initializes the NCh433 object with the specified seismic zone, design category, and soil type.

        Parameters:
        zonaSismica : str
            The seismic zone (e.g., 'Zona 1', 'Zona 2', 'Zona 3').
        categoriaDiseño : str
            The design category (e.g., 'Tipo I', 'Tipo II', 'Tipo III', 'Tipo IV').
        tipoSuelo : str
            The type of soil (e.g., 'A', 'B', 'C', 'D', 'E').
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
        
        # Calculations
        self.elastic_spectra = self.calculate_elastic_spectral_acceleration()
    
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
        return self.__str__()
    
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
            'A': {'S': 0.90, 'To': 0.15, 'T*': 0.20, 'n': 1.00, 'p': 2.00},
            'B': {'S': 1.00, 'To': 0.30, 'T*': 0.35, 'n': 1.33, 'p': 1.50},
            'C': {'S': 1.05, 'To': 0.40, 'T*': 0.45, 'n': 1.40, 'p': 1.60},
            'D': {'S': 1.20, 'To': 0.75, 'T*': 0.85, 'n': 1.80, 'p': 1.00},
            'E': {'S': 1.30, 'To': 1.20, 'T*': 1.35, 'n': 1.80, 'p': 1.00},
        }
        return {'zona': zona, 'categoria': categoria, 'suelo': suelo}
    
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
    
    def calculate_elastic_spectral_acceleration(self, T_lim=4, num_Tn_array=100):
        """
        Calculates the elastic spectral acceleration for the NCh433 code.

        Parameters:
        T_lim : float
            Maximum period for the spectrum (default is 4 seconds).
        num_Tn_array : int
            Number of discrete points for the period array (default is 100).

        Returns:
        dict: A dictionary containing periods ('T') and spectral accelerations ('Sa').
        """
        Tn_array = np.linspace(0, T_lim, num_Tn_array)
        
        def alpha_function(Tn, To, p):
            return (1 + 4.50 * (Tn / To)**p) / (1 + (Tn / To)**3)

        def Sa_function(S, Ao, alpha, I):
            return S * Ao * alpha * I

        alpha_array = alpha_function(Tn_array, self.To, self.p)
        Sa = Sa_function(self.S, self.Ao, alpha_array, self.I)
        return {'Sa': Sa, 'T': Tn_array}
    
    def plot_spectral_acceleration(self):
        """
        Plots the elastic spectral acceleration.
        """
        T = self.elastic_spectra['T']
        Sa = self.elastic_spectra['Sa']
        plt.figure(figsize=(10, 6))
        plt.plot(T, Sa, color=blueAPE, linewidth=2)
        plt.title("Elastic Spectral Acceleration", fontsize=14)
        plt.xlabel("Period (T)", fontsize=12)
        plt.ylabel("Spectral Acceleration (Sa)", fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.show()

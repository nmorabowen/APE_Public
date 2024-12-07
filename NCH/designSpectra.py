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
    elastic_spectra : dict
        Precomputed elastic spectral acceleration for the given parameters
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
        
        # Precompute elastic spectral acceleration
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
        Returns a detailed string representation of the NCh433 object for debugging.
        """
        return self.__str__()
    
    def _loadDefaultParams(self):
        """
        Loads the default parameters for the NCh433 code.

        Returns:
        dict: A dictionary containing the default parameters for the seismic zone,
              design category, and soil type.
        """
        # Default parameters for the code
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
        Prints a detailed log of the NCh433 object attributes.
        """
        print(f'La zona sísmica es {self.zonaSismica} donde Ao = {self.Ao:.2f}')
        print(f'La categoría de diseño es {self.categoriaDiseño} donde I = {self.I:.2f}')
        print(f'El tipo de suelo es {self.tipoSuelo} correspondiente a: {self.codeParams["suelo"][self.tipoSuelo]["Descripcion"]}')
        print(f'El coeficiente S es {self.S:.2f}')
        print(f'El coeficiente To es {self.To:.2f}')
        print(f'El coeficiente T* es {self.T_star:.2f}')
        print(f'El coeficiente n es {self.n:.2f}')
        print(f'El coeficiente p es {self.p:.2f}')

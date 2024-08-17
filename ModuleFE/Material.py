"""
Description:
    Este archivo contiene la clase material para el modulo de elementos finitos.

Date:
    2024-06-12
"""

__author__ = "Nicolás Mora Bowen"
__version__ = "1.0.0"

import numpy as np

class Material:
    def __init__(self,E,nu,gamma,type='Plane Stress'):
        # Definimos un modelo elastoplastico bilineal para el material
        self.E=E
        self.gamma=gamma
        self.nu=nu
        
        if type == 'Plane Stress':
            self.Emat=self.calculate_plane_stress(E,nu)
        elif type == 'Plane Strain':
            self.Emat=self.calculate_plane_strain(E,nu)
        elif type == 'Iso':
            self.Emat=self.calculate_isotropic(E,nu)
            
    def calculate_plane_strain(self,E,nu):
        Emat=np.array([[1-nu, nu, 0.],
                       [nu,1.-nu,0.],
                       [0.,0.,0.50-nu]])*(E)/((1.+nu)*(1.-2.*nu))
        return Emat
    
    def calculate_plane_stress(self,E,nu):
        Emat=np.array([[1, nu, 0.],
                       [nu,1.,0.],
                       [0.,0.,(1.-nu)/2.]])*(E/(1.-nu**2.))
        return Emat
    
    def calculate_isotropic(self, E, nu):
        Emat=np.array([[1-nu, nu, nu, 0, 0, 0],
                       [nu, 1-nu, nu, 0, 0, 0],
                       [nu, nu, 1-nu, 0, 0, 0],
                       [0, 0, 0, 0.5-nu, 0, 0],
                       [0, 0, 0, 0, 0.5-nu, 0],
                       [0, 0, 0, 0, 0, 0.5-nu]
                       ])*(E/((1+nu)*(1-2*nu)))
        
        return Emat
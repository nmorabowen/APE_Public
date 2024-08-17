"""
Description:
    Este archivo contiene la clase seccion para el modulo de elementos finitos.

Date:
    2024-06-12
"""

__author__ = "Nicolás Mora Bowen"
__version__ = "1.0.0"

class Section:
    def __init__(self,A,I,Av,material):
        self.A=A
        self.I=I
        self.Av=Av
        self.material=material
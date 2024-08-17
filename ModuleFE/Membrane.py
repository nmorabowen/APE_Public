"""
Description:
    Este archivo contiene la clase membrana para el modulo de elementos finitos.

Date:
    2024-06-12
"""

__author__ = "Nicolás Mora Bowen"
__version__ = "1.0.0"

class Membrane:
    def __init__(self, name, thickness, material):
        self.name=name
        self.thickness=thickness
        self.material=material
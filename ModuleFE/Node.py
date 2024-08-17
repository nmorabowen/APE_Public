"""
Description:
    Este archivo contiene la clase nudo para el modulo de elementos finitos.

Date:
    2024-06-26
"""

__author__ = "Nicolás Mora Bowen"
__version__ = "1.1.0"

import numpy as np
from ModuleFE import Parameters

class Node:
    def __init__(self,name,coord_list,restrain=None, load=None, restrain_displacement=None, nDof=None):
        self.name=name
        self.coord=np.array(coord_list)
        self.load=load
        self.restrain=restrain
        self.restrain_displacement=restrain_displacement
        self.number=name-1
        self.nDof=nDof
        
        # Inicializamos componentes
        self._initialize_nDof()
        self.index=self.set_index()
        self._initialize_load()
        self._initialize_boundaryConditions()
        self._initialize_restrainDisplacement()
        
    def __str__(self):
        return f"Node: {self.name}"
    
    def set_number(self, value):
        # Funcion para modificar el number despues de que se haya inializado el objeto
        self.number=int(value)
    
    def _initialize_nDof(self):
        if self.nDof is None:
            self.nDof=Parameters.nDof
            
    def set_index(self):
        # Determinamos los gdl del nudo y corregimos para empezar a numerar desde 0
        index=np.linspace(0, self.nDof-1, self.nDof)+(self.nDof*self.number)
        index=index.astype(int)
        return index

    def _initialize_load(self):
        if self.load is None:
            self.load=np.zeros(self.nDof)
        elif isinstance(self.load, list) and len(self.load)==self.nDof:
            self.load=np.array(self.load)
        else:
            raise ValueError('Las restricciones deben ser una lista y tener el tamaño correcto')
    
    def _initialize_boundaryConditions(self):
        # Funcion para establecer las condiciones de borde durante la instanciacion del objeto
        if self.restrain is None:
            self.restrain=np.full(self.nDof,'f')
        elif isinstance(self.restrain, list) and len(self.restrain)==self.nDof:
            self.restrain=np.array(self.restrain)
        else:
            raise ValueError('Las restricciones deben ser una lista y tener el tamaño correcto')
    
    def _initialize_restrainDisplacement(self):
        # Funcion para establecer las deformaciones del apoyo durante la instanciacion del objeto
        if self.restrain_displacement is None:
            self.restrain_displacement=np.zeros(self.nDof)
        elif isinstance(self.restrain_displacement, list) and len(self.restrain_displacement)==self.nDof:
            self.restrain_displacement=np.array(self.restrain_displacement)
        else:
            raise ValueError("Los desplazamientos de apoyo no tienen el tamaño correcto")
    
    def set_boundaryConditions(self,restrain):
        # Funcion para establecer las condiciones de borde despues de que se haya instanciado el objeto
        # 'f' es libres y 'r' es restringidas
        if isinstance(restrain,list) and len(restrain)==self.nDof:
            self.restrain=np.array(restrain)
        else:
            raise ValueError('Las restricciones deben ser una lista y tener el tamaño correcto')
        
    def set_restrainDisplacement(self, restrain_displacement):
        # Funcion para establecer las deformaciones del apoyo despues de que se haya instanciado el objeto
        if restrain_displacement is None:
            self.restrain_displacement=np.zeros(self.nDof)
        elif isinstance(restrain_displacement, list) and len(restrain_displacement)==self.nDof:
            self.restrain_displacement=np.array(restrain_displacement)
        else:
            raise ValueError("Los desplazamientos de apoyo no tienen el tamaño correcto")
        
    def apply_load(self,P):
        # Funcion para establecer las cargas despues de que se haya instanciado el objeto
        if isinstance(P,list) and len(P)==Parameters.nDof:
            self.load=self.load+np.array(P)
        else:
            raise ValueError('P debe ser una lista y tener el tamaño correcto')
        
    def plot_node(self, ax, plot_gmsh_name=False, plot_py_name=False):
        ax.plot(self.coord[0], self.coord[1], '.', color='k')
        if plot_gmsh_name is True:
            ax.text(self.coord[0],self.coord[1],str(self.name), fontsize=10)
        if plot_py_name is True:
            ax.text(self.coord[0],self.coord[1],str(self.number), fontsize=10)
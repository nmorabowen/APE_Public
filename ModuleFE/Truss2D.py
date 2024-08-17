"""
Description:
    Este archivo contiene la clase de barras para el modulo de elementos finitos.

Date:
    2024-06-12
"""

__author__ = "Nicolás Mora Bowen"
__version__ = "1.0.0"

import numpy as np

class Truss2D:
    def __init__(self, node_i, node_j, section, gravity_load=True):
        self.node_i=node_i
        self.node_j=node_j
        self.section=section
        self.F_fe_global=np.array([0,0,0,0])
        
        self.index, self.restrains = self.calculate_index()
        self.L, self.alpha_rad, self.alpha = self.calculate_geometry()
        self.Tbl, self.Tlg = self.calculate_transformationMatrix()
        self.Kb, self.Kl, self.Kg = self.calculate_initialStiffness()
        
        if gravity_load is True:
            gravity_load_global=self.set_load_gravity()
            self.F_fe_global = self.F_fe_global + gravity_load_global
    
    def calculate_index(self):
        index_i=self.node_i.index
        index_j=self.node_j.index
        index=np.append(index_i,index_j)
        
        i_restrain=self.node_i.restrain
        j_restrain=self.node_j.restrain
        restrains=np.append(i_restrain,j_restrain)
        
        return index, restrains
      
    def calculate_geometry(self):
        vector=self.node_j.coord-self.node_i.coord
        L=np.linalg.norm(vector)
        angle_rad=np.arctan2(vector[1],vector[0])
        angle_deg=np.degrees(angle_rad)
        return L, angle_rad, angle_deg
    
    def calculate_transformationMatrix(self):
        alpha=self.alpha_rad
        c=np.cos(alpha)
        s=np.sin(alpha)
        
        Tbl=np.array([[-1, 1]])
        Tlg=np.array([[c,s,0,0],
                      [0,0,c,s]])
        
        return Tbl, Tlg
    
    def set_load_gravity(self):
        q=self.section.material.gamma*self.section.A
        L=self.L
        F_fe_global=np.array([0,q*L/2,0,q*L/2])
        
        return F_fe_global
    
    def calculate_initialStiffness(self):
        E=self.section.material.E
        A=self.section.A
        L=self.L
        
        Kb=np.array([[E*A/L]])
        Kl=np.dot(self.Tbl.transpose(),np.dot(Kb, self.Tbl))
        Kg=np.dot(self.Tlg.transpose(),np.dot(Kl,self.Tlg))
        
        return Kb, Kl, Kg
    
    def get_displacements(self, u_global):
        u_global_bar=u_global[self.index]
        u_local_bar=self.Tlg@u_global_bar
        u_basic_bar=self.Tbl@u_local_bar
        
        return u_basic_bar, u_local_bar, u_global_bar
    
    def get_forces(self, u_global):
        u_basic, u_local, u_global = self.get_displacements(u_global)
        F_basic = self.Kb@u_basic
        F_local = self.Kl@u_local
        F_global = self.Kg@u_global
        
        return F_basic, F_local, F_global
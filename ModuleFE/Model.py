"""
Description:
    Este archivo contiene la clase Quad2D para el modulo de elementos finitos.

Date:
    2024-06-12
"""
__author__ = "Nicolás Mora Bowen"
__version__ = "1.0.0"

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import roots_legendre
import matplotlib.patches as patches
from ModuleFE.Funciones import matrixReplacement
from ModuleFE.Funciones import matrixExtract
from ModuleFE import Parameters
            
class Model:
    def __init__(self, nodes, elements):
        self.nodes=nodes
        self.elements=elements
        
        self.number_elements, self.number_nodes, self.number_DoF = self.calculate_gen_properties()
        self.node_index, self.node_index_restrains, self.index_free, self.index_restrains = self.calculate_index_matrix()
        
    def calculate_gen_properties(self):
        """Calculamos las propiedades generales de los elementos del sistema

        Returns:
            number_elements: Numero total de elementos en el sistema
            number_nodes: Numero total de nudos en el sistema
            number_DoF: Numero de grados de libertad del sistema
        """
        number_elements=len(self.elements)
        number_nodes=len(self.nodes)
        number_DoF=number_nodes*Parameters.nDof
        return number_elements, number_nodes, number_DoF
    
    def calculate_index_matrix(self):
        """ Calculamos lo indices de colocacion del sistema

        Returns:
            node_index (np.array): numero de los nudos
            node_index_restrains (np.array): numero de los nudos restringidos
            index_free (np.array): indices de los nudos libres
            index_restrains (np.array): indices de los nudos restringidos
        """
        node_index=np.zeros(self.number_DoF)
        node_index_restrains=np.full(self.number_DoF,'',dtype=str)
        for node in self.nodes[:,0]:
            node_index[node.index]=node.index
            node_index=node_index.astype(int)
            node_index_restrains[node.index]=node.restrain
        
        # Encontramos los indices libres
        indices_free=np.where(node_index_restrains == 'f')[0]
        index_free=node_index[indices_free]
        
        # Encontramos los indices libres
        indices_restrain=np.where(node_index_restrains == 'r')[0]
        index_restrains=node_index[indices_restrain]
        
        return node_index, node_index_restrains, index_free, index_restrains
    
    def calculate_stiffnessMatrix(self):
        """ Procedimiento de ensamble de la matriz de rigidez

        Returns:
            K (np.array): Matriz de rigidez emsamblada completa
        """
        K=np.zeros((self.number_DoF, self.number_DoF))
        for element in self.elements[:,0]:
            K_mapping=element.Kg
            K=matrixReplacement(K,K_mapping,element.index,element.index)
        return K
    
    def calculate_forceVector(self):
        """Calculo del los vectores de fuerza de nudos y de elemento

        Returns:
            F_nodes (np.array): Cargas aplicadas a los nudos
            F_fe (np.array): Cargas de empotramiento perfecto o cargas de cuerpo
        """
        F_nodes=np.zeros(self.number_DoF)
        for node in self.nodes[:,0]:
            F_nodes[node.index]=F_nodes[node.index]+node.load
        F_nodes=F_nodes[:, np.newaxis]
        
        F_fe=np.zeros(self.number_DoF)
        for element in self.elements[:,0]:
            F_fe[element.index]=F_fe[element.index]+element.F_fe_global
        F_fe=F_fe[:, np.newaxis]
        
        return F_nodes, F_fe
    
    def calculate_restrainDisplacements(self):
        """Calculo de los desplazamientos iniciales en los apoyos

        Returns:
            u_restrains (np.array): Calculo del vector de desplazamiento de apoyo
        """
        u_restrains=np.zeros(self.number_DoF)
        for node in self.nodes[:,0]:
            u_restrains[node.index]=u_restrains[node.index]+node.restrain_displacement
            
        return u_restrains
    
    def partition_stiffnessMatrix(self, K):
        """Algoritomo de particion de la matriz de rigidez

        Args:
            K (np.array): matriz de rigidez completa

        Returns:
            Kff (np.array): particion libre/libre
            Kfr (np.array): particion libre/restringida
            Krf (np.array): particion restringida/libre
            Krr (np.array): particion restringida/restringida
        """
        index_free= self.index_free
        index_restrain=self.index_restrains
        Kff=matrixExtract(K,index_free,index_free)
        Kfr=matrixExtract(K,index_free,index_restrain)
        Krf=matrixExtract(K,index_restrain,index_free)
        Krr=matrixExtract(K,index_restrain,index_restrain)
        
        return Kff, Kfr, Krf, Krr
    
    def solveMMS_forces(self):
        index_free= self.index_free
        index_restrain=self.index_restrains
        K=self.calculate_stiffnessMatrix()
        Kff, Kfr, Krf, Krr = self.partition_stiffnessMatrix(K)
        F_nodes, F_fe = self.calculate_forceVector()
        print(f'La resultanto total de cargas aplicadas es: {np.round(np.sum(F_nodes+F_fe),2)}')
        u_restrain = self.calculate_restrainDisplacements()
        
        # Resolvemos el sistema de ecuaciones
        #[F]-[F_fe]=[K][u]
        #[F_f]-[F_fe_f]=[Kff][uf]+[Kfr][ur]
        #[uf] = [Kff]^-1([F_f]-[F_fe_f]-[Kfr][ur])
        ur=u_restrain[index_restrain]
        ur=ur[:,np.newaxis]
        #F_f=Kff@uf+Kfr@ur
        F_f=Kfr@ur
        
        return F_f
        
        
    
    def solve_linearElastic(self):
        """Calculo linear elastico

        Returns:
            uf (np.array): vector de desplzamientos libres
            F_r (np.array): vector de reacciones
            u (np.array): vector de desplazamientos
            F (np.array): cector de fuerzas
        """
        index_free= self.index_free
        index_restrain=self.index_restrains
        K=self.calculate_stiffnessMatrix()
        Kff, Kfr, Krf, Krr = self.partition_stiffnessMatrix(K)
        F_nodes, F_fe = self.calculate_forceVector()
        print(f'La resultanto total de cargas aplicadas es: {np.round(np.sum(F_nodes+F_fe),2)}')
        u_restrain = self.calculate_restrainDisplacements()
        
        # Resolvemos el sistema de ecuaciones
        #[F]-[F_fe]=[K][u]
        #[F_f]-[F_fe_f]=[Kff][uf]+[Kfr][ur]
        #[uf] = [Kff]^-1([F_f]-[F_fe_f]-[Kfr][ur])
        ur=u_restrain[index_restrain]
        ur=ur[:,np.newaxis]
        F_f=F_nodes[index_free]-F_fe[index_free]+Kfr@ur
        uf=np.linalg.solve(Kff, F_f)
        
        # Calculamos las reacciones
        #[F_r]-[F_fe_r]=[Kfr][uf]+[Krr][ur]
        #[F_r]=[Kfr][uf]+[Krr][ur]+[F_fe_r]
        F_r=Krf@uf+Krr@ur+F_fe[index_restrain]-F_nodes[index_restrain]
        
        # Ensamblamos el vector completo de desplazamientos y de Fuerzas
        u=np.zeros((self.number_DoF,1))
        F=np.zeros((self.number_DoF,1))
        
        u[index_free]=uf
        u[index_restrain]=ur
        
        F[index_free]=F_nodes[index_free]
        F[index_restrain]=F_r
        F[index_restrain]+=F_nodes[index_restrain]
        
        
        print(f'La resultante de fuerzas y reacciones es: {np.round(np.sum(F),2)}')
        
        return uf, F_r, u, F
    
    def draw_model(self):
        plt.figure()
        for node in self.nodes[:,0]:
            plt.plot(node.coord[0], node.coord[1], 'ko')
        for element in self.elements[:,0]:
            x_value=[element.node_i.coord[0],element.node_j.coord[0]]
            y_value=[element.node_i.coord[1],element.node_j.coord[1]]
            plt.plot(x_value,y_value, linestyle='-', color='k')
            
    def set_results(self,u):
        elements=self.elements
        for index, element in enumerate(self.elements):
            stress, strain, ue = element[0].get_element_stress(u)
            principal_stress=element[0].calculate_principal_stress(stress)
            principal_strain=element[0].calculate_principal_strain(strain)
            elements[index][0].set_results(stress, strain, ue, principal_stress, principal_strain)
        return elements   

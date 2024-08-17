"""
Description:
    Este archivo contiene la clase CST para el modulo de elementos finitos.

Date:
    2024-06-12
"""

__author__ = "Nicolás Mora Bowen"
__version__ = "1.1.0"

import numpy as np

class CST:
    def __init__(self,elementTag, node_list, section, load_direction=None):
        
        """
        Initialize the CST element with nodes, section properties, and optional load direction.

        Args:
            elementTag (int): Unique identifier for the element.
            node_list (list): List of three nodes defining the CST element. The list is expected to be ordered.
            section (object): Section object containing material properties and thickness.
            node_index (list, optional): List of node indices. Defaults to None.
            load_direction (list, optional): List [Cx, Cy] for gravitational load direction. Defaults to None.
        """
        # Number of elements validation
        if len(node_list) != 3:
            raise ValueError("node_list must contain exactly 3 nodes.")
        
        self.elementTag=elementTag
        self.node_list=node_list
        self.section=section
        self.load_direction=load_direction
        
        self._initialize_load_direction()
        
        self.xy=self.calculate_xy()
        self.center_coord=self.calculate_centerPoint()
        self.B, self.A=self.calculate_B_matrix()
        self.Kg=self.calculate_K0()
        self.index=self.calculate_indices()
        self.F_fe_global=self.calculo_bodyForces()
        
    def __str__(self):
        node_name_list=[node.name for node in self.node_list]
        return f'CST {node_name_list}'
        
    def _initialize_load_direction(self):
        """Algoritmo para inicializar las fuerzas de cuerpo.
           Si no se especifica se asume que no se quiere considerar carga gravitacional.
           Se busca una lista [Cx, Cy] donde se establecen los coeficientes para el calculo de las fuerzas de cuerpo gravitacionales.
        """
        if self.load_direction is None:
            self.load_direction=[0,0]
            
    def calculate_centerPoint(self):
        """Calculamos las coordenas del centroide del CST

        Returns:
            center_coord (np.array): Vector con las coordenadas [x,y] del centroide
        """
        center_coord = np.mean(self.xy, axis=0)
        return center_coord
    
    def calculate_indices(self):
        """Calculamos los indices de colocacion del elemento CST

        Returns:
            index (np.array): vector de (1x6) con los indices asociados a los GDL del elemento CST
        """
        index=[]
        index = np.hstack([node.index for node in self.node_list])
        return index
    
    def calculate_xy(self):
        """
        Calculate the coordinates of the nodes in the element.
        
        Returns:
            np.array: A array with each row representing the coordinates of a node.
        """
        # Preallocate a zero array with dimensions (number of nodes, number of coordinates per node)
        xy=np.zeros((len(self.node_list), len(self.node_list[0].coord)))
        # Fill the preallocated array with the coordinates of each node
        for i, node in enumerate(self.node_list):
            xy[i,:]=node.coord
        return xy
    
    def calculate_B_matrix(self):
        """Algoritmo de calculo de la matriz de interpolacion de desplazamientos y el area del elemento CST

        Returns:
            B (np.array): Matriz de interpolacion de desplazamientos (3x6)
            A (double): Area del elemento CST
        """
        E=self.section.material.E
        t=self.section.thickness
        
        # Coordinates of the nodes
        x0, y0 = self.xy[0]
        x1, y1 = self.xy[1]
        x2, y2 = self.xy[2]
        
        #...  implement
        A2 = x0*y1 - x0*y2 - x1*y0 + x1*y2 + x2*y0 - x2*y1
        dζ0_dx = (y1 - y2)/A2
        dζ1_dx = (-y0 + y2)/A2
        dζ2_dx = (y0 - y1)/A2
        dζ0_dy = (-x1 + x2)/A2
        dζ1_dy = (x0 - x2)/A2
        dζ2_dy = (-x0 + x1)/A2
        A = abs(A2) / 2
        
        B = np.array(
                    [[dζ0_dx , 0      , dζ1_dx , 0      , dζ2_dx , 0       ] ,
                    [     0  , dζ0_dy , 0      , dζ1_dy , 0      , dζ2_dy  ] ,
                    [dζ0_dy  , dζ0_dx , dζ1_dy , dζ1_dx , dζ2_dy , dζ2_dx  ] ]
                    , dtype=np.double)
        
        return B, A
            
    def calculate_K0(self):
        """Calculo de la matriz de rigidez inicial

        Returns:
            Ko: Matriz de rigidez inicial (6x6)
        """
        E_mat = self.section.material.Emat
        thickness = self.section.thickness

        Ko = np.dot(np.dot(self.B.T, E_mat), self.B) * self.A * thickness
        return Ko
    
    def calculo_bodyForces(self):
        """Calculo de las fuerzas de cuerpo, definido como Volumen*gamma*vector de direccion

        Returns:
            F_fe_global (np.array): vector de fuerzas de cuerpo
        """
        gamma=self.section.material.gamma
        A=self.A
        t=self.section.thickness
        V=A*t
        load_direction=np.array(self.load_direction)
        F=V*gamma*load_direction
        F_fe_global=np.array([F[0]/3,F[1]/3,F[0]/3,F[1]/3,F[0]/3,F[1]/3], dtype=np.double)
        return F_fe_global
       
    def get_element_displacements(self, u):
        """Algoritomo para extraer los desplazamientos del elemento a partir de los desplazamientos del sistema

        Args:
            u (np.array): vector de desplazamientos de todo el sistema

        Returns:
            ue (np.array): vector de desplazamientos del elemento
        """
        index=self.index
        ue=u[index]
        return ue
    
    def get_element_strains(self, u):
        ue=self.get_element_displacements(u)
        epsilon_e=self.B@ue
        return epsilon_e, ue
    
    def get_element_stress(self, u):
        epsilon_e, ue = self.get_element_strains(u)
        sigma_e=self.section.material.Emat@epsilon_e
        return sigma_e, epsilon_e, ue
    
    def set_results(self, stress, strain, displacement, principal_stress, principal_strain):
        self.sigma=stress
        self.epsilon=strain
        self.displacement=displacement
        self.principal_stress=principal_stress
        self.principal_strain=principal_strain
        
    def calculate_principal_stress(self, sigma):
        
        sx=sigma[0][0]
        sy=sigma[1][0]
        sxy=sigma[2][0]
        
        stress_matrix = np.array([[sx, sxy],
                                  [sxy, sy]])

        # Diagonalize the stress matrix
        eigenvalues, eigenvectors = np.linalg.eig(stress_matrix)

        # Sort eigenvalues and eigenvectors
        sorted_indices = np.argsort(eigenvalues)[::-1]  # Sort in descending order
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]

        # Principal stresses are the eigenvalues
        sigma1, sigma2 = eigenvalues
        
        return np.array([[sigma1],[sigma2]])
    
    def calculate_principal_strain(self, epsilon):
        
        ex=epsilon[0][0]
        ey=epsilon[1][0]
        exy=epsilon[2][0]
        
        strain_matrix = np.array([[ex, exy],
                                  [exy, ey]])

        # Diagonalize the stress matrix
        eigenvalues, eigenvectors = np.linalg.eig(strain_matrix)

        # Sort eigenvalues and eigenvectors
        sorted_indices = np.argsort(eigenvalues)[::-1]  # Sort in descending order
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]

        # Principal stresses are the eigenvalues
        epsilon1, epsilon2 = eigenvalues
        
        return np.array([[epsilon1],[epsilon2]])
    
    def plot_CST(self, ax, plot_name=False, plot_nodes=False, plot_nodes_gmsh_names=False, plot_nodes_py_names=False, marker_size=10, font_size=10):
        # TIENE QUE SER REVISADO!!!!!
        """
        Plot the CST element.

        Args:
            ax (matplotlib.axes.Axes): Matplotlib Axes object to plot on.
            plot_name (bool, optional): Plot element name at centroid. Defaults to False.
            plot_nodes (bool, optional): Plot nodes of the element. Defaults to False.
            plot_nodes_gmsh_names (bool, optional): Plot nodes using GMSH names. Defaults to False.
            plot_nodes_py_names (bool, optional): Plot nodes using Python names. Defaults to False.
            marker_size (int, optional): Size of the node markers. Defaults to 10.
            font_size (int, optional): Font size for text annotations. Defaults to 10.
        """
        x_coord, y_coord = zip(*self.xy)
        ax.plot(x_coord + (x_coord[0],), y_coord + (y_coord[0],), color='black', linewidth=1)
        ax.fill_between(x_coord, y_coord, color='skyblue', alpha=0.5)

        if plot_name:
            ax.text(self.center_coord[0], self.center_coord[1], str(self.elementTag), ha='center', va='center', fontsize=font_size)

        if plot_nodes:
            for i, node in enumerate(self.node_list):
                ax.plot(node.coord[0], node.coord[1], marker='.', color='k', markersize=marker_size)
                if plot_nodes_gmsh_names:
                    ax.text(node.coord[0], node.coord[1], str(node.name), color='k', fontsize=font_size)
                elif plot_nodes_py_names:
                    ax.text(node.coord[0], node.coord[1], str(i), color='k', fontsize=font_size)
"""
Description:
    Este archivo contiene la clase Tet4 para el modulo de elementos finitos.

Date:
    2024-06-29
"""
__author__ = "Nicolás Mora Bowen"
__version__ = "1.0.0"

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class Tet4:
    """
    Class representing a 4-node tetrahedral finite element.

    Attributes:
        nDof (int): Number of degrees of freedom per node.
        node_list (list): List of nodes defining the element.
        elementTag (int): Identifier for the element.
        material (object): Material properties for the element.
        load_direction (list): Direction of the applied load. Defaults to [0, 0, 0].
        nodes (ndarray): Coordinates of the nodes.
        Kg (ndarray): Element stiffness matrix.
        B (ndarray): Strain-displacement matrix.
        F_fe_global (ndarray): Element force vector.
        V (float): Volume of the element.
        index (ndarray): Indices of the degrees of freedom.
    """

    
    def __init__(self, elementTag, node_list, material, load_direction=None, nDof=3):
        """
        Initializes a new Tet4 element.

        Args:
            elementTag (int): Identifier for the element.
            node_list (list): List of nodes defining the element.
            material (object): Material properties for the element.
            load_direction (list, optional): Direction of the applied load. Defaults to None.
            nDof (int, optional): Number of degrees of freedom per node. Defaults to 3.
        
        Raises:
            ValueError: If node_list does not contain exactly 4 nodes or if any node's coordinates are not 3x1 arrays.
        """
        
        # Number of elements validation
        if len(node_list) != 4:
            raise ValueError("node_list must contain exactly 4 nodes.")
        
        # Validate that each element in node_list is a 3x1 array
        for node in node_list:
            if not isinstance(node.coord, np.ndarray) or node.coord.shape != (3,):
                raise ValueError("Each node in node_list must be a 3x1 array (3 elements).")
        
        self.nDof=nDof
        self.node_list=node_list
        self.elementTag = elementTag
        self.material = material
        self.load_direction = load_direction
        
        self._initialize_load_direction()
        
        self.Kg, self.B, self.F_fe_global, self.V = self.calculate_K0()
        self.index = self.calculate_indices()
        
    def __str__(self):
        """
        Returns a string representation of the Tet4 element.
        
        Returns:
            str: String representation of the element.
        """
        node_name_list=[node.name for node in self.node_list]
        return f'Tet4 {node_name_list}'
    
    def _initialize_load_direction(self):
        """
        Initializes the direction of the body force. If not specified, defaults to [0, 0].
        """
        if self.load_direction is None:
            self.load_direction = [0, 0, 0]
    
    def calculate_B_matrix(self):
        """
        Method to calculate the strain-displacement matrix, and the volume.
        These values are to be evaluated at each Gaussian point.

        Raises:
            ValueError: Display error when the Jacobian determinant is less than zero

        Returns:
            B (ndarray): Strain-displacement matrix
            V (float): Volume of the element
        """
        
        x1, y1, z1 = self.node_list[0].coord[0], self.node_list[0].coord[1], self.node_list[0].coord[2]
        x2, y2, z2 = self.node_list[1].coord[0], self.node_list[1].coord[1], self.node_list[1].coord[2]
        x3, y3, z3 = self.node_list[2].coord[0], self.node_list[2].coord[1], self.node_list[2].coord[2]
        x4, y4, z4 = self.node_list[3].coord[0], self.node_list[3].coord[1], self.node_list[3].coord[2]
        
        # Determinamos los coeficientes
        a1=y2*(z4-z3)-y3*(z4-z2)+y4*(z3-z2)
        a2=-y1*(z4-z3)+y3*(z4-z1)-y4*(z3-z1)
        a3=y1*(z4-z2)-y2*(z4-z1)+y4*(z2-z1)
        a4=-y1*(z3-z2)+y2*(z3-z1)-y3*(z2-z1)
        
        b1=-x2*(z4-z3)+x3*(z4-z2)-x4*(z3-z2)
        b2=x1*(z4-z3)-x3*(z4-z1)+x4*(z3-z1)
        b3=-x1*(z4-z2)+x2*(z4-z1)-x4*(z2-z1)
        b4=x1*(z3-z2)-x2*(z3-z1)+x3*(z2-z1)
        
        c1=x2*(y4-y3)-x3*(y4-y2)+x4*(y3-y2)
        c2=-x1*(y4-y3)+x3*(y4-y1)-x4*(y3-y1)
        c3=x1*(y4-y2)-x2*(y4-y1)+x4*(y2-y1)
        c4=-x1*(y3-y2)+x2*(y3-y1)-x3*(y2-y1)
        
        V=((x2-x1)*((y3-y1)*(z4-z1)-(y4-y1)*(z3-z1))+
           (y2-y1)*((x4-x1)*(z3-z1)-(x3-x1)*(z4-z1))+
           (z2-z1)*((x3-x1)*(y4-y1)-(x4-x1)*(y3-y1)))/6
        
        B=np.array([[a1,0,0,a2,0,0,a3,0,0,a4,0,0],
                    [0,b1,0,0,b2,0,0,b3,0,0,b4,0],
                    [0,0,c1,0,0,c2,0,0,c3,0,0,c4],
                    [b1,a1,0,b2,a2,0,b3,a3,0,b4,a4,0],
                    [0,c1,b1,0,c2,b2,0,c3,b3,0,c4,b4],
                    [c1,0,a1,c2,0,a2,c3,0,a3,c4,0,a4]]
                   )*(1/(6*V))
        
        if V <= 0:
            raise ValueError("The Jacobian determinant is less than or equal to zero, indicating an invalid or degenerate element.")
    
        return B, V
    
    def calculate_K0(self):
        """
        Calculates the element stiffness matrix, strain-displacement matrix, force vector, and volume.

        Returns:
            Ke (ndarray): Element stiffness matrix.
            B (ndarray): Strain-displacement matrix.
            fe (ndarray): Element force vector.
            V (float): Volume of the element.
        """
        
        nDof=self.nDof
        # Calculate the number of degrees of freedom per element
        nDof_element=len(self.node_list)*nDof
        
        C=self.material.Emat
        b=np.array(self.load_direction)
        b=b.reshape(-1, 1)
        b=np.tile(b,(4,1))
        
        B, V = self.calculate_B_matrix()
        
        # Calculo de la matriz de rigidez y vector de fuerzas
        Ke=V*(B.T@C@B)
        
        gamma=self.material.gamma
        W=V*gamma
        fe=b*W/4
        return Ke, B, fe, V
        
    def calculate_indices(self):
        """
        Calculates the indices of the degrees of freedom for the element.
        
        Returns:
            index (ndarray): Indices of the degrees of freedom.
        """
        index=np.hstack([node.index for node in self.node_list])
        return index
    
    def get_element_displacements(self, u):
        """
        Extracts the displacements of the element from the global displacement vector.
        
        Args:
            u (ndarray): Global displacement vector.
        
        Returns:
            ue (ndarray): Displacement vector of the element.
        """
        index = self.index
        ue = u[index]
        return ue
    
    def get_element_strains(self, u):
        """
        Calculates the strains in the element.
        
        Args:
            u (ndarray): Global displacement vector.
        
        Returns:
            epsilon_e (ndarray): Strain vector of the element.
            ue (ndarray): Displacement vector of the element.
        """
        ue = self.get_element_displacements(u)
        B=self.B
        epsilon_e = B @ ue
        return epsilon_e, ue
    
    def get_element_stress(self, u):
        """
        Calculates the stresses in the element.
        
        Args:
            u (ndarray): Global displacement vector.
        
        Returns:
            sigma_e (ndarray): Stress vector of the element.
            epsilon_e (ndarray): Strain vector of the element.
            ue (ndarray): Displacement vector of the element.
        """
        epsilon_e, ue = self.get_element_strains(u)
        sigma_e = self.material.Emat @ epsilon_e
        return sigma_e, epsilon_e, ue
    
    def set_results(self, stress, strain, displacement, principal_stress, principal_strain):
        """
        Sets the results of the analysis for the element.
        
        Args:
            stress (ndarray): Stress vector of the element.
            strain (ndarray): Strain vector of the element.
            displacement (ndarray): Displacement vector of the element.
            principal_stress (ndarray): Principal stresses of the element.
            principal_strain (ndarray): Principal strains of the element.
        """
        self.sigma = stress
        self.epsilon = strain
        self.displacement = displacement
        self.principal_stress = principal_stress
        self.principal_strain = principal_strain
        
    def calculate_principal_stress(self, sigma):
        """
        Calculates the principal stresses from the stress tensor.
        
        Args:
            sigma (ndarray): Stress tensor of the element.
        
        Returns:
            principal_stress (ndarray): Principal stresses of the element.
        """
        sx = sigma[0][0]
        sy = sigma[1][0]
        sxy = sigma[2][0]
        
        stress_matrix = np.array([[sx, sxy], [sxy, sy]])
        eigenvalues, _ = np.linalg.eig(stress_matrix)
        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_indices]
        sigma1, sigma2 = eigenvalues
        
        return np.array([[sigma1], [sigma2]])
    
    def calculate_principal_strain(self, epsilon):
        """
        Calculates the principal strains from the strain tensor.
        
        Args:
            epsilon (ndarray): Strain tensor of the element.
        
        Returns:
            principal_strain (ndarray): Principal strains of the element.
        """
        ex = epsilon[0][0]
        ey = epsilon[1][0]
        exy = epsilon[2][0]
        
        strain_matrix = np.array([[ex, exy], [exy, ey]])
        eigenvalues, _ = np.linalg.eig(strain_matrix)
        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_indices]
        epsilon1, epsilon2 = eigenvalues
        
        return np.array([[epsilon1], [epsilon2]])
    
    def element_visualization(self, offset=0):
        """
        Visualizes the quadrilateral element.
        
        Args:
            offset (float): Offset for the text labels. Default is 0.
        """
        fig, ax = plt.subplots()
        ax.set_aspect('equal', adjustable='box')

        # Plot nodes
        for n, node in enumerate(self.nodes):
            ax.plot(node[0], node[1], 'ko', ms=6)
            label = str(self.node_list[n].name)
            ax.text(node[0] + offset, node[1] + offset, label, fontsize=10)

        # Plot the element as a polygon using all nodes
        ax.add_patch(patches.Polygon(xy=self.nodes, edgecolor='black', facecolor='grey', alpha=0.30))
        ax.set_ylabel('Distance [m]')
        ax.set_xlabel('Distance [m]')
        ax.set_title('2D Element')
        ax.grid(True)
        
        plt.show()

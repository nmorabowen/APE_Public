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
from scipy.special import roots_legendre
import matplotlib.patches as     patches
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

class Brick:

    
    def __init__(self, elementTag, node_list, material, samplingPoints=5, load_direction=None, eval_points=[0,0,0], nDof=3):

        
        # Number of elements validation
        if len(node_list) != 8:
            raise ValueError("node_list must contain exactly 8 nodes.")
        
        self.nDof=nDof
        self.node_list=node_list
        self.elementTag = elementTag
        self.material = material
        self.samplingPoints = samplingPoints
        self.load_direction = load_direction
        self.eval_points=eval_points
        
        self.nodes = self.calculate_xy()
        self.C = material.Emat
        
        self._initialize_load_direction()
        
        self.Kg, self.V, self.F_fe_global = self.calculate_K0()
        self.index = self.calculate_indices()
        
    def __str__(self):
        node_name_list=[node.name for node in self.node_list]
        return f'Brick {node_name_list}'
    
    def _initialize_load_direction(self):
        """
        Initializes the direction of the body force. If not specified, defaults to [0, 0].
        """
        if self.load_direction is None:
            self.load_direction = [0, 0, 0]
    
    def calculate_xy(self):
        """
        Creates the list of node coordinates and their indices.
        
        Returns:
            nodes (ndarray): Array of node coordinates.
            nodes_idx (ndarray): Array of node indices.
        """
        xy=np.array([node.coord for node in self.node_list])
        
        return xy
    
    def calculate_interpolation_functions(self, zeta, eta, mu):
        """
        Calculates the interpolation functions and their partial derivatives for a quadrilateral element
        in natural coordinates (zeta, eta).

        Args:
            zeta (float): Natural coordinate corresponding to the zeta axis.
            eta (float): Natural coordinate corresponding to the eta axis.

        Returns:
            N (ndarray): Interpolation function matrix for the given natural coordinates.
            dNnatural (ndarray): Matrix of partial derivatives of the interpolation functions with respect to zeta and eta (2x4).

        """
        
        N1 = (1/8) * (1-zeta)*(1-eta)*(1-mu)
        N2 = (1/8) * (1+zeta)*(1+eta)*(1-mu)
        N3 = (1/8) * (1-zeta)*(1-eta)*(1+mu)
        N4 = (1/8) * (1+zeta)*(1+eta)*(1+mu)
        N5 = (1/8) * (1+zeta)*(1-eta)*(1-mu)
        N6 = (1/8) * (1-zeta)*(1+eta)*(1-mu)
        N7 = (1/8) * (1+zeta)*(1-eta)*(1+mu)
        N8 = (1/8) * (1-zeta)*(1+eta)*(1+mu)
        
        # Matriz de funciones de interpolacion
        N=np.array([
            [N1,0,0,N2,0,0,N3,0,0,N4,0,0,N5,0,0,N6,0,0,N7,0,0,N8,0,0],
            [0,N1,0,0,N2,0,0,N3,0,0,N4,0,0,N5,0,0,N6,0,0,N7,0,0,N8,0],
            [0,0,N1,0,0,N2,0,0,N3,0,0,N4,0,0,N5,0,0,N6,0,0,N7,0,0,N8]
        ])
        
        # Partial derivatives with respect to zeta
        dN1dzeta = (-0.125) * (1-eta)*(1-mu)
        dN2dzeta = (0.125) * (1-eta)*(1-mu)
        dN3dzeta = (0.125) * (1+eta)*(1-mu)
        dN4dzeta = (-0.125) * (1+eta)*(1-mu)
        dN5dzeta = (-0.125) * (1-eta)*(1+mu)
        dN6dzeta = (0.125) * (1-eta)*(1+mu)
        dN7dzeta = (0.125) * (1+eta)*(1-mu)
        dN8dzeta = (-0.125) * (1+eta)*(1-mu)
        
        # Partial derivatives with respect to eta
        dN1deta = (-0.125) * (1-zeta)*(1-mu)
        dN2deta = (-0.125) * (1+zeta)*(1-mu)
        dN3deta = (0.125) * (1+zeta)*(1-mu)
        dN4deta = (0.125) * (1-zeta)*(1-mu)
        dN5deta = (-0.125) * (1-zeta)*(1+mu)
        dN6deta = (-0.125) * (1+zeta)*(1+mu)
        dN7deta = (0.125) * (1+zeta)*(1+mu)
        dN8deta = (0.125) * (1-zeta)*(1+mu)
        
        # Partial derivatives with respect to mu
        dN1dmu = (-0.125) * (1-zeta)*(1-eta)
        dN2dmu = (-0.125) * (1+zeta)*(1-eta)
        dN3dmu = (-0.125) * (1+zeta)*(1+eta)
        dN4dmu = (-0.125) * (1-zeta)*(1+eta)
        dN5dmu = (0.125) * (1-zeta)*(1-eta)
        dN6dmu = (0.125) * (1+zeta)*(1-eta)
        dN7dmu = (0.125) * (1+zeta)*(1+eta)
        dN8dmu = (0.125) * (1-zeta)*(1+eta)
        
        # Derivada de N con respecto a eta y zeta
        dNnatural=np.array([
            [dN1dzeta, dN2dzeta, dN3dzeta, dN4dzeta, dN5dzeta, dN6dzeta, dN7dzeta, dN8dzeta],
            [dN1deta, dN2deta, dN3deta, dN4deta, dN5deta, dN6deta, dN7deta, dN8deta],
            [dN1dmu, dN2dmu, dN3dmu, dN4dmu, dN5dmu, dN6dmu, dN7dmu, dN8dmu]
        ])
        
        return N, dNnatural
    
    def transform_to_physical(self, zeta, eta, mu):
        N, _ = self.calculate_interpolation_functions(zeta, eta, mu)
        coordenadas_cartesianas = np.dot(N, self.nodes.flatten())
        return coordenadas_cartesianas
    
    def calculate_B_matrix(self,zeta,eta,mu):
        """
        Method to calculate the strain displacement matrix, the Jacobian and it determinant, and the interpolation matrix
        This values are to be evaluated at each Gaussian point

        Args:
            zeta (float): natural coordinate corresponding to a gausssian point
            eta (float): natural coordinate correponding to a gaussian point

        Raises:
            ValueError: Display error when the Jacobian determinate is less than zero

        Returns:
            B (ndarray): strain displacement matrix
            J (ndarray): Jacobian
            J_det (float): Jacobian determinant
            
        """
        
        # Determinamos la matriz de coordenadas xy
        xy=self.nodes
        
        N, dNnatural = self.calculate_interpolation_functions(zeta, eta, mu)
        
        # J=PX
        J=np.dot(dNnatural, xy)
        J_det = np.linalg.det(J)
        
        # Si el determinante es menor a zero la forma del elemento es inadecuada
        if J_det < 0:
            print('Jacobiano Negativo!')
        
        # Derivada de N con respecto a x y y
        # dNnatural = J x dNcartesian        
        dNcartesian=np.linalg.solve(J,dNnatural)
        
        B = np.zeros((6, 3 * len(xy)))
        B[0, 0::3] = dNcartesian[0, :]
        B[1, 1::3] = dNcartesian[1, :]
        B[2, 2::3] = dNcartesian[2, :]
        B[3, 0::3] = dNcartesian[1, :]
        B[3, 1::3] = dNcartesian[0, :]
        B[4, 1::3] = dNcartesian[2, :]
        B[4, 2::3] = dNcartesian[1, :]
        B[5, 0::3] = dNcartesian[2, :]
        B[5, 2::3] = dNcartesian[0, :]
        
        return B, J, J_det, N
    
    def calculate_K0(self):
        """
        Calculates the initial stiffness matrix and area of the element.
        
        Returns:
            Ke (ndarray): Stiffness matrix of the element.
            A (float): Area of the element.
            B (ndarray): Strain Displacement matrix.
        """
        nDof = self.nDof
        nDof_element = len(self.node_list) * nDof
        
        C = self.material.Emat
        sampling_points = self.samplingPoints
        roots, weights = roots_legendre(sampling_points)
        
        b = np.array(self.load_direction).reshape(-1, 1)
        
        V = 0
        Ke = np.zeros((nDof_element, nDof_element))
        fe = np.zeros((nDof_element, 1))

        for zeta, weight_r in zip(roots, weights):
            for eta, weight_s in zip(roots, weights):
                for mu, weight_t in zip(roots, weights):
                    B, _, J_det, N = self.calculate_B_matrix(zeta, eta, mu)
                    V += weight_r * weight_s * weight_t * np.abs(J_det)
                    Ke += weight_r * weight_s * weight_t * B.T @ C @ B * J_det
                    fe += weight_r * weight_s * weight_t * N.T @ b * J_det
        
        gamma = self.material.gamma
        fe = fe * gamma
        fe = fe.flatten()
        return Ke, V, fe
    
    def calculate_Ke_difference(self, sampling_point_i, sampling_point_j):
        Ke_i, _, _ = self.calculate_K0_for_sampling_points(sampling_point_i)
        Ke_j, _, _ = self.calculate_K0_for_sampling_points(sampling_point_j)
        delta_i_j = np.round(np.abs(np.divide(Ke_i - Ke_j, Ke_i)) * 100, 2)
        max_diff = np.max(delta_i_j)
        return delta_i_j, max_diff
        
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
        eval_point=self.eval_points
        B, _, _, _ = self.calculate_B_matrix(eval_point[0],eval_point[1])
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
        sx = sigma[0][0]
        sy = sigma[1][0]
        sz = sigma[2][0]
        sxy = sigma[3][0]
        syz = sigma[4][0]
        szx = sigma[5][0]
        
        stress_matrix = np.array([[sx, sxy, szx], [sxy, sy, syz], [szx, syz, sz]])
        eigenvalues, _ = np.linalg.eig(stress_matrix)
        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_indices]
        sigma1, sigma2, sigma3 = eigenvalues
        
        return np.array([[sigma1], [sigma2], [sigma3]])
    
    def calculate_principal_strain(self, epsilon):
        ex = epsilon[0][0]
        ey = epsilon[1][0]
        ez = epsilon[2][0]
        exy = epsilon[3][0]
        eyz = epsilon[4][0]
        ezx = epsilon[5][0]
        
        strain_matrix = np.array([[ex, exy, ezx], [exy, ey, eyz], [ezx, eyz, ez]])
        eigenvalues, _ = np.linalg.eig(strain_matrix)
        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_indices]
        epsilon1, epsilon2, epsilon3 = eigenvalues
        
        return np.array([[epsilon1], [epsilon2], [epsilon3]])
    
    def element_visualization(self, offset=0):
        """
        Visualizes the hexahedral element.
        
        Args:
            offset (float): Offset for the text labels. Default is 0.
        """
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_aspect('auto')
        
        # Plot nodes
        for n, node in enumerate(self.nodes):
            ax.scatter(node[0], node[1], node[2], c='k', marker='o')
            label = str(self.node_list[n].name)
            ax.text(node[0] + offset, node[1] + offset, node[2] + offset, label, fontsize=10)
        
        # Sampling points for Gaussian quadrature
        sampling_points = self.samplingPoints
        roots, _ = roots_legendre(sampling_points)
        
        # Transform sampling points to physical coordinates
        for r in roots:
            for s in roots:
                for t in roots:
                    coord_cartesianas = self.transform_to_physical(r, s, t)
                    x_physical = coord_cartesianas[0]
                    y_physical = coord_cartesianas[1]
                    z_physical = coord_cartesianas[2]
                    ax.scatter(x_physical, y_physical, z_physical, c='r', marker='o')  # Plot sampling points in red for distinction
        
        # Plot the element as a polyhedron using the correct node order
        verts = [
            [self.nodes[0], self.nodes[1], self.nodes[5], self.nodes[4]],  # bottom face
            [self.nodes[1], self.nodes[2], self.nodes[6], self.nodes[5]],  # front face
            [self.nodes[2], self.nodes[3], self.nodes[7], self.nodes[6]],  # top face
            [self.nodes[3], self.nodes[0], self.nodes[4], self.nodes[7]],  # back face
            [self.nodes[0], self.nodes[1], self.nodes[2], self.nodes[3]],  # left face
            [self.nodes[4], self.nodes[5], self.nodes[6], self.nodes[7]]   # right face
        ]
        ax.add_collection3d(Poly3DCollection(verts, facecolors='gray', linewidths=1, edgecolors='black', alpha=0.30))
        
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.set_zlabel('Z [m]')
        ax.set_title('3D Hexahedral Element')
        plt.show()

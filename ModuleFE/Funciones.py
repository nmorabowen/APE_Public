
"""
Description:
    Este archivo contiene las funciones de apoyo requeridas por el modulo de elementos finitos.

Date:
    2024-06-12
"""

__author__ = "Nicolás Mora Bowen"
__version__ = "1.0.0"

# ====================================
# Dependencias
# ====================================

from gmshtools import get_elements_and_nodes_in_physical_group
from gmshtools import get_physical_groups_map
import numpy as np
import tkinter as tk
from tkinter import filedialog

# ====================================
# Funciones
# ====================================

def matrixExtract(matrix,rowsIDX,colsIDX):
    """ Funcion para extraer valores de matrices con indices

    Args:
        matrix (ndarray): Matriz principal
        rowsIDX (ndarray): _description_
        colsIDX (ndarray): _description_

    Returns:
        ndarray: Matriz con los valores buscados
    """
    # Realizamos la indexacion requerida por python para extraer la informacion
    cols=np.tile(colsIDX,(len(rowsIDX),1))
    rows=np.tile(rowsIDX[:,np.newaxis],(1,len(colsIDX)))
    
    matrixMOD=matrix[rows,cols]
    
    return matrixMOD

def matrixReplacement(matrix,matrixADD,rowsIDX,colsIDX):
    """ Funcion para sumar submatrices en la matriz principal

    Args:
        matrix (ndarray): _description_
        matrixADD (ndarray): _description_
        rowsIDX (ndarray): _description_
        colsIDX (ndarray): _description_

    Returns:
        ndarray: La nueva matriz con la submatriz sumada
    """
    # Funcion para extraer valores de matrices con indices
    cols=np.tile(colsIDX,(len(rowsIDX),1))
    rows=np.tile(rowsIDX[:,np.newaxis],(1,len(colsIDX)))
    
    matrix[rows,cols]=matrix[rows,cols]+matrixADD
    
    return matrix

def filter_nodes(physical_group_name, model, node_tags):
    # Obtenemos la informacion de los nudos restringidos
    _, element_nodes_list, _, _ = get_elements_and_nodes_in_physical_group(physical_group_name, model)

    # filtramos la informacion para obtener un vector con los nudos restringidos
    nodes=np.array(element_nodes_list)
    nodes=nodes.reshape(1,-1)
    nodes=nodes.flatten()
    nodes=np.unique(nodes)

    #print(f'Los nudos(gmsh) son: {nodes}')

    # Buscamos el indice del nudo, es decir las posiciones en el vector de nudos donde estan los objetos que buscamos restringir
    nodes_index_values=np.zeros(len(nodes))
    for index, node in enumerate(nodes):
        nodes_index_values[index]=np.where(node_tags == node)[0]
        
    #print(f'Los indices de los nudos restringidos son: {nodes_index_values}')
    
    return nodes, nodes_index_values

def calculo_cargas_superficie(list,wj,wk,nodes,node_tags,alpha_degree=None, printValue=False):
    """Calculo de un vector de cargas globales para los nudos

    Args:
        list (_type_): Lista 1x2 con el nombre de gmsh del nudo inicial y el nudo final
        wj (_type_): Carga aplicada en el nudo inicial
        wk (_type_): Carga aplicada en el nudo final
        alpha_degree (_type_): Angulo de la carga respecto a la inclinacion local, si se deja en None se calcula la proyeccion vertical
        nodes (_type_): Lista de nudos
        node_tags (_type_): Lista de los tags de gmsh para los nudos

    Returns:
        F_global (np.array): Vector de cargas globales para los dos nudos [1x4]
    """
    # Determinamos el indice correspondiente al nombre de gmsh para la lista de dos nudos
    node_index=np.array([np.where(node_tags==list[0])[0][0], np.where(node_tags==list[1])[0][0]])
    
    # Seleccionamos los nudos con los que se va a trabajar
    n0=nodes[node_index[0],0]
    n1=nodes[node_index[1],0]
    
    delta_n=n1.coord-n0.coord
    L=(delta_n[0]**2+delta_n[1]**2)**0.5
    
    theta=np.arctan2(delta_n[1], delta_n[0])
    
    # Determinamos el angulo de aplicacion de la carga
    if alpha_degree is None:
        alpha=np.radians(90)-theta
    else:
        alpha=np.radians(alpha_degree)
    
    w1=wk
    w2=wj-wk

    wjx=-w1*np.cos(alpha)*L/2-w2*np.cos(alpha)*L/3
    wkx=-w1*np.cos(alpha)*L/2-w2*np.cos(alpha)*L/6
    wjy=-w1*np.sin(alpha)*L/2-w2*np.sin(alpha)*L/3
    wky=-w1*np.sin(alpha)*L/2-w2*np.sin(alpha)*L/6
    
    # Cambiamos el signo ya que las cargas anteriores corresponden a las reacciones
    F_local=-np.array([[wjx],
                       [wjy],
                       [wkx],
                       [wky]])

    c=np.cos(theta)
    s=np.sin(theta)
    
    Tlg=np.array([[c,s,0,0],
                  [-s,c,0,0],
                  [0,0,c,s],
                  [0,0,-s,c]])

    F_global=Tlg.T@F_local
    
    if printValue is True:
        print(f'La longitud para el calculo es: {L}')
        print(f'Los valores de la carga son {w1} y {w2} con un theta {np.degrees(theta)}')
        print(f'El vector de cargas de empotramiento local es: {np.round(F_local,3)}')
        print(f'El vector de cargas de empotramiento es: {np.round(F_global,3)}')
    
    return F_global

def set_cargas_superficie(nodes, node_tags, list, F_global):
    
    # Determinamos el indice correspondiente al nombre de gmsh para la lista de dos nudos
    node_index=np.array([np.where(node_tags==list[0])[0][0], np.where(node_tags==list[1])[0][0]])
    n0=nodes[node_index[0],0]
    n1=nodes[node_index[1],0]
    
    # Aplicamos las cargas en los nudos
    n0.apply_load([F_global[0,0], F_global[1,0]])
    n1.apply_load([F_global[2,0], F_global[3,0]])
    
    return node_index, n0, n1

def find_node_index(node_list, node_tags):
    # encontrar el indice de un nudo en la lista node_tags
    index_values_list=np.zeros(len(node_list), dtype=int)
    for i,node in enumerate(node_list):
        index_values_list[i]=np.where(node_tags==node)[0][0]
    return index_values_list

def select_file():
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    file_selected = filedialog.askopenfilename()
    if not file_selected:
        print("No folder selected. Exiting...")
        sys.exit(1)  # Exit the script with a non-zero exit code
    return file_selected
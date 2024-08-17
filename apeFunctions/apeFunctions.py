import os
import numpy as np
import tkinter as tk
from tkinter import filedialog
import sys

# Function to check if a file exists and delete it
def check_and_delete_file(file_path):
    # Check if the given path is just a file name
    if not os.path.isabs(file_path):
        # Prepend the current working directory to the file name
        file_path = os.path.join(os.getcwd(), file_path)

    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"Deleted existing file: {file_path}")
        
def select_file():
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    file_selected = filedialog.askopenfilename()
    if not file_selected:
        print("No folder selected. Exiting...")
        sys.exit(1)  # Exit the script with a non-zero exit code
    return file_selected

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
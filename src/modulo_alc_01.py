# Funciones del Laboratorio 01 que se piden en la entrega.
# Para esta entrega, sólo las funciones del Módulo ALC.

import numpy as np

def error(x, y) -> float :
    
    x = np.float64(x) 
    y = np.float64(y) 
    
    return np.abs(x - y) 

def error_relativo(x, y, tol = 1e-15) -> float :

    x = np.float64(x) 
    y = np.float64(y) 
    
    if abs(x) <= tol :
        return np.abs(y)
    
    return np.abs(x - y) / np.abs(x)

def matricesIguales(A, B, tol = 1e-12) -> bool :
    
    if (len(A) != len(B)) :
        return False
    
    if (any(len(A[i]) != len(B[i]) for i in range(len(A)))) : # Miro que todas las filas sean de igual tamaño.
        return False

    for i in range(len(A)) :
        for j in range(len(A[i])) :
            if (error(A[i][j], B[i][j]) > tol) :   # Uso la función 'error()' que implemente antes.
                return False
    return True

# End.
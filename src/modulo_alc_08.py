# Funciones del Laboratorio 08 - Funciones del Modulo ALC.
# Para esta entrega, (... aún no confirmaron las funciones a entregar ...)

import numpy as np
from funciones_generales import multiplicar_matrices, traspuesta, normaliza_columnas, calcularAx
from modulo_alc_06 import diagRH


def svd_reducida(A:np.ndarray, k = "max", tol:float = 1e-15, reducida:bool = True) : 

    filas, columnas = A.shape
    
    trans = False
    if (filas < columnas) :
        A = traspuesta(A)
        trans = True
        filas, columnas = A.shape   # Actualizo las variables de dimension. 
        
    B = multiplicar_matrices(traspuesta(A), A)
    
    V, D = diagRH(B, tol)
    
    valores_singulares:list[float] = []   # No nulos.
    
    for i in range(len(D)) :
        
        if (D[i, i] <= tol) :
            break
           
        valores_singulares.append(np.sqrt(D[i, i]))

    if k == "max":
        k = len(valores_singulares)
    else:
        k = min(k, len(valores_singulares))
    
    # Si es reducida, recortar
    if reducida:
        valores_singulares = valores_singulares[:k]
        V = V[:, :k]
    
    V = normaliza_columnas(V, 2)
    U = np.zeros((filas, filas if not reducida else k), dtype=np.float64)
    
    for i in range(k):
        
        if valores_singulares[i] <= tol:
            U[:, i] = 0
            
        else:
            U[:, i] = calcularAx(A, V[:, i], vector_fila=True) / valores_singulares[i]
            
    U = normaliza_columnas(U, 2)   # Normalizamos (para que quede unitaria). 
    
    if (trans) :
        
        return V, valores_singulares, U
    
    else:
        
        return U, valores_singulares, V


# End.
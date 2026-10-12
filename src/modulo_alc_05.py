# Funciones del Laboratorio 05 - Funciones del Modulo ALC.
# Para esta entrega, (... aún no confirmaron las funciones a entregar ...)

import numpy as np
from funciones_generales import esCuadrada, producto_interno, multiplicar_matrices, traspuesta
from modulo_alc_03 import norma


def QR_con_GS(A,tol=1e-12,retorna_nops=False):
    
    # A debe ser cuadrada.
    if (not esCuadrada(A)) :
        return None

    A = np.array(A, dtype=np.float64)
    n:int = A.shape[0]
    Q:np.ndarray = np.zeros((n, n), dtype=np.float64)
    R:np.ndarray = np.zeros((n, n), dtype=np.float64)

    nops:int = 0  # Contador de operaciones.

    for j in range(0, n) :
            
        v:np.ndarray = np.copy(A[:, j])  # Columna j.

        for i in range(0, j) :
            qi = Q[:, i]
                
            r_ij = producto_interno(qi, v) 
            R[i, j] = r_ij
            v = v - r_ij * qi
                
            # Contamos Operaciones: producto interno ('n' multiplicaciones + 'n - 1' sumas), escala y resta.
            nops += (2 * n - 1) + n + n

        r_jj = norma(v, 2)
            
        if (r_jj > tol) :
                
            Q[:, j] = v / r_jj
            R[j, j] = r_jj
                
            # Contamos las operaciones de la normalización: 'n' multiplicaciones + 'n' divisiones.
            nops += n + n
                
        else :
                
            Q[:, j] = 0.0
            R[j, j] = 0.0

    if (retorna_nops) :
        return Q, R, nops
        
    return Q, R 


def QR_con_HH(A,tol=1e-12,extras=False):
    return None


def calculaQR(A,metodo='RH',tol=1e-12):
    return None


# End.
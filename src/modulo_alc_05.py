# Funciones del Laboratorio 05 - Funciones del Modulo ALC.
# Para esta entrega, (... aún no confirmaron las funciones a entregar ...)

import numpy as np
from funciones_generales import esCuadrada, producto_interno, multiplicar_matrices, traspuesta
from modulo_alc_03 import norma


def QR_con_GS(A, tol=1e-12, retorna_nops=False):
    
    # A debe ser cuadrada.
    if (not esCuadrada(A)) :
        return None

    #? A = np.array(A, dtype=np.float64)     -> Si pido por especificación un array de Numpy, no hace falta esto.

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


def QR_con_HH(A, tol=1e-12, extras=False):

    #? A = np.array(A, dtype=np.float64)     -> Si pido por especificación un array de Numpy, no hace falta esto.

    m, n = A.shape
    
    if (m < n) :
        return None

    R = np.copy(A)
    Q = np.eye(m, dtype=np.float64)

    if (extras) :
        extra_info = {'R_matrices':[], 
                      'Q_matrices':[]}

    for k in range(0, n) :
        
        x = R[k:, k].copy()
        norm_x = norma(x, 2)
        
        if (norm_x < tol) :
            continue

        # Atajo el caso cuando x[0] == 0 (la función 'np.sign()' devuelve 0 y no quiero eso porque me cancela todo).
        if (abs(x[0]) < tol) :
            alpha = - norm_x
        else:
            alpha = - np.sign(x[0]) * norm_x
        
        # Armo el canónico.
        e1 = np.zeros_like(x)
        e1[0] = 1.0
        
        # Ahora construyo 'u'
        u = x - alpha * e1
        norm_u = norma(u, 2) 
        
        if (norm_u < tol) :
            continue
        
        u = u / norm_u

        # Hk = I - 2 u u^T
        u_col = np.array([[ui] for ui in u], dtype=np.float64)
        u_row = np.array([u], dtype=np.float64)
        uuT = multiplicar_matrices(u_col, u_row)
        Hk = np.eye(len(u), dtype=np.float64) - 2.0 * uuT
        
        # Es este último bloque (de arriba) no puedo utilizar 'multiplicar_matrices()' de una, porque le voy a estar pasando 
        # dos vectores (que para numpy tienen dimensión 1). Entonces, al desempaquetar en dos variables 'np.shape()' (esto es 
        # una parte clave de 'multiplicar_matrices()') se rompe porque numpy interpreta los vectores como de dimensión 1. 
        # Para evitar este problema, fuerzo las dimensiones construyendo las matrices a mano.

        # Extiendo a H̃k en dimensión 'm' y le enchufo Hk donde corresponde.
        H_tilde = np.eye(m)
        H_tilde[k:, k:] = Hk

        # Actualizo R y Q.
        R = multiplicar_matrices(H_tilde, R)
        Q = multiplicar_matrices(Q, traspuesta(H_tilde))

        if (extras) :
            extra_info['R_matrices'].append(np.copy(R))
            extra_info['Q_matrices'].append(np.copy(Q))

    # Para ser consistente con los test (y evitar dolores de cabeza), "limpio" la parte nula de la matriz.
    # La idea es que no me queden residuos de números muy pequeños QUE NO SON CERO (pero que los tomamos como tal por lo 
    # pequeño que son).
    R[np.abs(R) < tol] = 0.0

    if (extras) :
        return Q, R, extra_info

    return Q, R 


def calculaQR(A, metodo='RH', tol=1e-12):

    if (not esCuadrada(A)) :
        return None

    if (metodo == 'GS') :
        return QR_con_GS(A, tol = tol)
    
    elif (metodo == 'RH') :
        return QR_con_HH(A, tol = tol)
    
    else :
        return None


# End.
# Funciones del Laboratorio 06 - Funciones del Modulo ALC.
# Para esta entrega, (... aún no confirmaron las funciones a entregar ...)

import numpy as np
from funciones_generales import calcularAx, producto_interno, esSimetrica, multiplicar_matrices, traspuesta
from modulo_alc_03 import norma


def metpot2k(A:np.ndarray, tol:float = 1e-15, K:int = 1000) : 
    
    A = np.asarray(A, dtype=np.float64)
    filas, columnas = np.shape(A) 
    
    # Checkeo que la matriz sea cuadrada.
    if (filas != columnas) : 
        return None 
    
    # Genero un vector aleatorio (proveniente de una distribución normal).
    v:np.ndarray = np.random.rand(filas).astype(np.float64)
    
    # Definición de fA función auxiliar.
    def fA(vector:np.ndarray) : 
        
        Av = calcularAx(A, vector, vector_fila = True)
        
        norma_Av = norma(Av, 2)
        
        if (norma_Av < tol) :
            return np.zeros_like(Av)
        
        return (Av / norma_Av) 
    
    # Aplico dos veces la matriz A (la transformación 'f') al vector 'v'.
    v_tilde:np.ndarray = fA(fA(v)) 
    
    # Acá me dicen que haga 'transpuesto(v_virulete) * v', es lo mismo que hacer producto interno entre ambos. 
    e:float = producto_interno(v_tilde, v) 
    
    iteraciones:int = 0 
    
    while ((abs(e - 1) > tol) and (iteraciones < K)) : 
        v = v_tilde 
        v_tilde = fA(fA(v)) 
        e = producto_interno(v_tilde, v) 
        iteraciones += 1 
    
    Av_tilde:np.ndarray = calcularAx(A, v_tilde, vector_fila = True) 
    
    autovalor:float = producto_interno(v_tilde, Av_tilde) 
    
    return v_tilde, autovalor, iteraciones 


def diagRH(A: np.ndarray, tol: float = 1e-15, K: int = 1000) :

    # Verificación de simetría (numérica).
    if (not esSimetrica(A, atol=1e-8)) :
        return None, None

    n = A.shape[0]

    # Caso base 1 -> 'n = 1'.
    if (n == 1) :
        return np.array([[1.0]], dtype = np.float64), np.array([[A[0, 0]]], dtype = np.float64)

    # Primer AutoVector y AutoValor por Método de la Potencia.
    v1, l1, _ = metpot2k(A, tol, K)

    # Construyo el Reflector de Householder: H = I - 2 * (u u^T) / (u^T u)
    e1 = np.zeros_like(v1)
    e1[0] = 1.0
    u = e1 - v1
    denom = producto_interno(u, u)

    if (denom < tol) :   # No hay que dividir por cero...
        H_v1 = np.eye(n)
        
    else:
        u = u.reshape(-1, 1)   # vector columna
        H_v1 = np.eye(n) - 2.0 * multiplicar_matrices(u, traspuesta(u)) / denom

    # Construyo B = H * A * H^T
    B = multiplicar_matrices(H_v1, multiplicar_matrices(A, traspuesta(H_v1)))
    B[np.abs(B) < tol] = 0.0   # Limpio números que tomamos como cero (lo hardcodeo como cero).

    # Caso base 2 -> 'n = 2'.
    if (n == 2) :
        return H_v1, B

    # Paso recursivo.
    A_tilde = B[1:, 1:]
    S_tilde, D_tilde = diagRH(A_tilde, tol, K)

    # Construcción de D.
    D = np.eye(n)
    D[0, 0] = l1
    D[1:, 1:] = D_tilde

    # Construcción de S.
    S = np.eye(n)
    S[1:, 1:] = S_tilde
    S = multiplicar_matrices(H_v1, S)

    return S, D


# End.
# Funciones del Laboratorio 04 que se piden en la entrega.
# Para esta entrega, (... aún no confirmaron las funciones a entregar ...)

import numpy as np
from funciones_generales import multiplicar_matrices, esSimetrica

#! No pasa el Test del Servidor de Entregas: El código funciona pero no cumple el enunciado.
#! Por ahora, me dijeron que ignore el servidor y si corre los test en local, estoy bien.
def calculaLU(A: np.ndarray) :
    
    # Validaciones iniciales
    if (A is None) :
        return None, None, 0
    
    if (len(np.shape(A)) != 2) :
        return None, None, 0
    
    filas, columnas = np.shape(A)
    
    if (filas != columnas) :
        return None, None, 0

    U = np.copy(A).astype(np.float64)
    L = np.eye(filas, dtype = np.float64)
    nops = 0

    for k in range(filas - 1) : 

        pivote = U[k, k]
        
        if (pivote == 0.0) :
            return None, None, 0  # No factorizable sin pivoteo.

        for i in range(k + 1, filas) :
            mult = U[i, k] / pivote
            L[i, k] = mult
            nops += 1  # División

            # Guardamos la fila de U[k, :] para reducir accesos y ahorrar tiempo. 
            Uk_fila = U[k, k:filas]
            Ui_fila = U[i, k:filas]

            for j in range(filas - k) :
                Ui_fila[j] = Ui_fila[j] - mult * Uk_fila[j] 
                
                if (j != 0) :
                    nops += 2  # mult + resta

    return L, U, nops 

def res_tri(L:np.ndarray, b:np.ndarray, inferior: bool = True) :
    
    if (L is None or b is None) :
        return None

    filas, columnas = np.shape(L)
    
    if (filas != columnas or b.shape[0] != filas) :
        return None

    res = np.zeros(filas, dtype = np.float64)

    if (inferior) :
        # Algoritmo de Forward Substitution. 
        for i in range(filas) : 
            Li_fila = L[i, :] 
            suma = 0.0
            
            for j in range(i) : 
                suma += Li_fila[j] * res[j] 
                
            diag = Li_fila[i] 
            
            if (diag == 0.0) :   # Ojo con divididr por cero. 
                return None
            
            res[i] = (b[i] - suma) / diag 
            
    else:
        # Algoritmo de Backward Substitution.
        for i in range(filas - 1, -1, -1) :
            Li_fila = L[i, :]
            suma = 0.0
            
            for j in range(i + 1, filas) :
                suma += Li_fila[j] * res[j] 
                
            diag = Li_fila[i]
            
            if (diag == 0.0) :   # Ojo con dividir por cero (otra vez).
                return None
            
            res[i] = (b[i] - suma) / diag

    return res 

def inversa(A: np.ndarray) : 
    
    if (A is None) :
        return None
    
    if (len(np.shape(A)) != 2) :
        return None 
    
    filas, columnas = np.shape(A)
    
    if (filas != columnas) :
        return None

    # Factorización A = L U.
    L, U, nops = calculaLU(A) 
    
    if ((L is None) or (U is None)) :
        return None

    # Construyo identidad y matriz resultado. 
    identidad = np.eye(filas, dtype = np.float64)
    res = np.zeros((filas, filas), dtype = np.float64)

    # A ver si me sale ahorrar memoria...
    for i in range(filas) :
        e = identidad[:, i]
        y = res_tri(L, e, inferior = True) 
        
        if (y is None) :
            return None
        
        x = res_tri(U, y, inferior=False)
        
        if (x is None) :
            return None 
        
        res[:, i] = x

    return res 

def calculaLDV(A: np.ndarray) : 
    
    # Validaciones iniciales.
    if (A is None) :
        return None, None, None 
    
    if (len(np.shape(A)) != 2) :
        return None, None, None 
    
    filas, columnas = np.shape(A)
    
    if (filas != columnas) :
        return None, None, None 

    # Factorizo A = L * U.
    L, U, nops = calculaLU(A) 
    
    if ((L is None) or (U is None)) :
        return None, None, None 

    # Construyo D diagonal (solo con los pivotes de U).
    D = np.zeros((filas, filas), dtype = np.float64)
    
    for i in range(filas) :
        Dii = U[i, i]
        
        if (Dii == 0.0) :
            return None, None, None 
        
        D[i, i] = Dii

    # Construyo V = U -> Cada fila dividida por su pivote, que es el elemento que estoy sacando para armar la matriz D.
    V = np.zeros_like(U, dtype = np.float64)
    
    for i in range(filas) :
        piv = D[i, i]
        
        if (piv == 0.0) :
            return None, None, None 
        
        Ui_fila = U[i, :]
        Vi_fila = V[i, :]
        for j in range(filas) :
            Vi_fila[j] = Ui_fila[j] / piv

    return L, D, V 

#! No pasa el Test del Servidor de Entregas: No se pueden correr los tests.
#! Por ahora, me dijeron que ignore el servidor y si corre los test en local, estoy bien.
def esSDP(A:np.ndarray, atol:float = 1e-8) -> bool :
    
    if (A is None) :
        return False
    
    if (len(np.shape(A)) != 2) :
        return False
    
    filas, columnas = np.shape(A)
    
    if (filas != columnas) :
        return False

    # Verifico simetría.
    if (not esSimetrica(A, atol = atol)) :
        return False

    # Factorización A = L D V.
    L, D, V = calculaLDV(A)
    
    if ((L is None) or (D is None) or (V is None)) :
        return False

    # Reviso positividad de D (diagonal).
    for i in range(filas) :
        if (D[i, i] < atol) :
            return False

    return True 

def calculaCholesky(A:np.ndarray, atol=1e-10) -> np.ndarray :

    # Verificaciones iniciales.
    if (A is None) :
        return None

    filas, columnas = np.shape(A)
    if (filas != columnas) :
        return None

    if (not esSDP(A)) :
        return None

    L, D, V = calculaLDV(A)

    D = np.sqrt(D)

    res:np.ndarray = multiplicar_matrices(L, D)

    return res


# End.
import numpy as np

def error(x, y) :
    
    x = float(x) 
    y = float(y) 
    
    return abs(x - y) 

def error_relativo(x, y) :

    x = float(x) 
    y = float(y) 
    
    if (abs(x) <= 1e-15) :
        return abs(y)
    
    return abs(x - y) / abs(x)
    
def esSimetrica(matriz, atol=1e-5) :
    
    # Verifico si es cuadrada y algunas cositas más (que me llevaron a errores con el tiempo).
    if (matriz is None) :
        return False
    
    if (len(np.shape(matriz)) != 2) :
        return False
    
    filas, columnas = np.shape(matriz)
    
    if (filas != columnas) :
        return False

    # Compara solo la mitad superior (para ahorrar tiempo).
    for i in range(filas) :
        for j in range(i + 1, columnas) :
            if (error(matriz[i, j], matriz[j, i]) > atol) or (error_relativo(matriz[i, j], matriz[j, i]) > atol) :
                return False
            
    return True
    
def calculaLU(A) :
    
    # Validaciones iniciales
    if (A is None) :
        return None, None, 0
    
    if (len(np.shape(A)) != 2) :
        return None, None, 0
    
    filas, columnas = np.shape(A)
    
    if (filas != columnas) :
        return None, None, 0

    U = A.copy()
    L = np.eye(filas, dtype = float)
    nops = 0

    for k in range(filas - 1) : 

        pivote = U[k, k]
        
        if (pivote < 1e-5) :
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
    
def calculaLDV(A) : 
    
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
    D = np.zeros((filas, filas), dtype = float)
    
    for i in range(filas) :
        Dii = U[i, i]
        
        if (Dii < 1e-5) :
            return None, None, None 
        
        D[i, i] = Dii

    # Construyo V = U -> Cada fila dividida por su pivote, que es el elemento que estoy sacando para armar la matriz D.
    V = np.zeros_like(U, dtype = float)
    
    for i in range(filas) :
        piv = D[i, i]
        
        if (piv < 1e-5) :
            return None, None, None 
        
        Ui_fila = U[i, :]
        Vi_fila = V[i, :]
        for j in range(filas) :
            Vi_fila[j] = Ui_fila[j] / piv

    return L, D, V 
    
def esSDP(A, atol:float = 1e-5) -> bool :
    
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
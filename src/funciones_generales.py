# Funciones de Uso General.

import numpy as np

def esCuadrada(matriz:np.ndarray) -> bool : 
    if (matriz.shape[0] == 0) :
        return False
    return (matriz.shape[0] == matriz.shape[1])

def diagonal(matriz:np.ndarray) -> np.ndarray : 
    if not(esCuadrada(matriz)) :
        print("La matriz no es cuadrada, no puede definirse una diagonal principal.") 
        return matriz 
    else :
        res:np.ndarray = np.zeros(matriz.shape)
        nro_filas:int = matriz.shape[0]

        for i in range(0, nro_filas) :
            res[i][i] = matriz[i][i] 

        return res

def traza(matriz: np.ndarray) -> float:
    if not(esCuadrada(matriz)):
        print("La matriz no es cuadrada, no puede definirse una diagonal principal")
        return 0
    
    nro_filas = np.shape(matriz)[0]
    res = 0.0
    
    for i in range(nro_filas):
        res += matriz[i][i]
    
    return np.float64(res)

def traspuesta(matriz:np.ndarray) -> np.ndarray :

    # Caso 1: vector 1D -> lo tratamos como matriz fila (1 × n).
    if (matriz.ndim == 1):
        n = matriz.shape[0]
        res = np.zeros((n, 1), dtype=np.float64)
        res[:, 0] = matriz[:]
        return res

    # Caso general: matriz 2D.
    filas, columnas = np.shape(matriz)
    res = np.zeros((columnas, filas), dtype = np.float64)

    for i in range(filas) :
        res[:, i] = matriz[i, :]

    return res 

def esSimetrica(matriz:np.ndarray, atol: float = 1e-12) -> bool :

    from modulo_alc_01 import error, error_relativo
    
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

def multiplicar_matrices(A:np.ndarray, B:np.ndarray) -> np.ndarray :
    # Si B es vector 1D, lo convierto en columna
    if B.ndim == 1:
        B = B.reshape(-1, 1)
        
    filas_A, cols_A = A.shape
    filas_B, cols_B = B.shape
    
    # Verifico compatibilidad.
    if (cols_A != filas_B) :
        raise ValueError("Las dimensiones de las matrices no son compatibles para la multiplicación.")
    
    C = np.zeros((filas_A, cols_B), dtype = np.float64)
    
    for i in range(0, filas_A) :
        for j in range(0, cols_B) :
            suma:float = np.float64(0.0) 
            for k in range(0, cols_A) :
                suma += np.float64(A[i, k] * B[k, j])
            C[i, j] = suma 
    
    return C 

def calcularAx(A:np.ndarray, x:np.ndarray, vector_fila:bool = False) -> np.ndarray:

    n_filas, n_cols = np.shape(A)
    forma_x = np.shape(x)

    # Normalizo x a vector 1D.
    if (len(forma_x) == 2) :
        f, c = forma_x
        
        if (c == 1) :          # vector columna
            x = x[:, 0]
            
        elif (f == 1) :        # vector fila
            x = x[0, :]
            
        else:
            raise ValueError("La 'matriz_x' no es un vector válido.")
            
    elif (len(forma_x) != 1) :
        raise ValueError("La 'matriz_x' no es un vector válido.")

    # Verifico de compatibilidad.
    if (len(x) != n_cols) :
        raise ValueError("Dimensiones incompatibles para la multiplicación A * x.")

    # Resultado (vector columna)
    res = np.zeros((n_filas, 1), dtype = np.float64)

    for i in range(n_filas) :
        fila_i = A[i] 
        suma = 0.0
        
        for j in range(n_cols) :
            suma += fila_i[j] * x[j] 
            
        res[i, 0] = suma 

    if (vector_fila) :
        res = res[:, 0] 

    return res 

def producto_interno(x1:np.ndarray, x2:np.ndarray) -> float : 
    
    if (len(x1) != len(x2)) : 
        raise ValueError("Las dimensiones de los vectores no son compatibles para el producto inetrno (no son iguales).")
    
    long_vectores:int = len(x1) 
    
    res:float = 0 
    
    for i in range(0, long_vectores) : 
        res += np.float64(x1[i] * x2[i]) 
        
    return res 

def producto_interno_Especial(x1: np.ndarray, x2: np.ndarray) -> float :

    # Producto interno entre vectores pasados como vectores columna.
    # La version anterior trabaja con vectores fila.

    x1 = np.array(x1, dtype=np.float64).reshape(-1)
    x2 = np.array(x2, dtype=np.float64).reshape(-1)

    if x1.shape != x2.shape:
        raise ValueError("Dimensiones incompatibles para producto interno.")

    res = 0.0
    for i in range(len(x1)):
        res += x1[i] * x2[i]

    return res

def normaliza_columnas(matriz:np.ndarray, p:int, tol:float = 1e-15) -> np.ndarray : 

    from modulo_alc_03 import norma

    # Recibe una matriz y retorna la matriz con sus columnas normalizadas.
    
    filas, columnas = matriz.shape
    res:list[list[float]] = [] 
    
    for i in range(0, columnas) :
        
        columna_actual = matriz[:,i] 
        norma_columna = norma(columna_actual, p) 
        
        if (norma_columna <= tol) : 
            columna_actual = np.zeros(filas, dtype = np.float64) 
            
        else : 
            columna_actual = columna_actual / norma_columna
            
        res.append(columna_actual) 
        
    return traspuesta(np.array(res, dtype = np.float64)) 

def triangSup(matriz:np.ndarray) -> np.ndarray :
    
    if not(esCuadrada(matriz)) :
        print("La matriz no es cuadrada, no puede definirse una diagonal principal.")
        return matriz 
    
    else :
        res:np.ndarray = matriz.copy() 
        nro_filas:int = res.shape[0]
        
        for i in range(0, nro_filas) :
            for j in range(0, i + 1):
                if (j <= i) :
                    res[i][j] = 0
                    
        return res 

def triangInf(matriz:np.ndarray) -> np.ndarray :
    
    if not(esCuadrada(matriz)) :
        print("La matriz no es cuadrada, no puede definirse una diagonal principal.") 
        return matriz 
    
    else :
        res:np.ndarray = matriz.copy() 
        nro_filas:int = res.shape[0] 
        nro_columnas:int = res.shape[1] 
        
        for i in range(0, nro_filas) :
            for j in range(i, nro_columnas) :
                res[i][j] = 0 
        
        return res 

# End.
# Alumno: Francisco Lautaro Pucciarelli
# LU: 802/22

import numpy as np

# Funciones Laboratorio 1 ------------------------------------------------------------------------------------------------------------------------------------------------------- #

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
    
    if (len(A) != len(B)) :   # Miro que tengan igual número de filas.
        return False
    
    if (any(len(A[i]) != len(B[i]) for i in range(len(A)))) : # Miro que todas las filas sean de igual tamaño (y que tengan igual cantidad de columnas).
        return False

    for i in range(len(A)) :
        for j in range(len(A[i])) :
            if (error(A[i][j], B[i][j]) > tol) :   # Uso la función 'error()' que implemente antes.
                return False
    return True


# Funciones Auxiliares para Laboratorio 2 --------------------------------------------------------------------------------------------------------------------------------------- #

def multiplicar_matrices(A:np.ndarray, B:np.ndarray) -> np.ndarray :
    # Si B es vector 1D, lo convierto en columna.
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


# Funciones Laboratorio 2 ------------------------------------------------------------------------------------------------------------------------------------------------------- #

def rota(theta:float) -> np.ndarray :
    
    c:float = np.cos(theta)
    s:float = np.sin(theta) 
    
    res:np.ndarray = np.array([[c, -s], 
                               [s, c]])
    
    return res 

def escala(s) -> np.ndarray :
    
    s = np.asarray(s, dtype = float)   # 's' es una "tira de números", por las dudas lo transformo en array de numpy.
    
    n:int = s.size
    
    res:np.ndarray = np.zeros((n, n) , dtype = float) 
    
    for i in range(0, len(s)) :
        res[i][i] = s[i]
    
    return res 

def rota_y_escala(theta:float, s) -> np.ndarray : 
    
    matriz_escala:np.ndarray = escala(s) 
    matriz_rotacion:np.ndarray = rota(theta) 
    
    res:np.ndarray = multiplicar_matrices(matriz_escala, matriz_rotacion) 
    
    return res 

def afin(theta:float, s, b) -> np.ndarray : 
    
    s = np.asarray(s, dtype = float) 
    b = np.asarray(b, dtype = float)
    
    m_rotar_escalar:np.ndarray = rota_y_escala(theta, s)
    
    res:np.ndarray = np.array([[m_rotar_escalar[0][0], m_rotar_escalar[0][1], b[0]], 
                               [m_rotar_escalar[1][0], m_rotar_escalar[1][1], b[1]], 
                               [0                    , 0                    , 1]])
    
    return res 

def trans_afin(v, theta:float, s, b) -> np.ndarray : 
    
    v = np.asarray(v, dtype = float) 
    s = np.asarray(s, dtype = float) 
    b = np.asarray(b, dtype = float)
    
    m_afin:np.ndarray = afin(theta, s, b) 
    v_extendido:np.ndarray = np.array([v[0], v[1], 1.0])
    
    res_aux:np.ndarray = calcularAx(m_afin, v_extendido, vector_fila=True) 
    res:np.ndarray = np.array([res_aux[0], res_aux[1]])
    
    return res 


# End.
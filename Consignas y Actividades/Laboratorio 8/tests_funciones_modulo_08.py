# Tests L08

'''
No supe cómo automatizar estos tests en un archivo para correr con Pytest.
Así que pego todas las funciones y corro el archivo como está.
'''

import numpy as np

def error(x, y) -> float :
    
    x = np.float64(x) 
    y = np.float64(y) 
    
    return np.abs(x - y) 

def error_relativo(x, y) -> float :

    x = np.float64(x) 
    y = np.float64(y) 
    
    if (abs(x) <= 1e-15) :
        return np.abs(y)
    
    return np.abs(x - y) / np.abs(x)

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

def normaliza_columnas(matriz:np.ndarray, p:int, tol:float = 1e-15) -> np.ndarray : 

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

def norma(x:np.ndarray, p:int) -> float :
    
    x = np.array(x) 
    
    i:int = 0
    suma:float = 0

    while (i < len(x)) :
        
        # Caso especial -> Norma Infinito.
        if (p == "inf") :
            suma = max(suma, abs(x[i]))
            
        else: 
            suma += (abs(x[i]))**p
            
        i += 1
        
    if (p == "inf") : 
        return suma
    
    return np.float64(suma**(1/p)) 

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

# Ahora la función de este laboratorio.

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


# Van los tests.
 
# Matrices al azar
def genera_matriz_para_test(m,n=2,tam_nucleo=0):
    if tam_nucleo == 0:
        A = np.random.random((m,n))
    else:
        A = np.random.random((m,tam_nucleo))
        A = np.hstack([A,A])
    return(A)

def test_svd_reducida_mn(A,tol=1e-15):
    m,n = A.shape
    hU,hS,hV = svd_reducida(A,tol=tol)
    nU,nS,nVT = np.linalg.svd(A)
    r = len(hS)+1
    assert np.all(np.abs(np.abs(np.diag(hU.T @ nU))-1)<10**r*tol), 'Revisar calculo de hat U en ' + str((m,n))
    assert np.all(np.abs(np.abs(np.diag(nVT @ hV))-1)<10**r*tol), 'Revisar calculo de hat V en ' + str((m,n))
    assert len(hS) == len(nS[np.abs(nS)>tol]), 'Hay cantidades distintas de valores singulares en ' + str((m,n))
    assert np.all(np.abs(hS-nS[np.abs(nS)>tol])<10**r*tol), 'Hay diferencias en los valores singulares en ' + str((m,n))

for m in [2,5,10,20]:
    for n in [2,5,10,20]:
        for _ in range(10):
            A = genera_matriz_para_test(m,n)
            test_svd_reducida_mn(A)


# Matrices con nucleo

m = 12
for tam_nucleo in [2,4,6]:
    for _ in range(10):
        A = genera_matriz_para_test(m,tam_nucleo=tam_nucleo)
        test_svd_reducida_mn(A)

# Tamaños de las reducidas
A = np.random.random((8,6))
for k in [1,3,5]:
    hU,hS,hV = svd_reducida(A,k=k)
    assert hU.shape[0] == A.shape[0], 'Dimensiones de hU incorrectas (caso a)'
    assert hV.shape[0] == A.shape[1], 'Dimensiones de hV incorrectas(caso a)'
    assert hU.shape[1] <= k, 'Dimensiones de hU incorrectas (caso a)'
    assert hV.shape[1] <= k, 'Dimensiones de hV incorrectas(caso a)'
    assert len(hS) <= k, 'Tamaño de hS incorrecto'

print("Todo corrió y funcionó correctamente.")

# End.
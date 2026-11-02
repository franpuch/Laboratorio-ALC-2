# Grupo: Jr Army

# Funciones del Módulo ALC.

import numpy as np

# Funciones Auxiliares Propias -------------------------------------------------------------------------------------------------------------------- #

def esCuadrada(matriz:np.ndarray) -> bool : 
    if (matriz.shape[0] == 0) :
        return False
    return (matriz.shape[0] == matriz.shape[1])

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
            if (abs(matriz[i, j] - matriz[j, i]) > atol) :
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

def multMAt(A, B):
    es_vector = np.ndim(B) == 1

    if es_vector:
        B = np.reshape(B, (len(B), 1))

    filasA, columnasA = A.shape
    filasB, columnasB = B.shape

    if columnasA != filasB:
        return None

    res = np.zeros((filasA, columnasB))

    for i in range(filasA):
        for j in range(columnasB):
            res[i][j] = producto_interno(A[i,:], B[:,j])

    if es_vector:
        return res[:,0]

    return res

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

# Funciones Laboratorio 01 ------------------------------------------------------------------------------------------------------------------------ #

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

def matricesIguales(A, B) -> bool :
    
    if (len(A) != len(B)) :   # Miro que tengan igual número de filas.
        return False
    
    if (any(len(A[i]) != len(B[i]) for i in range(len(A)))) : # Miro que todas las filas sean de igual tamaño (y que tengan igual cantidad de columnas).
        return False

    for i in range(len(A)) :
        for j in range(len(A[i])) :
            if (error(A[i][j], B[i][j]) > 1e-12) :   # Uso la función 'error()' que implemente antes.
                return False
    return True

# Funciones Laboratorio 02 ------------------------------------------------------------------------------------------------------------------------ #

def rota(theta:float) -> np.ndarray :
    
    c:float = np.cos(theta)
    s:float = np.sin(theta) 
    
    res:np.ndarray = np.array([[c, -s], 
                               [s, c]])
    
    return res 

def escala(s) -> np.ndarray :
    
    s = np.asarray(s, dtype = float)
    
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

# Funciones Laboratorio 03 ------------------------------------------------------------------------------------------------------------------------ #

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

def normaliza(x:np.ndarray, p:int) -> list[list[float]] :

    res = []
    
    for elemento in x :
        # Calculo la norma una sola vez por vector.
        n = norma(elemento, p)
        
        # Si la norma es 0, es el vector nulo.
        if (n == 0.0) :  
            res.append(elemento)
            
        else: 
            res.append(elemento / n)
    
    return res

def normaMatMC(A:np.ndarray, q:int, p:int, Np:np.ndarray) -> tuple[float, np.ndarray] :
    
    columnas = A.shape[1]
    
    ## 2. Ajusto el muestreo al rango [-1, 1] para explorar todas las direcciones.
    x = np.random.rand(Np, columnas) * 2 - 1
    
    x = normaliza(x, p)
    
    res = -1.0
    vector_max = x[0] 
    
    for elemento in x :
        ## A 'calcularAx' le pido explícitamente 'vector_fila=True' para que entregue un 1D y sea más eficiente.
        ax = calcularAx(A, elemento, vector_fila=True)
        norma_actual = norma(ax, q) 
        
        if (res <= norma_actual) : 
            res = norma_actual
            vector_max = elemento
            
    return (res, vector_max)

def normaExacta(A:np.ndarray, p=[1, 'inf']) -> float | tuple[float,float] | None : 
    
    nro_filas, nro_columnas = np.shape(A) 
    
    if (p == 1) : 
        suma_max:float = 0 
        
        for j in range(0, nro_columnas) :
            suma_actual = 0
            
            for i in range(0, nro_filas) : 
                suma_actual += np.abs(A[i][j]) 
                
            if (suma_max < suma_actual) : 
                suma_max = suma_actual 
        
        return np.float64(suma_max) 
    
    elif (p == 'inf') : 
        suma_max:float = 0 
        
        for i in range(0, nro_filas) : 
            suma_actual = 0 
            
            for j in range(0, nro_columnas) : 
                suma_actual += np.abs(A[i][j]) 
                
            if (suma_max < suma_actual) : 
                suma_max = suma_actual 
        
        return np.float64(suma_max) 

    elif (p == [1, 'inf']) :

        # Primero calculamos la Norma 1.
        norma_1:float = 0 
                
        for j in range(0, nro_columnas) :
            suma_actual = 0
                    
            for i in range(0, nro_filas) : 
                suma_actual += np.abs(A[i][j]) 
                        
            if (norma_1 < suma_actual) : 
                norma_1 = suma_actual 

        # Ahora la Norma Infinito.
        norma_inf:float = 0 
                
        for i in range(0, nro_filas) : 
            suma_actual = 0 
                    
            for j in range(0, nro_columnas) : 
                suma_actual += np.abs(A[i][j]) 
                        
            if (norma_inf < suma_actual) : 
                norma_inf = suma_actual 

        return [norma_1, norma_inf]
    
    else : 
        return None 

def condMC(A:np.ndarray, p:int | str, Np:int=10000) -> float : 
    
    A_inversa:np.ndarray = np.linalg.inv(A) 
    
    norma_A:float = normaMatMC(A, p, p, Np)[0] 
    norma_A_inv:float = normaMatMC(A_inversa, p, p, Np)[0] 
    
    return np.float64(norma_A * norma_A_inv) 

def condExacta(A:np.ndarray, p:int | str) -> float | None : 
    
    A_inversa:np.ndarray = np.linalg.inv(A) 
    
    norma_A:float = normaExacta(A, p) 
    norma_A_inv:float = normaExacta(A_inversa, p) 
    
    if (norma_A == None) or (norma_A_inv == None) : 
        return None 
    
    return np.float64(norma_A * norma_A_inv)

# Funciones Laboratorio 04 ------------------------------------------------------------------------------------------------------------------------ #

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

# Funciones Laboratorio 05 ------------------------------------------------------------------------------------------------------------------------ #

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

# Funciones Laboratorio 06 ------------------------------------------------------------------------------------------------------------------------ #

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
    B = (B + traspuesta(B)) / 2
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

# Funciones Laboratorio 07 ------------------------------------------------------------------------------------------------------------------------ #

def transiciones_al_azar_continuas(n:int) -> np.ndarray : 
    
    res:np.ndarray = np.random.rand(n, n) 
    
    for columna in range(0, n) : 
            
        suma:float = np.sum(res[:, columna]) 
        
        # Normalizo "a la suma" las columnas.
        for fila in range(0, n) : 
            
            res[fila, columna] = res[fila, columna] / suma 
    
    return res 

def transiciones_al_azar_uniformes(n:int, thres:float) -> np.ndarray : 
    
    res:np.ndarray = np.random.rand(n, n) 
    
    for fila in range(0, n) : 
        
        for columna in range(0, n) : 
            
            if (res[fila, columna] <= thres) : 
                res[fila, columna] = 1 
                
            else : 
                res[fila, columna] = 1 
                
    # Ahora normalizo "a la suma" las columnas.
    for columna in range(0, n) : 
        
        suma:int = np.sum(res[:, columna]) 
        
        for fila in range(0, n) : 
            
            res[fila, columna] = res[fila, columna] / suma 
        
    return res 

def nucleo(A:np.ndarray, tol:float = 1e-15) -> np.ndarray : 
    
    A = multiplicar_matrices(A, traspuesta(A)) 
    n:int = A.shape[0]
    
    # 'vec' matriz de autovectores, 'val' matriz de autovalores. 
    vec, val = diagRH(A) 
    
    # Quiero saber cuántas veces 0 es autovector, y en qué columnas está.
    posiciones:list[int] = [] 
    for i in range(0, val.shape[0]) : 
        if (val[i, i] <= tol) : 
            pos:int = i   # Lo copio para evitar problemas con punteros al meter este valor en una lista. 
            posiciones.append(pos) 
    
    if (len(posiciones) == 0) : 
        return np.zeros((0,))
    
    else : 
        k = len(posiciones)
        res = np.zeros((n, k)) 
        index_aux:int = 0 
        
        for i in posiciones : 
            autovector:np.ndarray = np.array(vec[:, i]) 
            
            res[:,index_aux] = autovector 
            index_aux += 1 
        
        return res 

def crea_rala(listado:list[list[float]], m_filas:int, n_columnas:int, tol:float = 1e-15) -> list[dict[int, tuple[int, float]], tuple[int, int]] : 
    
    dict_res:dict[int, tuple[int, float]] = {} 
    
    # Caso Borde -> listado = [] (lista vacía) 
    if (len(listado) == 0) : 
        return [dict_res, (m_filas, n_columnas)] 
    
    # Ahora sí, vamos con la vaina.
    for index in range(0, len(listado[1])) : 
        
        valor:int = listado[2][index] 
        
        if (valor <= tol) : 
            continue   # Valores menores/iguales a la toleracia se descartan (según enunciado).
        
        fila:int = listado[0][index] 
        columna:int = listado[1][index] 
        
        dict_res[(fila, columna)] = valor 
    
    dim:tuple[int, int] = (m_filas, n_columnas) 
    
    return [dict_res, dim]

def multiplica_rala_vector(A:list[dict[tuple[int, int], float], tuple[int, int]], v:np.ndarray) -> np.ndarray : 
    
    matriz, (filas, columnas) = A
    res = np.zeros(filas, dtype = np.float64)
    
    # Recorro únicamente las posiciones no nulas.
    for (i, j), valor in matriz.items() :
        res[i] += valor * v[j]
    
    return res 

# Funciones Laboratorio 08 ------------------------------------------------------------------------------------------------------------------------ #

def svd_reducida(A, k="max", tol=1e-15):
    m, n = A.shape

    transpuesta_original = False

    if m < n:
        A = traspuesta(A)
        transpuesta_original = True

    m, n = A.shape

    if k == "max":
        k = n
    else:
        k = min(k, n)

    hatSig = np.zeros(k)
    hatV = np.zeros((n, k))
    hatU = np.zeros((m, k))

    AtA = multMAt(traspuesta(A), A)

    S, D = diagRH(AtA)

    j = 0

    while j < k and D[j][j] > tol:

        hatSig[j] = np.sqrt(D[j][j])

        hatV[:, j] = S[:, j]

        hatU[:, j] = multMAt(A, hatV[:, j]) / hatSig[j]

        j += 1

    hatSig = hatSig[:j]
    hatV = hatV[:, :j]
    hatU = hatU[:, :j]

    if transpuesta_original:
        return hatV, hatSig, hatU

    return hatU, hatSig, hatV

# Funciones de TP ###################################################################################################################################

# Ejercicio 2 ------------------------------------------------------------------------------------------------------------------------------------- #

def obtenerQ3d(imgPts, objPts):

    n = objPts.shape[0]   # Cantidad de imagenes

    Q = np.zeros((2 * n, 12))

    for i in range(n):

        U, V, W = objPts[i]   # Agarramos el punto.
        u, v = imgPts[i]      # Agarramos las coordenadas (u,v) del punto.

        P = np.array([U, V, W, 1])

        # Primera ecuación del punto i.
        Q[2 * i, 0:4] = P
        Q[2 * i, 8:12] = -u * P

        # Segunda ecuación del punto i
        Q[2 * i + 1, 4:8] = P
        Q[2 * i + 1, 8:12] = -v * P

        # Q es matriz de ceros, no hay que cambiar las columnas que quedan así.

    return Q

def resolverDLT3D(Q):

    U, S, V = svd_reducida(Q)

    m = V[:, -1]

    M = np.reshape(m, (3, 4))

    return M

# Funcion Auxiliar.
def corregirSignos(K, R, t):

    p = 1 if K[0,0] > 0 else -1
    q = 1 if K[1,1] > 0 else -1

    D = np.array([[p, 0, 0],
                  [0, q, 0],
                  [0, 0, 1]])

    K = multiplicar_matrices(K, D)
    R = multiplicar_matrices(D, R)
    t = calcularAx(D, t, vector_fila=True)

    # Calculamos el determinante de R
    detR = (R[0,0] * (R[1,1] * R[2,2] - R[1,2] * R[2,1])
          - R[0,1] * (R[1,0] * R[2,2] - R[1,2] * R[2,0])
          + R[0,2] * (R[1,0] * R[2,1] - R[1,1] * R[2,0]))

    if (detR < 0) :
        R = -R
        t = -t

    return K, R, t

def devolverMatricesCalibracion3d(M, alphas_iguales = False, skew_cero = False):

    # Separamos M en sus primeras 3 columnas y la última.
    A = M[:, 0:3]
    b = M[:, 3]

    # Matriz que invierte el orden de las coordenadas.
    J = np.array([[0, 0, 1],
                  [0, 1, 0],
                  [1, 0, 0]])

    # Calculamos QR de A^T * J
    B = multiplicar_matrices(traspuesta(A), J)
    Q, T = QR_con_HH(B)   # Usamos HouseHolder porque es mas estable.

    # Recuperamos K y R.
    K = multiplicar_matrices(J, multiplicar_matrices(traspuesta(T), J))

    R = multiplicar_matrices(J, traspuesta(Q))

    # Recuperamos t resolviendo Kt = b
    t = res_tri(K, b, inferior=False)

    # Normalizamos K para que K[2,2] = 1
    K = K / K[2, 2]

    if (alphas_iguales) :
      K, R, t = corregirSignos(K, R, t)

      f = (K[0,0] + K[1,1]) / 2   # Usamos punto medio para quedarnos con un valor.

      K[0,0] = f
      K[1,1] = f

    if (skew_cero) :
      K[0,1] = 0

    return R, t, K

# Ejercicio 3 ------------------------------------------------------------------------------------------------------------------------------------- #

def obtenerQ2d(imgPts, objPts):

    n = objPts.shape[0]

    Q = np.zeros((2*n, 9))

    for i in range(n):

        U, V = objPts[i]
        u, v = imgPts[i]

        P = np.array([U, V, 1])

        # Primera fila correspondiente al punto i
        Q[2*i, 0:3] = P
        Q[2*i, 6:9] = -u * P

        # Segunda fila correspondiente al punto i
        Q[2*i+1, 3:6] = P
        Q[2*i+1, 6:9] = -v * P

    return Q

# Función Auxiliar.
def resolverH4(Q):

    # Fijamos h9 = 1 y separamos las 8 incógnitas restantes.
    A = Q[:, :8]
    b = -Q[:, 8]

    # Factorizamos A mediante QR.
    Qort, R = QR_con_HH(A)

    # Calculamos el lado derecho del sistema triangular.
    y = calcularAx(traspuesta(Qort), b, vector_fila=True)

    # Resolvemos R*x = y por sustitución hacia atrás.
    x = res_tri(R, y, inferior=False)

    # Reconstruimos el vector h, agregando h9 = 1.
    h = np.zeros(9)
    h[:8] = x
    h[8] = 1

    # Convertimos el vector h en la homografía H.
    H = np.reshape(h, (3, 3))

    return H

# Función Auxiliar para la construcción de v_ij (vector para cálculo de A), según el paper citado.
def armarVij(ci, cj):

    res = np.array([
        ci[0] * cj[0],
        ci[0] * cj[1] + ci[1] * cj[0],
        ci[1] * cj[1],
        ci[0] * cj[2] + ci[2] * cj[0],
        ci[1] * cj[2] + ci[2] * cj[1],
        ci[2] * cj[2]
    ])

    return res

def armarA(Hlst):

    N = len(Hlst)
    A = np.zeros((2 * N, 6))

    for j in range(N):

        H = Hlst[j]

        # Primeras dos columnas de la homografía.
        c1 = H[:, 0]
        c2 = H[:, 1]

        # Calculamos los tres vectores necesarios.
        v12 = armarVij(c1, c2)
        v11 = armarVij(c1, c1)
        v22 = armarVij(c2, c2)

        # Las dos ecuaciones correspondientes a H_j
        A[2 * j, :] = v12
        A[2 * j + 1, :] = v11 - v22

    return A

def calcularMatrizK2D(A):

    C = multiplicar_matrices(traspuesta(A), A)

    V, D = diagRH(C)

    # Buscamos la posición del menor autovalor.
    indice_min = 0

    for i in range(1, 6):
        if D[i, i] < D[indice_min, indice_min]:
            indice_min = i

    # Autovector asociado al menor autovalor.
    b = V[:, indice_min]

    # Reconstruimos la matriz simétrica B.
    B = np.array([
        [b[0], b[1], b[3]],
        [b[1], b[2], b[4]],
        [b[3], b[4], b[5]]
    ])

    # B se determina a menos de escala y signo.
    # Elegimos el signo que la hace definida positiva.
    L = calculaCholesky(B)

    if (L is None) :
        L = calculaCholesky(-B)

    if (L is None) :   # Si B ni -B es definida positiva, no podemos aplicar Cholesky.
        return None

    # Como B = L L^T, recuperamos K = (L^T)^-1.
    K = inversa(traspuesta(L))

    # Normalizamos para que K[2,2] = 1.
    K = K / K[2, 2]

    return K

def obtenerPose2D(Hj, K):

    # Separamos las tres columnas de la homografía.
    c1 = Hj[:, 0]
    c2 = Hj[:, 1]
    c3 = Hj[:, 2]

    # Resolvemos K * x = ci.
    # Como K es triangular superior, usamos res_tri.
    a1 = res_tri(K, c1, inferior=False)
    a2 = res_tri(K, c2, inferior=False)
    a3 = res_tri(K, c3, inferior=False)

    # Calculamos el factor de escala.
    lam = 1 / norma(a1, 2)

    # Recuperamos las dos primeras columnas de R.
    r1 = lam * a1
    r2 = lam * a2

    # GS entre r1 y r2 para ortogonalidad.
    r2 = r2 - producto_interno(r1, r2) * r1
    r2 = r2 / norma(r2, 2)

    # Calculamos r3 mediante el producto vectorial.
    r3 = np.array([
        r1[1] * r2[2] - r1[2] * r2[1],
        r1[2] * r2[0] - r1[0] * r2[2],
        r1[0] * r2[1] - r1[1] * r2[0]
    ])

    # Construimos R por columnas.
    R = np.zeros((3, 3))
    R[:, 0] = r1
    R[:, 1] = r2
    R[:, 2] = r3

    # Recuperamos la traslación.
    t = lam * a3

    return R, t

# Ejercicio 6 ------------------------------------------------------------------------------------------------------------------------------------- #

def calculoLambdaMu(I0, J0):

    d = producto_interno(J0, J0) - producto_interno(I0, I0)
    p = -producto_interno(I0, J0)

    if (p != 0) :

        lam = np.sqrt((d + np.sqrt(d**2 + 4*(p**2))) / 2)
        mu = p / lam

    else:

        if (d > 0) :
            lam = np.sqrt(d)
            mu = 0

        elif (d < 0) :
            lam = 0
            mu = np.sqrt(-d)

        else:
            lam = 0
            mu = 0

    return (lam, mu)

def calcularSalidaPOSIT(Is, Js, fx, m0, M0):

    Z0 = fx / norma(Is, 2)

    # Calculamos las primeras dos filas de R.
    i = Is / norma(Is, 2)
    j = Js / norma(Js, 2)

    # Tercer fila con producto vectorial.
    k = np.array([
        i[1] * j[2] - i[2] * j[1],
        i[2] * j[0] - i[0] * j[2],
        i[0] * j[1] - i[1] * j[0]
    ])

    # Construimos la matriz de rotación R.
    R = np.array([i, j, k])

    # Coordenadas de M0 en el sistema de la cámara.
    T0 = np.array([
        Z0 * m0[0] / fx,
        Z0 * m0[1] / fx,
        Z0
    ])

    # Despejamos 't' de la ecuación (15) tomando M0.
    t = T0 - calcularAx(R, M0, vector_fila=True)

    return i, j, k, Z0, R, t

def evaluarPose(R, t, obj_vecs, img_vecs, K):

    validas = 0
    suma_errores = 0

    for i in range(len(obj_vecs)):

        # Pasamos el punto al sistema de la cámara.
        P = calcularAx(R, obj_vecs[i], vector_fila=True) + t

        # Verificamos que esté delante de la cámara.
        if (P[2] > 0) :

            # Con K obtenemos las coordenadas en la foto medida en pixeles.
            proy = calcularAx(K, P, vector_fila=True)

            # Como siempre, dividimos por la 3er coordenada.
            u_hat = proy[0] / proy[2]
            v_hat = proy[1] / proy[2]

            # Error cuadrático respecto del punto original.
            error = (u_hat - img_vecs[i, 0])**2 + (v_hat - img_vecs[i, 1])**2

            suma_errores += error
            validas += 1

    # Si ningún punto es válido, no podemos calcular el RMS.
    if (validas == 0) :
        return 0, "No hay puntos validos."

    RMS = np.sqrt(suma_errores / validas)

    return validas, RMS

def pseudoInversa(A):

    U_hat, S_hat, V_hat = svd_reducida(A)

    k = len(S_hat)

    # Construimos la matriz diagonal con los inversos de los valores singulares no nulos.
    S_inv = np.zeros((k, k))

    for i in range(k):
        S_inv[i, i] = 1 / S_hat[i]

    # A+ = V * S_inv * U.T
    pseudo = multiplicar_matrices(
        multiplicar_matrices(V_hat, S_inv),
        traspuesta(U_hat)
    )

    return pseudo 

def aplicarPOSIT(obj_vecs, img_vecs, K, MaxIt, atol):

    n_puntos = len(obj_vecs)

    fx = K[0, 0]
    fy = K[1, 1]
    cu = K[0, 2]
    cv = K[1, 2]

    # 1. Centramos las coordenadas de la imagen.
    img_centradas = np.zeros((n_puntos, 2))

    for i in range(n_puntos):
        img_centradas[i, 0] = img_vecs[i, 0] - cu
        img_centradas[i, 1] = (img_vecs[i, 1] - cv) * fx / fy

    M0 = obj_vecs[0]
    m0 = img_centradas[0]

    # 2. Construimos A con los vectores Mi - M0.
    A = np.zeros((n_puntos - 1, 3))

    for i in range(1, n_puntos):
        A[i - 1, :] = obj_vecs[i] - M0

    # Calculamos la pseudoinversa (no cambia con las iteraciones).
    A_pseudo = pseudoInversa(A)

    # 3. Calculamos el vector normal al plano de la H.
    # Los primeros dos vectores de A no son paralelos.
    a1 = A[0]
    a2 = A[1]

    n = np.array([
        a1[1] * a2[2] - a1[2] * a2[1],
        a1[2] * a2[0] - a1[0] * a2[2],
        a1[0] * a2[1] - a1[1] * a2[0]
    ])

    n = n / norma(n, 2)

    # Inicializamos los factores de perspectiva.
    eps = np.zeros(n_puntos - 1)

    rms_anterior = float("inf")
    mejor_rms = float("inf")
    mejor_R = None
    mejor_t = None

    # 5. Iteraciones de POSIT.
    for iteracion in range(MaxIt):

        # Construimos los vectores b(u) y b(v).
        bu = np.zeros(n_puntos - 1)
        bv = np.zeros(n_puntos - 1)

        for i in range(n_puntos - 1):
          bu[i] = img_centradas[i + 1, 0] * (1 + eps[i]) - m0[0]   # En 'img_centradas' ignoramos la 1er fila que corresponde al m0
          bv[i] = img_centradas[i + 1, 1] * (1 + eps[i]) - m0[1]
        
        # Resolvemos los sistemas mediante la pseudoinversa.
        I0 = calcularAx(A_pseudo, bu, vector_fila=True)
        J0 = calcularAx(A_pseudo, bv, vector_fila=True)

        # Con la función del 6.b calculamos lambda y mu.
        lam, mu = calculoLambdaMu(I0, J0)

        rms_iteracion = float("inf")
        R_iteracion = None
        t_iteracion = None
        k_iteracion = None
        Z0_iteracion = None

        # Construimos las dos soluciones posibles.
        I1 = I0 + lam * n
        J1 = J0 + mu * n

        I2 = I0 - lam * n
        J2 = J0 - mu * n

        # Con la función del 6.c recuperamos las dos poses.
        i1, j1, k1, Z01, R1, t1 = calcularSalidaPOSIT(I1, J1, fx, m0, M0)

        i2, j2, k2, Z02, R2, t2 = calcularSalidaPOSIT(I2, J2, fx, m0, M0)

        # Con la función del 6.d evaluamos ambas poses.
        validas1, rms1 = evaluarPose(R1, t1, obj_vecs, img_vecs, K)
        validas2, rms2 = evaluarPose(R2, t2, obj_vecs, img_vecs, K)

        # Elegimos la mejor pose válida.
        if (validas1 == n_puntos) :

          # La primera pose es válida.
          if (validas2 != n_puntos) :

            # La segunda no es válida. Elegimos la primera.
            R_iteracion = R1
            t_iteracion = t1
            k_iteracion = k1
            Z0_iteracion = Z01
            rms_iteracion = rms1

          else :

            # Ambas son válidas. Comparamos los RMS.
            if (rms1 <= rms2):

              R_iteracion = R1
              t_iteracion = t1
              k_iteracion = k1
              Z0_iteracion = Z01
              rms_iteracion = rms1

            else :

              R_iteracion = R2
              t_iteracion = t2
              k_iteracion = k2
              Z0_iteracion = Z02
              rms_iteracion = rms2

        else:

          # La primera pose no es válida.
          if (validas2 == n_puntos) :

            # La segunda es válida. La elegimos.
            R_iteracion = R2
            t_iteracion = t2
            k_iteracion = k2
            Z0_iteracion = Z02
            rms_iteracion = rms2

          else:

            # Ninguna de las dos poses es válida.
            break

        # Guardamos la mejor pose encontrada.
        if (rms_iteracion < mejor_rms) :
            mejor_rms = rms_iteracion
            mejor_R = R_iteracion
            mejor_t = t_iteracion

        # Convergencia.
        if (abs(rms_anterior - rms_iteracion) < atol) :
            break

        # Actualizamos los factores de perspectiva.
        for i in range(0, n_puntos - 1) :

            eps[i] = producto_interno(A[i], k_iteracion) / Z0_iteracion

        rms_anterior = rms_iteracion

    return mejor_R, mejor_t


# End.
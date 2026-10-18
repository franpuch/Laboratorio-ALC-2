# Funciones del Laboratorio 07 - Funciones del Modulo ALC.
# Para esta entrega, (... aún no confirmaron las funciones a entregar ...)

import numpy as np
from funciones_generales import multiplicar_matrices, traspuesta
from modulo_alc_06 import diagRH


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


# End.
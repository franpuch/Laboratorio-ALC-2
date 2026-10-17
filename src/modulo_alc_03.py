# Funciones del Laboratorio 03 que se piden en la entrega.
# Para esta entrega, sólo las funciones del Módulo ALC.

import numpy as np
from funciones_generales import multiplicar_matrices, calcularAx

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

# End.
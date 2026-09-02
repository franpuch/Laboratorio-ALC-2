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
    
    res:list[list[float]] = []
    
    for elemento in x :
        
        if not(np.any(elemento)) :   # Es el vector nulo. 
            res.append(elemento)
        
        else: 
            res.append(elemento/norma(elemento, p))
    
    return res 

#! No pasa el Test del Servidor de Entregas: La ejecución demoró más de lo permitido.
def normaMatMC(A:np.ndarray, q:int, p:int, Np:np.ndarray) -> tuple[float, np.ndarray] :
    
    columnas:int = A.shape[1]
    
    x = np.random.rand(Np, columnas)
    
    x = normaliza(x, p)
    
    res:float = 0
    vector_max = x[0] 
    
    for elemento in x :
        
        norma_actual = norma(calcularAx(A, elemento), q) 
        
        if (res <= norma_actual) : 
            res = norma_actual
            vector_max = elemento
            
    return (np.float64(res), vector_max) 

#! No pasa el Test del Servidor de Entregas: No se pueden correr los test.
def normaExacta(A:np.ndarray, p:int) -> float | None : 
    
    A = np.array(A) 
    
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
    
    else : 
        return None 

def condMC(A:np.ndarray, p:int | str) -> float : 
    
    A_inversa:np.ndarray = np.linalg.inv(A) 

    Np:int = 10000 
    
    norma_A:float = normaMatMC(A, p, p, Np)[0] 
    norma_A_inv:float = normaMatMC(A_inversa, p, p, Np)[0] 
    
    return np.float64(norma_A * norma_A_inv) 

def condExacto(A:np.ndarray, p:int | str) -> float | None : 
    
    A_inversa:np.ndarray = np.linalg.inv(A) 
    
    norma_A:float = normaExacta(A, p) 
    norma_A_inv:float = normaExacta(A_inversa, p) 
    
    if (norma_A == None) or (norma_A_inv == None) : 
        return None 
    
    return np.float64(norma_A * norma_A_inv)

# End.
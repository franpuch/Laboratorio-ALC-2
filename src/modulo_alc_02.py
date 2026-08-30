# Funciones del Laboratorio 02 que se piden en la entrega.
# Para esta entrega, sólo las funciones del Módulo ALC.

import numpy as np
from funciones_generales import multiplicar_matrices, calcularAx

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

    '''
    Lo que busco es una composición de transformaciones lineales: f(g(v)) donde 'g()' es la 'TL rotación' y 'f()' es la 
    'TL escalado'. Como vengo trabajando con matrices (de las TLs), la composición es la multiplicacion de las 
    respectivas matrices. 
    '''
    
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
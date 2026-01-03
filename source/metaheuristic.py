from pyMetaheuristic.algorithm import bat_algorithm
import numpy as np
from geo_module import *
from solver import *

def objective_function(vetor_parametros):

    penalidadeTC=0
    penalidadeFb=0
    
    data_path='data/'
    geo_generator(filename='teste_heuristica', num_segmentos = 8, l_base = 5, h_base = 5, ponto_final = (-30.0, 0.0, 15.0), ID = 1)
    nodes,elements = geo_reader(data_path+'teste_heuristica1.geo')
    deslocamentos, tensoes, elementos_falha, lambdas = fem_solver_3d(nodes, elements-1, ID=1)
    #print(f'Deslocamentos: {deslocamentos}')
    #print(f'Tensões: {tensoes}')
    #print(f'Elementos falha: {elementos_falha}')
    #print(f'lambdas: {lambdas}')
    #print(f'Nós: {nodes}')
    #print(f'Elementos: {elements}')


    # Massa da estrutura
    densidade_aco = 7330.0; area_secao = 0.005; g = 9.81
    volume_total = 0.0
    for elem in (elements-1):
        ponto1 = nodes[elem[0]]
        ponto2 = nodes[elem[1]]
        comprimento_barra = np.linalg.norm(ponto2 - ponto1)
        volume_total += area_secao*comprimento_barra
    massa_total_kg = volume_total*densidade_aco

    if len(elementos_falha)!=0:
        penalidadeTC+=1000000000
    for flambagem in lambdas:
        if flambagem >=1:
            penalidadeFb+=1000000000

    valor_objetivo=massa_total_kg+penalidadeTC+penalidadeFb
    print(valor_objetivo)
    return valor_objetivo

objective_function([])
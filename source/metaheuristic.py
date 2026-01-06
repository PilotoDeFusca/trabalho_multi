from pyMetaheuristic.algorithm import bat_algorithm
from pyMetaheuristic.algorithm import genetic_algorithm
import numpy as np
from geo_module import *
from solver import *

def objective_function(vetor_variaveis=[8,5,5,-30,0,15]):

    num_segmentos,l_base, h_base, ponto_final_x, ponto_final_y, ponto_final_z= vetor_variaveis

    penalidadeTC=0
    penalidadeFb=0
    
    data_path='data/'
    filename='teste_heuristica'

    ponto_final=(ponto_final_x,ponto_final_y,ponto_final_z)

    #geo_generator(filename='teste_heuristica', num_segmentos = 8, l_base = 5, h_base = 5, ponto_final = (-30.0, 0.0, 15.0), ID = 1)
    geo_generator(filename, num_segmentos, l_base, h_base, ponto_final, ID=1)
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
        if flambagem <= 1:
            penalidadeFb+=1000000000

    valor_objetivo=massa_total_kg+penalidadeTC+penalidadeFb
    print(valor_objetivo)
    return valor_objetivo


parameters_bat = {
    'swarm_size': 10,
    'min_values': (8,3,3,-30.0,0.0,15.0),
    'max_values': (15.9,10,10,-30.0,0.0,15.0),
    'iterations': 10,
    'alpha': 0.8,
    'gama': 0.8,
    'fmin': 0,
    'fmax': 900000000000,
	  'verbose': True,
	  'start_init': None,
	  'target_value': None
}

# GA - Parameters
parameters_ga = {
    'population_size': 10,
    'min_values': (8,3,3,-30.0,0.0,15.0),
    'max_values': (15.9,10,10,-30.0,0.0,15.0),
    'generations': 10,
    'mutation_rate': 0.3,
    'elite': 1,
    'eta': 1,
    'mu': 1,
	  'verbose': True,
	  'start_init': None,
	  'target_value': None
}

objective_function([8,5,5,-30,0,15])

#bat = bat_algorithm(target_function = objective_function, **parameters_bat)

#variables = bat[:-1]
#minimum   = bat[ -1]
#print('Variables: ', np.around(variables, 4) , ' Minimum Value Found: ', round(minimum, 4) )

ga = genetic_algorithm(target_function = objective_function, **parameters_ga)

variables = ga[:-1]
minimum   = ga[ -1]
print('Variables: ', np.around(variables, 4) , ' Minimum Value Found: ', round(minimum, 4) )

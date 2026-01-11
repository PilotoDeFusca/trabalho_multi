from pyMetaheuristic.algorithm import bat_algorithm
from pyMetaheuristic.algorithm import genetic_algorithm
import numpy as np
from geo_module import *
from solver import *
from post import writer
from data_clean import *

data_path='data/'
filename='trelica'
objective_path=data_path+'/objetive_funcion_values.txt'
clean_path=data_path+'*'

data_clean(clean_path)


def objective_function(vetor_variaveis=[8,5,5,-30,0,15,2]):

    num_segmentos,l_base, h_base, ponto_final_x, ponto_final_y, ponto_final_z,id_trelica= vetor_variaveis

    penalidadeTC=0
    penalidadeFb=0
    
    data_path='data/'
    filename='teste_heuristica'

    ponto_final=(ponto_final_x,ponto_final_y,ponto_final_z)

    geo_path=geo_generator(filename, data_path, num_segmentos, l_base, h_base, ponto_final, id_trelica)
    nodes,elements = geo_reader(geo_path)
    deslocamentos, tensoes, elementos_falha, lambdas = fem_solver_3d(nodes, elements-1, ID=id_trelica)
    print(f'Flambagem: {lambdas}') 

    # Massa da estrutura
    densidade_aco = 7330.0; area_secao = 0.005; g = 9.81
    volume_total = 0.0
    for elem in (elements-1):
        ponto1 = nodes[elem[0]]
        ponto2 = nodes[elem[1]]
        comprimento_barra = np.linalg.norm(ponto2 - ponto1)
        volume_total += area_secao*comprimento_barra
    massa_total_kg = volume_total*densidade_aco

    # Penalidade por falha em tração/compressão - penaliza cada elemento que falha
    penalidadeTC = 1000000000 * len(elementos_falha) if len(elementos_falha) > 0 else 0
    
    # Penalidade por flambagem - penaliza proporcionalmente a quanto está abaixo de 1
    for flambagem in lambdas:
        if flambagem <= 1:
            penalidadeFb+=1000000000

    valor_objetivo=massa_total_kg+penalidadeTC+penalidadeFb
    print(valor_objetivo)
    writer(nodes, elements-1, elementos_falha, deslocamentos, tensoes, lambdas, valor_objetivo, objective_path)
        
    return valor_objetivo

def run_bat():

    #Bat - Parameters
    parameters_bat = {
        'swarm_size': 10,
        'min_values': (8,3,3,-30.0,0.0,15.0,1),
        'max_values': (20.9,15,15,-30.0,0.0,15.0,2.9),
        'iterations': 10,
        'alpha': 0.8,
        'gama': 0.8,
        'fmin': 0,
        'fmax': 900000000000,
        'verbose': True,
        'start_init': None,
        'target_value': None
    }
    
    bat = bat_algorithm(target_function = objective_function, **parameters_bat)

    variables = bat[:-1]
    minimum   = bat[ -1]
    print('Variables: ', np.around(variables, 4) , ' Minimum Value Found: ', round(minimum, 4) )

def run_ga():
# GA - Parameters
    parameters_ga = {
        'population_size': 10,
        'min_values': (8,3,3,-30.0,0.0,15.0,3),
        'max_values': (15.9,10,10,-30.0,0.0,15.0,3),
        'generations': 10,
        'mutation_rate': 0.3,
        'elite': 1,
        'eta': 1,
        'mu': 1,
        'verbose': True,
        'start_init': None,
        'target_value': None
    }

    ga = genetic_algorithm(target_function = objective_function, **parameters_ga)

    variables = ga[:-1]
    minimum   = ga[ -1]
    print('Variables: ', np.around(variables, 4) , ' Minimum Value Found: ', round(minimum, 4) )
    
run_bat()

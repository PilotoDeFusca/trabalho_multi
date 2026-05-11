from pyMetaheuristic.algorithm import bat_algorithm
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import sys

from geo_module import *
from plot import *
from props import *
from post import *
from solver import *
from otimizador_LP import *

data_path='data/'
filename='boom'
objective_path=data_path+'/objetive_funcion_values.txt'
clean_path=data_path+'*'
data_clean(clean_path)

# --------------- Configuração global do plot ---------------

plt.ion()
fig = plt.figure(figsize=(16, 6))
ax_mass = fig.add_subplot(1, 2, 1)
ax_struct = fig.add_subplot(1, 2, 2, projection='3d')

line, = ax_mass.plot([],[], 'b-o', label='Best (g)')
ax_mass.set_xlabel('Geração')
ax_mass.set_ylabel('Massa (g)')
ax_mass.grid(True)
ax_mass.legend()

ax_struct.set_title('Best (estrutura)')

# -----------------------------------------------------------

# --------------- Variáveis globais de controle -------------

global_best_val = float('inf'); best_mass_so_far = 0; eval_counter = 0 
x_data =[]; y_data =[]

best_nodes_so_far = None; best_elements_so_far = None; best_IDs_so_far = None

# -----------------------------------------------------------

def objective_function(vetor_variaveis):
    
    # Variáveis do plot
    global global_best_val, eval_counter, best_mass_so_far, x_data, y_data
    global best_nodes_so_far, best_elements_so_far, best_IDs_so_far

    data_path='data/'; filename='boom'

    num_segmentos, l_base, h_base, ponto_final_x, ponto_final_y, ponto_final_z,id_trelica, alpha = vetor_variaveis
    id_trelica = int(np.rint(id_trelica)); num_segmentos = int(np.rint(num_segmentos)); alpha = float(alpha)
    ponto_final=(ponto_final_x,ponto_final_y,ponto_final_z)

    geo_path = geo_generator(filename, data_path, num_segmentos, l_base, h_base, ponto_final, id_trelica, alpha)
    nodes,elements = geo_reader(geo_path)
    elements0 = elements - 1

    # Cada geometria é iniciada com todos os perfis mais resistentes, estes serão otimizados no bloco de LP
    biblioteca_perfis = [props_c1, props_c2]
    IDs_elementos = np.zeros(len(elements)); IDs_elementos.fill(props_c1['ID'])

    deslocamentos, tensoes, elementos_falha, lambdas = fem_solver_3d(nodes, elements0, ID=id_trelica, props_IDs=IDs_elementos, biblioteca_perfis=biblioteca_perfis, forces=forces1)  

    # Cria um dicionário para busca rápida das propriedades pelo ID do material
    prop_map = {perfil['ID']: perfil for perfil in biblioteca_perfis}

    volume_total = 0.0; massa_total_kg = 0.0; g = 9.81

    for i, elem in enumerate(elements0):
    
        id_elemento = IDs_elementos[i]; p = prop_map[id_elemento]; area_secao = p['A']; densidade = p['densidade']

        ponto1 = nodes[elem[0]]; ponto2 = nodes[elem[1]]
    
        # Cálculo individual
        comprimento_barra = np.linalg.norm(ponto2 - ponto1)
        volume_elemento = area_secao*comprimento_barra
    
        # Incremento nos totais
        volume_total += volume_elemento
        massa_total_kg += volume_elemento*densidade

    # Penalidades
    if(np.max(np.abs(deslocamentos.reshape(-1, 6)[:, 2])) > 0.03):
        penalidade_disp = 1000000000
    else:
        penalidade_disp = 0
        
    penalidadeTC = 1000000000*len(elementos_falha) if len(elementos_falha) > 0 else 0
    
    penalidadeFb = 0
    for flambagem in lambdas:
        if flambagem <= 1.2:
            penalidadeFb += 1000000000

    valor_objetivo = massa_total_kg + penalidadeTC + penalidadeFb + penalidade_disp

    # Bloco da LP
    ##########################################################################################

    A_atual, IDs_elementos_v2, historico_volume, L = otimizador_LP(nodes, elements, biblioteca_props=biblioteca_perfis, forces=forces1, max_iter=20, move_limit=0.06, tolerancia=1e-6, ID=id_trelica)

    massa_LP = 0.0
    for i in range(len(elements0)): massa_LP += A_atual[i]*L[i]*prop_map[IDs_elementos_v2[i]]['densidade']
    valor_objetivo = massa_LP + penalidadeTC + penalidadeFb + penalidade_disp

    ##########################################################################################

    # ----- ATUALIZAÇÃO DO MELHOR GLOBAL -----

    if valor_objetivo < global_best_val:

        global_best_val = valor_objetivo
        best_mass_so_far = massa_LP 
        best_nodes_so_far = np.array(nodes, copy=True) # Salva a melhor estrutura até agora
        best_elements_so_far = np.array(elements0, copy=True)
        best_IDs_so_far = np.array(IDs_elementos_v2, copy=True)
    
    eval_counter += 1

    # ---------------------------------------

    # --------- PLOT EM TEMPO REAL -----------

    if eval_counter % swarm_size_global == 0:

        generation = eval_counter//swarm_size_global
        
        x_data.append(generation); y_data.append(best_mass_so_far*1000)
        
        line.set_xdata(x_data); line.set_ydata(y_data)

        ax_mass.relim(); ax_mass.autoscale_view()

        if best_nodes_so_far is not None and best_elements_so_far is not None and best_IDs_so_far is not None:
            plotar_estrutura(best_nodes_so_far, best_elements_so_far, props_IDs=best_IDs_so_far, biblioteca_perfis=biblioteca_perfis, ax=ax_struct, clear=True, title=f'Best estrutura (gen {generation})', savepath=None, show=False,)

        plt.draw(); plt.pause(0.01) 

    # --------------------------------------

    return valor_objetivo

swarm_size_global = 10 # Tamanho do enxame (número de soluções por geração) 

def run_bat():
    
    global swarm_size_global
    
    parameters_bat = { # Definição dos upper e lower bounds e parâmetros do algoritmo
        'swarm_size': swarm_size_global, 
        'min_values': (2 ,0.03, 0.03, -0.66, 0.0, 0.344, 1, 0.2),
        'max_values': (15.9 , 0.12, 0.12, -0.66, 0.0, 0.344, 3, 3.0),
        'iterations': 10,
        'alpha': 0.8, 'gama': 0.8,
        'fmin': 0, 'fmax': 900000000000,
        'verbose': True, 'start_init': None, 'target_value': None}
    
    bat = bat_algorithm(target_function = objective_function, **parameters_bat)

    variables = bat[:-1]; minimum   = bat[ -1]
    print('Variables: ', np.around(variables, 4) , ' Minimum Value Found: ', round(minimum, 4) )

    plt.ioff(); plt.show()

run_bat()
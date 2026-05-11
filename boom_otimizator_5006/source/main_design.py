from plot import *
from solver import *
from geo_module import *
from post import *
from props import *

data_path='data/'; filename='boom'; clean_path=data_path+'*'; data_clean(clean_path)

# gerador de geometria parametrizado
# ID 1 -> TRIÂNGULO EM PÉ
# ID 2 -> TRIÂNGULO DE CABEÇA PRA BAIXO
# ID 3 -> SEÇÃO RETANGULAR

#                                      n_segmentos, l_base, h_base,    Coordenadas finais,   ID_trelica
geo_path = geo_generator(filename, data_path, 6,   0.12,   0.12, (-0.66,       0.0,      0.344), ID=1, alpha=0.5)                   
nodes, elements = geo_reader(geo_path)                                                                                   

biblioteca_perfis = [props_c1, props_c2]
IDs_elementos = np.zeros(len(elements)); IDs_elementos.fill(props_c1['ID'])

deslocamentos, tensoes, elementos_falha, lambdas = fem_solver_3d(nodes, elements-1, ID=1, props_IDs=IDs_elementos, biblioteca_perfis=biblioteca_perfis, forces=forces1)  
                                                                                
plotar_estrutura_deformada(nodes, elements-1, deslocamentos, escala_deformacao=1.0, props_IDs=IDs_elementos, biblioteca_perfis=biblioteca_perfis)                                    
plotar_tensoes(nodes, elements-1, tensoes, deslocamentos, escala_deformacao=1.0)                                         

printer(nodes, elements-1, elementos_falha, deslocamentos, tensoes, lambdas, props_IDs=IDs_elementos, biblioteca_perfis=biblioteca_perfis)  

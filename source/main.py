from plot import *
from solver import *
from geo_module import *
from post import *
from data_clean import *

data_path='data/'
filename='trelica'
clean_path=data_path+'*'

# gerador de geometria parametrizado
# ID 1 -> TRIÂNGULO EM PÉ
# ID 2 -> TRIÂNGULO DE CABEÇA PRA BAIXO
# ID 3 -> SEÇÃO RETANGULAR

data_clean(clean_path)
#geo_path=geo_generator( filename, data_path, 8.5771,   3.127,   5.0952,(-30.,       0.,      15.),       3)
#geo_path=geo_generator( filename, data_path, 8,   4.1598,   9.6704,(-30.,       0.,      15.),       2)
geo_path=geo_generator( filename, data_path, 8.917,   3,   6.6831,(-30.,       0.,      15.),       2)
#geo_generator(filename='trelica', num_segmentos = 8, l_base = 3.0909, h_base = 5.6557, ponto_final = (-30, 0, 15), ID = 1)

#geo_generator(filename='trelica', num_segmentos = 8, l_base = 4, h_base = 6, ponto_final = (-30.0, 0.0, 15.0), ID = 1)    # Benchmark 01
#geo_generator(filename='trelica', num_segmentos = 1, l_base = 3.0, h_base = 3.0, ponto_final = (-10.0, 0.0, 5.0), ID = 1)
#geo_generator(filename='trelica', num_segmentos = 10, l_base = 4.0, h_base = 5.0, ponto_final = (-15.0, 0.0, 10.0), ID = 3)  # geo_module
nodes,elements = geo_reader(geo_path)                                                                      # geo_module

deslocamentos, tensoes, elementos_falha, lambdas = fem_solver_3d(nodes, elements-1, ID=2)                                  # solver

plotar_estrutura(nodes, elements-1)                                                                                        # plot
plotar_estrutura_deformada(nodes, elements-1, deslocamentos, escala_deformacao=4.0)                                        # plot
plotar_tensoes(nodes, elements-1, tensoes, deslocamentos, escala_deformacao=4.0)                                           # plot

printer(nodes, elements-1, elementos_falha ,deslocamentos, tensoes, lambdas)                                               # post

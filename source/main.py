from plot import *
from solver import *
from geo_module import *
from post import *

data_path='data/'

# gerador de geometria parametrizado
# ID 1 -> TRIÂNGULO EM PÉ
# ID 2 -> TRIÂNGULO DE CABEÇA PRA BAIXO
# ID 3 -> SEÇÃO RETANGULAR

#eo_generator(filename='trelica', num_segmentos = 4, l_base = 5, h_base = 5, ponto_final = (-30.0, 0.0, 15.0), ID = 1)    # Benchmark 01
geo_generator(filename='trelica', num_segmentos = 1, l_base = 3.0, h_base = 3.0, ponto_final = (-10.0, 0.0, 5.0), ID = 1)  # geo_module
nodes,elements = geo_reader(data_path+'trelica1.geo')                                                                      # geo_module

deslocamentos, tensoes, elementos_falha, lambdas = fem_solver_3d(nodes, elements-1, ID=1)                                  # solver

plotar_estrutura(nodes, elements-1)                                                                                        # plot
plotar_estrutura_deformada(nodes, elements-1, deslocamentos, escala_deformacao=4.0)                                        # plot
plotar_tensoes(nodes, elements-1, tensoes, deslocamentos, escala_deformacao=4.0)                                           # plot

printer(nodes, elements-1, elementos_falha ,deslocamentos, tensoes, lambdas)                                               # post

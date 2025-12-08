from plot import *
from solver import *
from geo_generator_teste import *
from geo_reader_teste import *

data_path='data/'

# gerador de geometria parametrizado
# ID 1 -> TRIÂNGULO EM PÉ
# ID 2 -> TRIÂNGULO DE CABEÇA PRA BAIXO
# ID 3 -> SEÇÃO RETANGULAR

# geo_generator(
#         filename='trelica',
#         num_segmentos=4,
#         l_base=3.5,
#         h_base=5.5,
#         ponto_final=(-30.0, 0.0, 15.0),
#         ID = 1
#     )

nodes,elements=geo_reader(data_path+'trelica1.geo') 

# Plot sem deformação
plotar_estrutura(nodes, elements-1)

# Solver FEM3d
deslocamentos, tensoes = fem_solver_3d(nodes, elements-1, ID=1)

# Pós-processamento

# Massa da estrutura
densidade_aco = 7330.0; area_secao = 0.005; g = 9.81
volume_total = 0.0
for elem in (elements - 1):
  ponto1 = nodes[elem[0]]
  ponto2 = nodes[elem[1]]
  comprimento_barra = np.linalg.norm(ponto2 - ponto1)
  volume_total += area_secao*comprimento_barra
massa_total_kg = volume_total*densidade_aco

max_disp = np.max(np.abs(deslocamentos.reshape(-1, 6)[:, :3]))
max_stress_tra = np.max(tensoes) if tensoes.size > 0 else 0
min_stress_com = np.min(tensoes) if tensoes.size > 0 else 0

print(volume_total)
print(f"Deslocamento translacional máximo: {max_disp*1000:.4f} mm")
print(f"Tensão máxima (Tração): {max_stress_tra/1e6:.2f} MPa")
print(f"Tensão mínima (Compressão): {min_stress_com/1e6:.2f} MPa")
print(f"Massa total da estrutura: {massa_total_kg:.2f} kg")

plotar_estrutura_deformada(nodes, elements-1, deslocamentos, escala_deformacao=1.0)
plotar_tensoes(nodes, elements-1, tensoes, deslocamentos, escala_deformacao=1.0)
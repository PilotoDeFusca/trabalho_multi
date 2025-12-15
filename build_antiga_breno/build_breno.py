# O CÓDIGO FUNCIONA TODO NO SI

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math

# Gerador de geometria
################################################################################

def geo_generator(filename, num_segmentos, l_base, h_base,  ponto_final, ID):

    x0, y0, z0 = 0.0, 0.0, 0.0
    xf, yf, zf = ponto_final

    dx = (xf - x0)/num_segmentos
    dy = (yf - y0)/num_segmentos
    dz = (zf - z0)/num_segmentos

    nodes = []

    # --- LÓGICA CONDICIONAL PARA GERAÇÃO DE NÓS ---
    if ID == 1 or ID == 2:
      largura_inicial = l_base; altura_inicial = h_base
      lado_final = 3.0; largura_final = lado_final; altura_final = lado_final*math.sqrt(3)/2.0

      for i in range(num_segmentos + 1):
        x_c = x0 + dx*i; y_c = y0 + dy*i; z_c = z0 + dz*i
        taper = float(i)/num_segmentos
        largura_atual = largura_inicial*(1 - taper) + largura_final*taper
        altura_atual = altura_inicial*(1 - taper) + altura_final*taper

        if ID == 1: # Triângulo aponta para +X
          topo = [x_c, y_c, z_c]
          base_esq = [x_c - altura_atual, y_c - largura_atual/2, z_c]
          base_dir = [x_c - altura_atual, y_c + largura_atual/2, z_c]
          nodes.append(topo); nodes.append(base_esq); nodes.append(base_dir)
        else: # ID == 2, Triângulo aponta para -X
          base_esq = [x_c, y_c - largura_atual/2, z_c]
          base_dir = [x_c, y_c + largura_atual/2, z_c]
          topo = [x_c - altura_atual, y_c, z_c]
          nodes.append(topo); nodes.append(base_esq); nodes.append(base_dir)

    elif ID == 3:
      # Para ID=3, l_base é a largura (Y) e h_base é a altura (X)
      largura_inicial = l_base
      altura_inicial = h_base
      largura_final = 3.0
      altura_final = 3.0

      for i in range(num_segmentos + 1):
        x_c = x0 + dx*i; y_c = y0 + dy*i; z_c = z0 + dz*i
        taper = float(i)/num_segmentos

        # Interpolar largura e altura de forma independente
        largura_atual = largura_inicial*(1 - taper) + largura_final*taper
        altura_atual = altura_inicial*(1 - taper) + altura_final*taper

        meia_largura = largura_atual/2.0
        meia_altura = altura_atual/2.0

        # Nós do retângulo centrados em (x_c, y_c)
        no_inf_esq = [x_c - meia_altura, y_c - meia_largura, z_c]
        no_inf_dir = [x_c - meia_altura, y_c + meia_largura, z_c]
        no_sup_dir = [x_c + meia_altura, y_c + meia_largura, z_c]
        no_sup_esq = [x_c + meia_altura, y_c - meia_largura, z_c]
        nodes.append(no_inf_esq); nodes.append(no_inf_dir); nodes.append(no_sup_dir); nodes.append(no_sup_esq)

    nodes = np.array(nodes)

    # --- LÓGICA DE GERAÇÃO DE ELEMENTOS ---
    elements = []
    if ID == 1 or ID == 2:
      for i in range(num_segmentos):
        topo_i = i*3 + 1; esq_i = i*3 + 2; dir_i = i*3 + 3
        topo_i1 = (i + 1) * 3 + 1; esq_i1 = (i + 1) * 3 + 2; dir_i1 = (i + 1) * 3 + 3
        elements.append([topo_i, topo_i1]); elements.append([esq_i, esq_i1]); elements.append([dir_i, dir_i1])
        elements.append([topo_i, esq_i]); elements.append([esq_i, dir_i]); elements.append([dir_i, topo_i])
        elements.append([topo_i, esq_i1]); elements.append([topo_i, dir_i1]); elements.append([esq_i, dir_i1])
      final = num_segmentos
      topo_f = final*3 + 1; esq_f = final*3 + 2; dir_f = final*3 + 3
      elements.append([topo_f, esq_f]); elements.append([esq_f, dir_f]); elements.append([dir_f, topo_f])

      print(elements)

    elif ID == 3:
      for i in range(num_segmentos):
        ie_i = i*4 + 1; id_i = i*4 + 2; sd_i = i*4 + 3; se_i = i*4 + 4
        ie_i1 = (i + 1)*4 + 1; id_i1 = (i + 1)*4 + 2; sd_i1 = (i + 1)*4 + 3; se_i1 = (i + 1)*4 + 4
        elements.append([ie_i, ie_i1]); elements.append([id_i, id_i1]); elements.append([sd_i, sd_i1]); elements.append([se_i, se_i1])
        elements.append([ie_i, id_i]); elements.append([id_i, sd_i]); elements.append([sd_i, se_i]); elements.append([se_i, ie_i])
        elements.append([ie_i, sd_i])
        elements.append([ie_i, id_i1]); elements.append([se_i, sd_i1]); elements.append([ie_i, se_i1]); elements.append([id_i, sd_i1])
      final = num_segmentos
      ie_f = final * 4 + 1; id_f = final * 4 + 2; sd_f = final * 4 + 3; se_f = final * 4 + 4
      elements.append([ie_f, id_f]); elements.append([id_f, sd_f]); elements.append([sd_f, se_f]); elements.append([se_f, ie_f])
      elements.append([ie_f, sd_f])

    elements = np.array(list(set(map(tuple, map(sorted, elements)))))


    with open(filename, 'w') as f:
        f.write("Ensight New mesh step 1 \n"); f.write("mesh\n")
        f.write("node id given\nelement id given\n"); f.write("coordinates\n")
        f.write(f"{len(nodes):8d}\n")
        for i, node in enumerate(nodes):
            f.write(f"{i+1:8d}{node[0]:12.5E}{node[1]:12.5E}{node[2]:12.5E}\n")
        f.write("part 1\n"); f.write("material_1\n"); f.write("bar2\n")
        f.write(f"{len(elements):8d}\n")
        for i, elem in enumerate(elements):
            f.write(f"{i+1:8d}{elem[0]:8d}{elem[1]:8d}\n")

    print(f"Arquivo '{filename}' gerado com sucesso.")
    return nodes, elements

################################################################################

# Plots
################################################################################

def plotar_estrutura(nodes, elements):
  fig = plt.figure(figsize=(15, 10)); ax = fig.add_subplot(111, projection='3d')
  for elem in elements:
    n1, n2 = nodes[elem[0]], nodes[elem[1]]
    ax.plot([n1[0], n2[0]], [n1[1], n2[1]], [n1[2], n2[2]],'k-', lw=1.5)
  ax.set_box_aspect((np.ptp(nodes[:,0]), np.ptp(nodes[:,1]), np.ptp(nodes[:,2])))
  ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)'); ax.set_zlabel('Z (m)')
  ax.set_title('Estrutura Original (Não Deformada)'); ax.view_init(elev=25., azim=-100)
  plt.grid(True); plt.show()

def plotar_estrutura_deformada(nodes, elements, U, escala_deformacao):

    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection='3d')
    deslocamentos = U.reshape((len(nodes), 6))[:, :3]
    nodes_deformados = nodes + deslocamentos * escala_deformacao

    # Linhas originais
    for i, elem in enumerate(elements):

        n1, n2 = nodes[elem[0]], nodes[elem[1]]
        ax.plot([n1[0], n2[0]],
                [n1[1], n2[1]],
                [n1[2], n2[2]],
                '--', lw=1.5, color='gray',
                label='Original' if i == 0 else "")

    # Linhas deformadas
    for i, elem in enumerate(elements):

        n1, n2 = nodes_deformados[elem[0]], nodes_deformados[elem[1]]
        ax.plot([n1[0], n2[0]],
                [n1[1], n2[1]],
                [n1[2], n2[2]],
                '-', lw=2, color='red',
                label='Deformada' if i == 0 else "")
    ax.set_box_aspect((np.ptp(nodes[:,0]),
                       np.ptp(nodes[:,1]),
                       np.ptp(nodes[:,2])))
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.legend()
    ax.set_title(f'Estrutura Original vs. Deformada (Escala: {escala_deformacao}x)')

    # ======== Vista no plano XZ (modo 3D) ========
    ax.view_init(elev=0, azim=-90)

    #ax.view_init(elev=90, azim=-90)

    plt.grid(True)
    plt.show()

def plotar_tensoes(nodes, elements, stresses, U, escala_deformacao):
    fig = plt.figure(figsize=(17, 10)); ax = fig.add_subplot(111, projection='3d')
    deslocamentos = U.reshape((len(nodes), 6))[:, :3]
    nodes_deformados = nodes + deslocamentos * escala_deformacao
    cmap = plt.get_cmap('coolwarm'); norm = plt.Normalize(vmin=stresses.min(), vmax=stresses.max())

    for i, elem in enumerate(elements):
        n1, n2 = nodes_deformados[elem[0]], nodes_deformados[elem[1]]

        ax.plot([n1[0], n2[0]], [n1[1], n2[1]], [n1[2], n2[2]],
                        color=cmap(norm(stresses[i])), lw=3)

    ax.set_box_aspect((np.ptp(nodes[:,0]), np.ptp(nodes[:,1]), np.ptp(nodes[:,2])))
    ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)'); ax.set_zlabel('Z (m)')
    ax.set_title(f'Tensões e Falhas na Estrutura Deformada (Escala: {escala_deformacao}x)')
    ax.view_init(elev=30, azim=-60)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
    cbar = fig.colorbar(sm, shrink=0.8, aspect=10, ax=ax, pad=0.1)
    cbar.set_label('Tensão Axial (Pa) ')
    ax.view_init(elev=25., azim=-100); plt.grid(True); plt.show()

################################################################################

# FEM
################################################################################

def fem_solver_3d(nodes, elements, ID):

  # Parâmetros do material
  E = 210e9; G = 80e9; A = 0.005
  Iz = 1e-5; Iy = 1e-5; J = 2e-5

  num_nodes = len(nodes)
  num_dof = num_nodes * 6
  K_global = np.zeros((num_dof, num_dof))

  # Loop nos elementos
  for elem in elements:

    n1_idx, n2_idx = elem
    n1, n2 = nodes[n1_idx], nodes[n2_idx]
    L = np.linalg.norm(n2 - n1)
    if L == 0:
      raise ValueError(f"Elemento com comprimento zero entre nós {n1_idx} e {n2_idx}")

    # Cossenos diretores
    cx, cy, cz = (n2 - n1)/L

    # Matriz de rotação 3D T, R são as componentes de cada eixo
    l = np.sqrt(cx**2 + cy**2)
    R = np.array([
          [cx, cy, cz],
          [-cy/l, cx/l, 0.0],
          [-cx*cz/l, -cy*cz/l, l]
          ])

    T = np.zeros((12, 12))
    for k in range(2):
      T[k*6:(k+1)*6, k*6:(k+1)*6] = np.block([
        [R, np.zeros((3, 3))],
        [np.zeros((3, 3)), R]
        ])

    # Calcula matriz de rigidez local e transforma para global
    k_local = np.zeros((12, 12))
    k_local[0,0] = k_local[6,6] = E*A/L; k_local[0,6] = k_local[6,0] = -E*A/L
    k_local[1,1] = k_local[7,7] = 12*E*Iz/L**3; k_local[1,7] = k_local[7,1] = -12*E*Iz/L**3
    k_local[1,5] = k_local[5,1] = k_local[1,11] = k_local[11,1] = 6*E*Iz/L**2
    k_local[7,5] = k_local[5,7] = k_local[7,11] = k_local[11,7] = -6*E*Iz/L**2
    k_local[2,2] = k_local[8,8] = 12*E*Iy/L**3; k_local[2,8] = k_local[8,2] = -12*E*Iy/L**3
    k_local[2,4] = k_local[4,2] = k_local[2,10] = k_local[10,2] = -6*E*Iy/L**2
    k_local[8,4] = k_local[4,8] = k_local[8,10] = k_local[10,8] = 6*E*Iy/L**2
    k_local[3,3] = k_local[9,9] = G*J/L; k_local[3,9] = k_local[9,3] = -G*J/L
    k_local[4,4] = k_local[10,10] = 4*E*Iy/L; k_local[4,10] = k_local[10,4] = 2*E*Iy/L
    k_local[5,5] = k_local[11,11] = 4*E*Iz/L; k_local[5,11] = k_local[11,5] = 2*E*Iz/L

    k_global_elem = T.T@k_local@T

    # Assembly
    dofs = np.concatenate([np.arange(n1_idx*6, n1_idx*6+6), np.arange(n2_idx*6, n2_idx*6+6)])
    for ii in range(12):
      for jj in range(12):
          K_global[dofs[ii], dofs[jj]] += k_global_elem[ii, jj]

  # Vetor de forças global
  F = np.zeros(num_dof)
  total_force = -120000  # Força total a ser aplicada na direção -X (para baixo)
  # Encontra todos os nós na extremidade da treliça (maior Z)
  max_z = np.max(nodes[:, 2])
  end_nodes_indices = np.where(np.isclose(nodes[:, 2], max_z))[0]

  if ID == 1:
    # Para o triângulo apontando para +X, a força vai no vértice (maior X)
    target = end_nodes_indices[np.argmax(nodes[end_nodes_indices, 0])]
    F[target*6 + 2] = total_force

  elif ID == 2:
    # Para o triângulo apontando para -X, a força vai no vértice (menor X)
    target = end_nodes_indices[np.argmin(nodes[end_nodes_indices, 0])]
    F[target * 6 + 2] = total_force

  elif ID == 3:
    # Para o retângulo, distribui a força nos dois nós superiores (maior X)
    end_x_coords = nodes[end_nodes_indices, 0]
    max_x_val = np.max(end_x_coords)
    targets = end_nodes_indices[np.isclose(end_x_coords, max_x_val)]
    force = total_force/len(targets)
    for idx in targets:
      F[idx*6 + 2] = force


  # Condições de contorno (engasta os nós da base)
  dofs_fixos = []
  nodes_engaste_indices = np.where(np.isclose(nodes[:, 2], 0))[0]
  for node_idx in nodes_engaste_indices:
    dofs_fixos.extend(range(node_idx*6, node_idx*6 + 6))

  # Remove os dofs inativos e resolve sistema linear
  active_dofs = np.setdiff1d(np.arange(num_dof), dofs_fixos)
  K_reduced = K_global[np.ix_(active_dofs, active_dofs)]
  F_reduced = F[active_dofs]

  U_reduced = np.linalg.solve(K_reduced, F_reduced)

  U = np.zeros(num_dof)
  U[active_dofs] = U_reduced

  # Cálculo de tensões
  tensoes = np.zeros(len(elements))
  for i, elem in enumerate(elements):
      n1_idx, n2_idx = elem;
      n1, n2 = nodes[n1_idx], nodes[n2_idx]

      L = np.linalg.norm(n2 - n1);
      if L == 0:
        raise ValueError(f"Elemento com comprimento zero entre nós {n1_idx} e {n2_idx}")

      cx, cy, cz = (n2 - n1)/L
      l = np.sqrt(cx**2 + cy**2)
      R = np.array([
          [cx, cy, cz],
          [-cy/l, cx/l, 0.0],
          [-cx*cz/l, -cy*cz/l, l]
          ])

      T = np.zeros((12, 12))
      for k in range(2):
        T[k*6:(k+1)*6, k*6:(k+1)*6] = np.block([
          [R, np.zeros((3, 3))],
          [np.zeros((3, 3)), R]
          ])

      dofs = np.concatenate([np.arange(n1_idx*6, n1_idx*6+6), np.arange(n2_idx*6, n2_idx*6+6)])
      U_local_elem = T@U[dofs]
      delta_L = U_local_elem[6] - U_local_elem[0]

      # S11
      tensoes[i] = E*(delta_L/L)

  return U, tensoes

################################################################################

# Main
################################################################################

# gerador de geometria parametrizado
# ID 1 -> TRIÂNGULO EM PÉ
# ID 2 -> TRIÂNGULO DE CABEÇA PRA BAIXO
# ID 3 -> SEÇÃO RETANGULAR
nodes, elements = geo_generator(
        filename='trelica.geo',
        num_segmentos=4,
        l_base=3.5,
        h_base=5.5,
        ponto_final=(-30.0, 0.0, 15.0),
        ID = 1
    )

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

################################################################################
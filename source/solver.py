import numpy as np

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
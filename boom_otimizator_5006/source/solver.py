import numpy as np

def fem_solver_3d(nodes, elements, ID, props_IDs, biblioteca_perfis, forces):

    # Forças aplicadas
    ###########################################################################################

    total_force_x = forces['total_force_x']
    total_force_y = forces['total_force_y']
    total_force_z = forces['total_force_z']

    ###########################################################################################

    num_modos = 5; num_nodes = len(nodes); num_dof = num_nodes*6
    K_global = np.zeros((num_dof, num_dof)); K_geom = np.zeros((num_dof, num_dof))
    # Matriz de Rigidez Global               # Matriz Geométrica Global

    # Dicionário de busca para propriedades dos perfis com base no ID
    prop_map = {perfil['ID']: perfil for perfil in biblioteca_perfis}

    # Montagem da matriz de rigidez global (K_global)
    ###########################################################################################

    for i, elem in enumerate(elements):

        # Identifica o ID do material deste elemento específico e busca as propriedades
        id_elemento = props_IDs[i]; p = prop_map[id_elemento]
        
        E = p['E']; G = p['G']; A = p['A']; Iz = p['Iz']; Iy = p['Iy']; J = p['J']
        
        n1_idx, n2_idx = elem
        n1, n2 = nodes[n1_idx], nodes[n2_idx]
        L = np.linalg.norm(n2 - n1)
        cx, cy, cz = (n2 - n1)/L     # Cossenos diretos
        l = np.sqrt(cx**2 + cy**2)   # Comprimento projetado
        
        if l < 1e-6: # Tratamento de erro p/ elemento vertical
            R = np.array([[0, 0, np.sign(cz)],[0, 1, 0],[-np.sign(cz), 0, 0]])
        else:
            R = np.array([            # Matriz de rotação (3x3) 
                  [cx, cy, cz],
                  [-cy/l, cx/l, 0.0],[-cx*cz/l, -cy*cz/l, l]
                  ])
            
        T = np.zeros((12, 12))
        for k in range(2):
            T[k*6:(k+1)*6, k*6:(k+1)*6] = np.block([[R, np.zeros((3, 3))],[np.zeros((3, 3)), R]])

        # Matriz local elástica (k_local) com as propriedades locais do elemento
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

        k_global_elem = T.T@k_local@T   # Passando para coordenadas globais

        # Assembly 
        dofs = np.concatenate([np.arange(n1_idx*6, n1_idx*6+6), np.arange(n2_idx*6, n2_idx*6+6)])
        for ii in range(12):
            for jj in range(12):
                K_global[dofs[ii], dofs[jj]] += k_global_elem[ii, jj]

    ###########################################################################################

    # Aplicação das cargas
    ###########################################################################################

    F = np.zeros(num_dof)
    max_z = np.max(nodes[:, 2])
    end_nodes_indices = np.where(np.isclose(nodes[:, 2], max_z))[0]

    if ID == 1:
        target = end_nodes_indices[np.argmax(nodes[end_nodes_indices, 0])]
        F[target*6 + 0] = total_force_x; F[target*6 + 1] = total_force_y; F[target*6 + 2] = total_force_z
    elif ID == 2:
        target = end_nodes_indices[np.argmin(nodes[end_nodes_indices, 0])]
        F[target*6 + 0] = total_force_x; F[target*6 + 1] = total_force_y; F[target*6 + 2] = total_force_z
    elif ID == 3:
        end_x_coords = nodes[end_nodes_indices, 0]
        max_x_val = np.max(end_x_coords)
        targets = end_nodes_indices[np.isclose(end_x_coords, max_x_val)]
        force_x = total_force_x/len(targets); force_y = total_force_y/len(targets); force_z = total_force_z/len(targets)
        for idx in targets:
            F[idx*6 + 0] = force_x; F[idx*6 + 1] = force_y; F[idx*6 + 2] = force_z

    ###########################################################################################

    # Aplicação dos engastes
    ###########################################################################################

    dofs_fixos =[]
    nodes_engaste_indices = np.where(np.isclose(nodes[:, 2], 0))[0]
    for node_idx in nodes_engaste_indices:
        dofs_fixos.extend(range(node_idx*6, node_idx*6 + 6))

    active_dofs = np.setdiff1d(np.arange(num_dof), dofs_fixos)

    ###########################################################################################
    
    # Resolução do sistema linear
    ###########################################################################################

    K_f = K_global[np.ix_(active_dofs, active_dofs)]
    F_f = F[active_dofs]

    # Solução do sistema linear: K*U = F
    U_f = np.linalg.solve(K_f, F_f)
    
    U = np.zeros(num_dof)
    U[active_dofs] = U_f

    ###########################################################################################

    # Cálculo de tensões e montagem da matriz de rigidez geométrica
    ###########################################################################################

    tensoes = np.zeros(len(elements))
    elementos_falha =[]
    
    for i, elem in enumerate(elements):
        
        # Resgata novamente as propriedades para o cálculo de tensões locais
        id_elemento = props_IDs[i]
        p = prop_map[id_elemento]
        
        E = p['E']; G = p['G']; A = p['A']; Iz = p['Iz']; Iy = p['Iy']; J = p['J']; sigma_y = p['sigma_y']; r_geom = p['r_geom']
        
        n1_idx, n2_idx = elem
        n1, n2 = nodes[n1_idx], nodes[n2_idx]
        L = np.linalg.norm(n2 - n1)
        
        cx, cy, cz = (n2 - n1)/L
        l = np.sqrt(cx**2 + cy**2)

        if l < 1e-6: 
            R = np.array([[0, 0, np.sign(cz)],[0, 1, 0],[-np.sign(cz), 0, 0]])
        else:
            R = np.array([[cx, cy, cz],[-cy/l, cx/l, 0.0],[-cx*cz/l, -cy*cz/l, l]])

        T = np.zeros((12, 12))
        for k in range(2):
            T[k*6:(k+1)*6, k*6:(k+1)*6] = np.block([[R, np.zeros((3, 3))], [np.zeros((3, 3)), R]])

        # É necessário recalcular o K_local para extrair os esforços internos corretamente
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

        dofs = np.concatenate([np.arange(n1_idx*6, n1_idx*6+6), np.arange(n2_idx*6, n2_idx*6+6)])
        U_local_elem = T@U[dofs]                    # Deslocamentos locais

        F_local = k_local @ U_local_elem            # Forças locais
        
        # --- Cálculo de Tensões devido a normal e flexão ---

        N = F_local[6] # Força Axial 
        My_1, Mz_1 = F_local[4], F_local[5]   # Momentos fletores nos nós
        My_2, Mz_2 = F_local[10], F_local[11]

        sigma_axial = N/A                     # Tensão axial
        res_M1 = np.sqrt(My_1**2 + Mz_1**2)   
        res_M2 = np.sqrt(My_2**2 + Mz_2**2)   
        max_M = max(res_M1, res_M2)           # Maior momento fletor ao longo da barra
        sigma_flex = (max_M*r_geom)/Iz        # Tensão de flexão máxima

        s_max_tensao = sigma_axial + sigma_flex
        s_max_compressao = sigma_axial - sigma_flex
        tensao_critica = s_max_tensao if abs(s_max_tensao) > abs(s_max_compressao) else s_max_compressao

        tensoes[i] = tensao_critica
        if abs(tensao_critica) > sigma_y:
            elementos_falha.append(i)   
      
        # Matriz de rigidez geométrica 
        coef = N/(30.0*L)
        k_geom_local = np.zeros((12, 12))
        
        k_geom_local[1,1] = k_geom_local[7,7] = 36.0*coef
        k_geom_local[1,7] = k_geom_local[7,1] = -36.0*coef
        k_geom_local[1,5] = k_geom_local[5,1] = 3.0*L*coef
        k_geom_local[1,11] = k_geom_local[11,1] = 3.0*L*coef
        k_geom_local[7,5] = k_geom_local[5,7] = -3.0*L*coef
        k_geom_local[7,11] = k_geom_local[11,7] = -3.0*L*coef
        k_geom_local[5,5] = k_geom_local[11,11] = 4.0*L**2*coef
        k_geom_local[5,11] = k_geom_local[11,5] = -L**2*coef

        k_geom_local[2,2] = k_geom_local[8,8] = 36.0*coef
        k_geom_local[2,8] = k_geom_local[8,2] = -36.0*coef
        k_geom_local[2,4] = k_geom_local[4,2] = -3.0*L*coef
        k_geom_local[2,10] = k_geom_local[10,2] = -3.0*L*coef
        k_geom_local[8,4] = k_geom_local[4,8] = 3.0*L*coef
        k_geom_local[8,10] = k_geom_local[10,8] = 3.0*L*coef
        k_geom_local[4,4] = k_geom_local[10,10] = 4.0*L**2*coef
        k_geom_local[4,10] = k_geom_local[10,4] = -L**2*coef
        
        # Rotação para sistema global e soma
        k_geom_global = T.T@k_geom_local@T
        
        for ii in range(12):
            for jj in range(12):
                K_geom[dofs[ii], dofs[jj]] += k_geom_global[ii, jj]

    ###########################################################################################

    # Análise de flambagem (problema de autovalores)           
    ###########################################################################################

    K_g_f = K_geom[np.ix_(active_dofs, active_dofs)]
    
    D = np.linalg.solve(K_f, -K_g_f)   
    vals, _ = np.linalg.eig(D)
    vals = np.real(vals)
    valid_indices = np.where(vals > 1e-9)[0] 
    
    lambdas_inv = vals[valid_indices]
    lambdas = 1.0/lambdas_inv
    lambdas = np.sort(lambdas)[:num_modos]
    
    ###########################################################################################

    return U, tensoes, elementos_falha, lambdas
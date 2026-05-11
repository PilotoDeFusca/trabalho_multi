import numpy as np

def fem_solver_3d_sens(nodes, elements, ID, props, forces, areas):

    E = props['E']; G = props['G']; sigma_y = props['sigma_y']
    
    num_elems = len(elements); num_nodes = len(nodes); num_dof = num_nodes*6
    
    K_global = np.zeros((num_dof, num_dof))
    
    # Armazenamento de dados para não recalcular na fase de sensibilidade
    elem_data =[]

    # =========================================================================
    # --- MONTAGEM DA MATRIZ DE RIGIDEZ GLOBAL ---
    # =========================================================================
    for i, elem in enumerate(elements):

        A = areas[i]
        
        # Assumindo proporção r_int/r_ext = 0.5 
        c_ratio = 0.5
        R_ext = np.sqrt(A/(np.pi*(1.0 - c_ratio**2)))
        Iy = Iz = (np.pi/4.0)*(R_ext**4 - (c_ratio*R_ext)**4); J = 2.0*Iy

        n1_idx, n2_idx = elem
        n1, n2 = nodes[n1_idx], nodes[n2_idx]
        L = np.linalg.norm(n2 - n1)
        cx, cy, cz = (n2 - n1)/L
        l = np.sqrt(cx**2 + cy**2)
        
        if l < 1e-6:
            R = np.array([[0, 0, np.sign(cz)],[0, 1, 0], [-np.sign(cz), 0, 0]])
        else:
            R = np.array([[cx, cy, cz], [-cy/l, cx/l, 0.0], [-cx*cz/l, -cy*cz/l, l]])
            
        T = np.zeros((12, 12))
        for k in range(2):
            T[k*6:(k+1)*6, k*6:(k+1)*6] = np.block([[R, np.zeros((3, 3))], [np.zeros((3, 3)), R]])

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
        
        k_global_elem = T.T @ k_local @ T
        for ii in range(12):
            for jj in range(12):
                K_global[dofs[ii], dofs[jj]] += k_global_elem[ii, jj]

        # Salva dados do elemento (agora guardando o Raio Externo e inércias locais)
        elem_data.append({'T': T, 'k_local': k_local, 'dofs': dofs, 'L': L, 'A': A, 'R_ext': R_ext, 'Iy': Iy, 'J': J})

    # =========================================================================
    # --- APLICAÇÃO DE CARGAS E ENGASTES ---
    # =========================================================================
    F = np.zeros(num_dof)
    max_z = np.max(nodes[:, 2])
    end_nodes_indices = np.where(np.isclose(nodes[:, 2], max_z))[0]

    if ID == 1:
        target = end_nodes_indices[np.argmax(nodes[end_nodes_indices, 0])]
        F[target*6:target*6+3] =[forces['total_force_x'], forces['total_force_y'], forces['total_force_z']]
    elif ID == 2:
        target = end_nodes_indices[np.argmin(nodes[end_nodes_indices, 0])]
        F[target*6:target*6+3] = [forces['total_force_x'], forces['total_force_y'], forces['total_force_z']]
    elif ID == 3:
        end_x_coords = nodes[end_nodes_indices, 0]
        max_x_val = np.max(end_x_coords)
        targets = end_nodes_indices[np.isclose(end_x_coords, max_x_val)]
        for idx in targets:
            F[idx*6:idx*6+3] =[forces['total_force_x']/len(targets), forces['total_force_y']/len(targets), forces['total_force_z']/len(targets)]

    dofs_fixos =[]
    nodes_engaste_indices = np.where(np.isclose(nodes[:, 2], 0))[0]
    for node_idx in nodes_engaste_indices:
        dofs_fixos.extend(range(node_idx*6, node_idx*6 + 6))

    active_dofs = np.setdiff1d(np.arange(num_dof), dofs_fixos)

    K_f = K_global[np.ix_(active_dofs, active_dofs)]
    F_f = F[active_dofs]
    
    # Resolve sistema linear base
    U_f = np.linalg.solve(K_f, F_f)
    U = np.zeros(num_dof)
    U[active_dofs] = U_f

    # Inversão da matriz de rigidez global (K_f) para uso rápido nas derivadas
    K_f_inv = np.linalg.inv(K_f)

    # =========================================================================
    # --- TENSÕES E MATRIZ GEOMÉTRICA ---
    # =========================================================================
    tensoes = np.zeros(num_elems)
    elementos_falha =[]
    K_geom = np.zeros((num_dof, num_dof))

    for i in range(num_elems):

        data = elem_data[i]
        dofs, T, k_local = data['dofs'], data['T'], data['k_local']
        L, A, R_ext, Iy = data['L'], data['A'], data['R_ext'], data['Iy']

        U_local_elem = T@U[dofs]
        F_local = k_local@U_local_elem
        data['U_loc'] = U_local_elem 
        
        N = F_local[6]
        My_1, Mz_1 = F_local[4], F_local[5]
        My_2, Mz_2 = F_local[10], F_local[11]

        data['N'], data['My1'], data['Mz1'], data['My2'], data['Mz2'] = N, My_1, Mz_1, My_2, Mz_2

        sigma_axial = N / A
        M1 = np.sqrt(My_1**2 + Mz_1**2)
        M2 = np.sqrt(My_2**2 + Mz_2**2)
        max_M = M1 if M1 > M2 else M2
        
        data['M_max'] = max_M
        data['M_max_idx'] = 1 if M1 > M2 else 2
        
        # A tensão de flexão MÁXIMA ocorre na fibra externa (R_ext)
        sigma_flex = (max_M*R_ext)/Iy

        s_max_tensao = sigma_axial + sigma_flex
        s_max_compressao = sigma_axial - sigma_flex
        
        if abs(s_max_tensao) > abs(s_max_compressao):
            tensao_critica = s_max_tensao
            data['S_sign'] = 1.0 
        else:
            tensao_critica = s_max_compressao
            data['S_sign'] = -1.0

        tensoes[i] = tensao_critica
        if abs(tensao_critica) > sigma_y:
            elementos_falha.append(i)   

        # Rigidez Geométrica local
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
        
        k_geom_global = T.T@k_geom_local@T
        for ii in range(12):
            for jj in range(12):
                K_geom[dofs[ii], dofs[jj]] += k_geom_global[ii, jj]

    # =========================================================================
    # --- FLAMBAGEM (Autovalores e Autovetor) ---
    # =========================================================================
    K_g_f = K_geom[np.ix_(active_dofs, active_dofs)]
    try:
        D = np.linalg.solve(K_f, -K_g_f)
        vals, vecs = np.linalg.eig(D)
        
        valid_indices = np.where(np.real(vals) > 1e-9)[0]
        if len(valid_indices) > 0:
            idx_max = valid_indices[np.argmax(np.real(vals[valid_indices]))]
            lambda_1 = 1.0 / np.real(vals[idx_max])
            lambdas = np.array([lambda_1])
            phi_f = np.real(vecs[:, idx_max])
            
            norm_phi = np.linalg.norm(phi_f)
            if norm_phi > 1e-12:
                phi_f = phi_f / norm_phi
            else:
                phi_f = np.zeros_like(phi_f)
        else:
            lambdas = np.array([1000.0])
            phi_f = np.zeros(len(active_dofs))
            
    except np.linalg.LinAlgError:
        lambdas = np.array([1e-6])
        phi_f = np.zeros(len(active_dofs))


    # =========================================================================
    # --- CÁLCULO DAS MATRIZES DE SENSIBILIDADE ---
    # =========================================================================

    d_sigma = np.zeros((num_elems, num_elems)); d_lambda = np.zeros(num_elems)

    phi_full = np.zeros(num_dof)
    phi_full[active_dofs] = phi_f
    lambda_val = lambdas[0]

    for i in range(num_elems):

        data = elem_data[i]; A = data['A']; L = data['L']; T = data['T']; Iy_loc = data['Iy']; J_loc = data['J']
        
        # Derivadas analíticas exatas para tubos  Como I é proporcional a A^2 (I = k * A^2), então dI/dA = 2*k*A = 2*I/A
        dEA = E/L; dEI = 2.0*E*Iy_loc/A; dGJ = 2.0*G*J_loc/A
        
        dk_local = np.zeros((12, 12))

        dk_local[0,0] = dk_local[6,6] = dEA
        dk_local[0,6] = dk_local[6,0] = -dEA

        dk_local[1,1] = dk_local[7,7] = 12*dEI/L**3
        dk_local[1,7] = dk_local[7,1] = -12*dEI/L**3
        dk_local[1,5] = dk_local[5,1] = dk_local[1,11] = dk_local[11,1] = 6*dEI/L**2
        dk_local[7,5] = dk_local[5,7] = dk_local[7,11] = dk_local[11,7] = -6*dEI/L**2
        
        dk_local[2,2] = dk_local[8,8] = 12*dEI/L**3
        dk_local[2,8] = dk_local[8,2] = -12*dEI/L**3
        dk_local[2,4] = dk_local[4,2] = dk_local[2,10] = dk_local[10,2] = -6*dEI/L**2
        dk_local[8,4] = dk_local[4,8] = dk_local[8,10] = dk_local[10,8] = 6*dEI/L**2
        
        dk_local[3,3] = dk_local[9,9] = dGJ/L
        dk_local[3,9] = dk_local[9,3] = -dGJ/L
        
        dk_local[4,4] = dk_local[10,10] = 4*dEI/L
        dk_local[4,10] = dk_local[10,4] = 2*dEI/L
        
        dk_local[5,5] = dk_local[11,11] = 4*dEI/L
        dk_local[5,11] = dk_local[11,5] = 2*dEI/L
        
        data['dk_local'] = dk_local
        
        coef_hat = 1.0/(30.0*L)
        k_hat_geom_local = np.zeros((12, 12))
        k_hat_geom_local[1,1] = k_hat_geom_local[7,7] = 36.0*coef_hat
        k_hat_geom_local[1,7] = k_hat_geom_local[7,1] = -36.0*coef_hat
        k_hat_geom_local[1,5] = k_hat_geom_local[5,1] = 3.0*L*coef_hat
        k_hat_geom_local[1,11] = k_hat_geom_local[11,1] = 3.0*L*coef_hat
        k_hat_geom_local[7,5] = k_hat_geom_local[5,7] = -3.0*L*coef_hat
        k_hat_geom_local[7,11] = k_hat_geom_local[11,7] = -3.0*L*coef_hat
        k_hat_geom_local[5,5] = k_hat_geom_local[11,11] = 4.0*L**2*coef_hat
        k_hat_geom_local[5,11] = k_hat_geom_local[11,5] = -L**2*coef_hat
        k_hat_geom_local[2,2] = k_hat_geom_local[8,8] = 36.0*coef_hat
        k_hat_geom_local[2,8] = k_hat_geom_local[8,2] = -36.0*coef_hat
        k_hat_geom_local[2,4] = k_hat_geom_local[4,2] = -3.0*L*coef_hat
        k_hat_geom_local[2,10] = k_hat_geom_local[10,2] = -3.0*L * coef_hat
        k_hat_geom_local[8,4] = k_hat_geom_local[4,8] = 3.0*L*coef_hat
        k_hat_geom_local[8,10] = k_hat_geom_local[10,8] = 3.0*L*coef_hat
        k_hat_geom_local[4,4] = k_hat_geom_local[10,10] = 4.0*L**2*coef_hat
        k_hat_geom_local[4,10] = k_hat_geom_local[10,4] = -L**2*coef_hat
        
        data['k_hat_geom_global'] = T.T@k_hat_geom_local@T

    denom_lambda = phi_f.T@K_g_f@phi_f

    for j in range(num_elems):

        data_j = elem_data[j]; dofs_j = data_j['dofs']; T_j = data_j['T']; dk_local_j = data_j['dk_local']; U_loc_j = data_j['U_loc']
        
        pseudo_load = np.zeros(num_dof); pseudo_load[dofs_j] = T_j.T@(dk_local_j@U_loc_j)
        
        dU_f = - K_f_inv@pseudo_load[active_dofs]
        dU = np.zeros(num_dof); dU[active_dofs] = dU_f
        
        dN_dAj = np.zeros(num_elems)
        
        for i in range(num_elems):

            data_i = elem_data[i]; dofs_i = data_i['dofs']; T_i = data_i['T']; k_local_i = data_i['k_local']
            
            dU_loc_i = T_i@dU[dofs_i]
            
            dF_loc_i = k_local_i@dU_loc_i
            if i == j: 
                dF_loc_i += data_i['dk_local']@data_i['U_loc']
                
            dN = dF_loc_i[6]
            dN_dAj[i] = dN
            
            dMy_1, dMz_1 = dF_loc_i[4], dF_loc_i[5]
            dMy_2, dMz_2 = dF_loc_i[10], dF_loc_i[11]
            
            if data_i['M_max_idx'] == 1:
                My, Mz = data_i['My1'], data_i['Mz1']
                dMy, dMz = dMy_1, dMz_1
            else:
                My, Mz = data_i['My2'], data_i['Mz2']
                dMy, dMz = dMy_2, dMz_2
                
            M_max = data_i['M_max']
            if M_max > 1e-12:
                dM = (My*dMy + Mz*dMz)/M_max
            else:
                dM = 0.0
                
            A_i = data_i['A']; R_ext_i = data_i['R_ext']; I_i = data_i['Iy']
            S_sign = data_i['S_sign']
            
            d_sigma_implicit = dN/A_i + S_sign*(dM*R_ext_i/I_i)
            
            if i == j:
                d_sigma_explicit = - data_i['N']/(A_i**2) + S_sign*M_max*(-1.5*R_ext_i/(I_i*A_i))
            else:
                d_sigma_explicit = 0.0
                
            d_sigma[i, j] = d_sigma_implicit + d_sigma_explicit
            
        if abs(denom_lambda) > 1e-12: 
            phi_loc_j = T_j@phi_full[dofs_j]
            dKE_term = phi_loc_j.T@dk_local_j@phi_loc_j
            
            dKG_term = 0.0
            for i in range(num_elems):
                phi_i = phi_full[elem_data[i]['dofs']]
                k_hat_g = elem_data[i]['k_hat_geom_global']
                dKG_term += dN_dAj[i] * (phi_i.T @ k_hat_g @ phi_i)
                
            d_lambda[j] = - (dKE_term + lambda_val*dKG_term)/denom_lambda
        else:
            d_lambda[j] = 0.0

    return U, tensoes, elementos_falha, lambdas, d_sigma, d_lambda
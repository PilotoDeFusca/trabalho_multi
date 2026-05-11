import scipy.optimize as opt
import numpy as np

from solver_LP import *

def otimizador_LP(nodes, elements, biblioteca_props, forces, max_iter, move_limit, tolerancia, ID):
    
    num_elems = len(elements)

    # Extrai e ordena a biblioteca pelas áreas (do menor para o maior)
    biblioteca_ordenada = sorted(biblioteca_props, key=lambda x: x['A'])
    areas_lib = np.array([p['A'] for p in biblioteca_ordenada])
    ids_lib = np.array([p['ID'] for p in biblioteca_ordenada])
    
    # Limites baseados na biblioteca
    A_min = areas_lib[0]; A_max = areas_lib[-1]           
    
    # Usa propriedades do primeiro elemento como referência (assumindo mesmo material)
    props_base = biblioteca_ordenada[0]; sigma_y = props_base['sigma_y']  
    densidade = props_base['densidade']; lambda_min = 1.2               

    # Inicia a otimização com a maior área (A_max) para todas as barras (segurança inicial)
    A_atual = np.ones(num_elems)*A_max

    # Calcula os comprimentos das barras 
    L = np.zeros(num_elems)
    for i, elem in enumerate(elements - 1):

        n1_idx, n2_idx = elem
        L[i] = np.linalg.norm(nodes[n2_idx] - nodes[n1_idx])

    print("\n" + "="*60)
    print(" INICIANDO OTIMIZAÇÃO ESTRUTURAL CONTÍNUA (SLP) ")
    print("="*60)

    historico_volume = []

    for iteracao in range(max_iter):
        # Estrutura antes da otimização
        deslocamentos, tensoes, elementos_falha, lambdas, d_sigma, d_lambda = fem_solver_3d_sens(nodes, elements - 1, ID, props_base, forces, areas=A_atual)

        volume_atual = np.sum(A_atual*L); historico_volume.append(volume_atual)
        tensao_max = np.max(np.abs(tensoes)); lambda_atual = lambdas[0]

        print(f"Iteração {iteracao+1:02d} | Volume: {volume_atual:.6f} m³ | Max Tensão: {tensao_max/1e6:.2f} MPa | Fator Flambagem: {lambda_atual:.4f} | Peso: {volume_atual*densidade} kg")

        # Função Objetivo: Minimizar a variação de volume dV = Sum(L_i*dA_i)
        c = L
        A_ub = []; b_ub = []

        for i in range(num_elems):

            # Limite Superior e Inferior
            A_ub.append(d_sigma[i, :]); b_ub.append(sigma_y - tensoes[i])
            A_ub.append(-d_sigma[i, :]); b_ub.append(sigma_y + tensoes[i])
        
        # Restrição de Flambagem Global
        A_ub.append(-d_lambda); b_ub.append(lambda_atual - lambda_min)
    
        A_ub = np.array(A_ub); b_ub = np.array(b_ub)

        # Limites da área da seção transversal
        bounds = []
        for i in range(num_elems):

            lb = max(A_min - A_atual[i], -move_limit*A_atual[i])
            ub = min(A_max - A_atual[i], move_limit*A_atual[i])
            bounds.append((lb, ub))

        # Resolve o problema
        res = opt.linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')

        if not res.success:
            print(f" -> Falha na otimização na iteração {iteracao+1}: {res.message}")
            break

        dA = res.x; A_atual += dA

        # Verifica convergência 
        max_variacao = np.max(np.abs(dA/A_atual))
        if max_variacao < tolerancia:
            print(f"\n -> Convergência alcançada na iteração {iteracao+1}!")
            break

    print("\n" + "="*60)
    print(" MAPEANDO ÁREAS PARA A BIBLIOTECA DE PERFIS (ARREDONDAMENTO PARA CIMA) ")
    print("="*60)
    
    A_discreto = np.zeros(num_elems)
    IDs_elementos = np.zeros(num_elems, dtype=int)
    
    for i in range(num_elems):
        
        # Mapeia para o perfil da biblioteca com a área MAIS PRÓXIMA do valor otimizado
        idx = np.argmin(np.abs(areas_lib - A_atual[i]))
            
        A_discreto[i] = areas_lib[idx]
        IDs_elementos[i] = ids_lib[idx]

    return A_discreto, IDs_elementos, historico_volume, L
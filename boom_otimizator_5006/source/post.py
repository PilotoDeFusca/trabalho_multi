import numpy as np
import os
import glob

def data_clean(path):
    files = glob.glob(path)
    for f in files:
        os.remove(f)

def printer(nodes, elements, elementos_falha, deslocamentos, tensoes, lambdas, props_IDs, biblioteca_perfis):

    # Mapeamento rápido de propriedades pelo ID
    prop_map = {perfil['ID']: perfil for perfil in biblioteca_perfis}

    volume_total = 0.0; comp_total = 0.0; massa_total_kg = 0.0

    # Iterar sobre cada elemento e calcular suas grandezas de forma independente
    for i, elem in enumerate(elements):
        
        # Resgatar propriedades do elemento atual
        id_elemento = props_IDs[i]
        p = prop_map[id_elemento]
        area_secao = p['A']
        densidade = p['densidade']

        ponto1 = nodes[elem[0]]
        ponto2 = nodes[elem[1]]
        comprimento_barra = np.linalg.norm(ponto2 - ponto1)
        
        # Incremento iterativo
        comp_total += comprimento_barra
        volume_elemento = area_secao * comprimento_barra
        volume_total += volume_elemento
        massa_total_kg += volume_elemento * densidade

    max_disp = np.max(np.abs(deslocamentos.reshape(-1, 6)[:, :3]))
    max_stress_tra = np.max(tensoes) if tensoes.size > 0 else 0
    min_stress_com = np.min(tensoes) if tensoes.size > 0 else 0

    print(f"Volume total: {volume_total:.6f} m^3")
    print(f"Deslocamento translacional máximo: {max_disp:.4f} m")
    print(f"Tensão máxima (Tração): {max_stress_tra} Pa")
    print(f"Tensão mínima (Compressão): {min_stress_com} Pa")
    print(f"Massa total da estrutura: {massa_total_kg:.10f} kg")
    print(f"Lista de elementos com falha: {elementos_falha}")
    print(f"Fatores de carga críticos: {lambdas}")
    print(f"Comprimento total da estrutura: {comp_total:.8f} m")


def writer(nodes, elements, elementos_falha, deslocamentos, tensoes, lambdas, valor_objetivo, path, props_IDs, biblioteca_perfis): 

    # Mapeamento rápido de propriedades pelo ID
    prop_map = {perfil['ID']: perfil for perfil in biblioteca_perfis}

    volume_total = 0.0; comp_total = 0.0; massa_total_kg = 0.0

    # Iterar sobre cada elemento e calcular suas grandezas de forma independente
    for i, elem in enumerate(elements):

        # Resgatar propriedades do elemento atual
        id_elemento = props_IDs[i]
        p = prop_map[id_elemento]
        area_secao = p['A']
        densidade = p['densidade']

        ponto1 = nodes[elem[0]]
        ponto2 = nodes[elem[1]]
        comprimento_barra = np.linalg.norm(ponto2 - ponto1)
        
        # Incremento iterativo
        comp_total += comprimento_barra
        volume_elemento = area_secao * comprimento_barra
        volume_total += volume_elemento
        massa_total_kg += volume_elemento * densidade

    max_disp = np.max(np.abs(deslocamentos.reshape(-1, 6)[:, :3]))
    max_stress_tra = np.max(tensoes) if tensoes.size > 0 else 0
    min_stress_com = np.min(tensoes) if tensoes.size > 0 else 0

    with open(path, 'a') as f:
        f.write(f'Valor funcao objetivo: {str(round(valor_objetivo,2))}\n')
        f.write(f"Volume total: {volume_total:.6f} m^3\n")
        f.write(f"Deslocamento translacional maximo: {max_disp:.4f} m\n")
        f.write(f"Tensao maxima (Tracao): {max_stress_tra} Pa\n")
        f.write(f"Tensao minima (Compressao): {min_stress_com} Pa\n")
        f.write(f"Massa total da estrutura: {massa_total_kg:.5f} kg\n")
        f.write(f"Lista de elementos com falha: {elementos_falha}\n")
        f.write(f"Fatores de carga criticos: {lambdas}\n")
        f.write(f"Comprimento total da estrutura: {comp_total:.8f} m\n\n\n")

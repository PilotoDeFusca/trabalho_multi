import numpy as np

def printer(nodes, elements, elementos_falha, deslocamentos, tensoes, lambdas): # Essa função devera escrever um arquivo de saída

    # Massa da estrutura
    densidade_aco = 7330.0; area_secao = 0.005; g = 9.81
    volume_total = 0.0
    for elem in (elements):
        ponto1 = nodes[elem[0]]
        ponto2 = nodes[elem[1]]
        comprimento_barra = np.linalg.norm(ponto2 - ponto1)
        volume_total += area_secao*comprimento_barra
    massa_total_kg = volume_total*densidade_aco

    max_disp = np.max(np.abs(deslocamentos.reshape(-1, 6)[:, :3]))
    max_stress_tra = np.max(tensoes) if tensoes.size > 0 else 0
    min_stress_com = np.min(tensoes) if tensoes.size > 0 else 0

    print(f"Volume total: {volume_total:.4f} m^3")
    print(f"Deslocamento translacional máximo: {max_disp:.4f} m")
    print(f"Tensão máxima (Tração): {max_stress_tra} Pa")
    print(f"Tensão mínima (Compressão): {min_stress_com} Pa")
    print(f"Massa total da estrutura: {massa_total_kg:.2f} kg")
    print(f"Lista de elementos com falha: {elementos_falha}")
    print(f"Fatores de carga críticos: {lambdas}")

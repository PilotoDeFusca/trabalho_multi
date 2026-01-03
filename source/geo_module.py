import numpy as np
import math
import os
import re

def geo_generator(filename, num_segmentos, l_base, h_base,  ponto_final, ID):
    num_segmentos=int(num_segmentos)

    x0, y0, z0 = 0.0, 0.0, 0.0
    xf, yf, zf = ponto_final

    dx = (xf - x0)/num_segmentos
    dy = (yf - y0)/num_segmentos
    dz = (zf - z0)/num_segmentos

    nodes = []

    # =====================
    # CONTROLE INTERNO: número de elementos por barra
    # 1  -> sem refinamento
    # 2+ -> cada barra (elemento) é subdividida em 'num_elementos_por_barra' elementos
    # Ajuste este valor para refinar a malha.
    num_elementos_por_barra = 2
    # =====================

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

    # --- REFINAMENTO: subdividir cada barra em múltiplos elementos ---
    if num_elementos_por_barra is not None and int(num_elementos_por_barra) > 1:
      nodes_list = nodes.tolist()  # mantém IDs 1-based na escrita
      refined_elements = []

      for elem in elements:
        n1_id = int(elem[0])
        n2_id = int(elem[1])

        p1 = np.array(nodes_list[n1_id - 1], dtype=float)
        p2 = np.array(nodes_list[n2_id - 1], dtype=float)

        prev_id = n1_id
        ndiv = int(num_elementos_por_barra)
        for div in range(1, ndiv):
          t = div / float(ndiv)
          novo_no = (1.0 - t) * p1 + t * p2
          nodes_list.append(novo_no.tolist())
          novo_id = len(nodes_list)  # ID 1-based
          refined_elements.append((prev_id, novo_id))
          prev_id = novo_id

        refined_elements.append((prev_id, n2_id))

      nodes = np.array(nodes_list, dtype=float)
      elements = np.array(list(set(map(tuple, map(sorted, refined_elements)))) , dtype=int)


    # --- NOMEANDO ARQUIVOS .GEO ---
    geo_path=f'data/{filename}'
    geo_path_id=1
    #while os.path.exists(f'../data/{filename}'+str(geo_path_id)):
    #   geo_path_id+=1
    geo_path+=str(geo_path_id)+'.geo'

    with open(geo_path, 'w') as f:
        f.write("Ensight New mesh step 1 \n"); f.write("mesh\n")
        f.write("node id given\nelement id given\n"); f.write("coordinates\n")
        f.write(f"{len(nodes):8d}\n")
        for i, node in enumerate(nodes):
            f.write(f"{i+1:8d}{node[0]:12.5E}{node[1]:12.5E}{node[2]:12.5E}\n")
        f.write("part 1\n"); f.write("material_1\n"); f.write("bar2\n")
        f.write(f"{len(elements):8d}\n")
        for i, elem in enumerate(elements):
            f.write(f"{i+1:8d}{elem[0]:8d}{elem[1]:8d}\n")


def geo_reader(filename):

    nodes = []
    elements = []
    
    with open(filename, 'r') as file:
        lines = file.readlines()
    
    i = 0
    num_lines = len(lines)
    
    # Skip header lines
    while i < num_lines and lines[i].strip() != "coordinates":
        i += 1
    
    # Parse nodes
    if i < num_lines and lines[i].strip() == "coordinates":
        i += 1
        
        # Read number of nodes
        num_nodes = int(lines[i].strip())
        i += 1
        
        # Read node data
        for _ in range(num_nodes):
            line = lines[i].strip()
            i += 1
            
            line = re.sub(r'(\d)(-)', r'\1 \2', line)
            
            parts = line.split()
            if len(parts) >= 4:
                x = float(parts[1])
                y = float(parts[2])
                z = float(parts[3])
                nodes.append((x, y, z))

    nodes = np.array(nodes)
    
    # Skip to elements section
    while i < num_lines and not (lines[i].strip().startswith("bar") or lines[i].strip().startswith("bar2")):
        i += 1
    
    # Parse elements
    if i < num_lines and (lines[i].strip() == "bar2" or lines[i].strip().startswith("bar")):
        i += 1
        
        # Read number of elements
        num_elements = int(lines[i].strip())
        i += 1
        
        # Read element data
        for _ in range(num_elements):
            line = lines[i].strip()
            i += 1
            
            parts = line.split()
            if len(parts) >= 3:
                # Extract connected nodes (bar2 elements have 2 nodes)
                node1 = int(parts[1])
                node2 = int(parts[2])
                elements.append((node1, node2))
                
    elements = np.array(list(set(map(tuple, map(sorted, elements)))))
    
    
    return nodes, elements

# Biblioteca de materiais e propriedades geométricas e forças aplicadas
#######################################################################

forces1 = {
    'total_force_x': 0.0,  # Força total aplicada na direção x
    'total_force_y': 0.0,    # Força total aplicada na direção y
    #'total_force_z': -45.0   # Força total aplicada na direção z
    'total_force_z': -100.0   # Força total aplicada na direção z
}

props_c1 = {
    'ID': 1,
    'E': 40e9,           # Módulo de Young (Pa) 
    'G': 4e9,            # Módulo de cisalhamento (Pa) - estimado para compósito epóxi/fibra de carbono
    'A': 9.4248e-6,      # Área da seção transversal (m^2) - tubo 4mm DE x 2mm DI
    'Iz': 1.1781e-11,    # Momento de inércia em torno do eixo z (m^4) - seção tubular
    'Iy': 1.1781e-11,    # Momento de inércia em torno do eixo y (m^4) - seção tubular
    'J': 2.3562e-11,     # Módulo de torção (m^4) - seção circular tubular
    'sigma_y': 600e6,    # Tensão máxima (Pa) - resistência à flexão 0° do PDF
    'r_geom': 0.0015,    # Raio geométrico médio (m) - (r_ext + r_int)/2 = (0.002 + 0.001)/2
    'densidade': 1500.0  # Densidade do material (kg/m^3) 
}

props_c2 = {
    'ID': 2,
    'E': 40e9,           # Módulo de Young (Pa) 
    'G': 4e9,            # Módulo de cisalhamento (Pa) 
    'A': 5.3014e-6,      # Área da seção transversal (m²) - tubo 3mm DE x 1.5mm DI
    'Iz': 3.7276e-12,    # Momento de inércia em torno do eixo z (m⁴) - seção tubular
    'Iy': 3.7276e-12,    # Momento de inércia em torno do eixo y (m⁴) - seção tubular
    'J': 7.4552e-12,     # Módulo de torção (m⁴) - seção circular tubular
    'sigma_y': 600e6,    # Tensão máxima (Pa) 
    'r_geom': 0.001125,  # Raio geométrico médio (m) - (r_ext + r_int)/2
    'densidade': 1500.0  # Densidade do material (kg/m³) 
}

#######################################################################

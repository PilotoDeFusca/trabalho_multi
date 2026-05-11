from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np

def plotar_estrutura(nodes, elements, props_IDs=None, biblioteca_perfis=None, ax=None, *, clear=True, title='Estrutura Original (Não Deformada)', view=(25.0, -100.0), savepath=None, show=True):

    nodes = np.asarray(nodes)
    elements = np.asarray(elements)

    if ax is None:
        fig = plt.figure(figsize=(15, 10))
        ax = fig.add_subplot(111, projection='3d')
    else:
        fig = ax.get_figure()
        if clear:
            ax.cla()

    # Se os IDs e a biblioteca forem passados, plota colorido
    if props_IDs is not None and biblioteca_perfis is not None:
        cores_disponiveis =['red', 'blue', 'green', 'orange', 'purple', 'cyan', 'magenta', 'brown', 'pink']
        mapa_cores = {}
        
        # Mapeia cada ID da biblioteca para uma cor
        for idx, perfil in enumerate(biblioteca_perfis):
            mapa_cores[perfil['ID']] = cores_disponiveis[idx % len(cores_disponiveis)]
            
        legendas_adicionadas = set()

        for i, elem in enumerate(elements):
            n1, n2 = nodes[int(elem[0])], nodes[int(elem[1])]
            
            id_elemento = props_IDs[i]
            cor_elemento = mapa_cores.get(id_elemento, 'k') # Padrão preto se não achar
            
            label_perfil = f'Perfil ID: {id_elemento}'
            
            if label_perfil not in legendas_adicionadas:
                ax.plot([n1[0], n2[0]], [n1[1], n2[1]], [n1[2], n2[2]], '-', color=cor_elemento, lw=2.5, label=label_perfil)
                legendas_adicionadas.add(label_perfil)
            else:
                ax.plot([n1[0], n2[0]],[n1[1], n2[1]], [n1[2], n2[2]], '-', color=cor_elemento, lw=2.5)
                
        # Posiciona a legenda fora do desenho para não tampar a treliça
        ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1.0))

    else:
        # Comportamento antigo (se não passar os IDs, plota tudo em preto)
        for elem in elements:
            n1, n2 = nodes[int(elem[0])], nodes[int(elem[1])]
            ax.plot([n1[0], n2[0]], [n1[1], n2[1]], [n1[2], n2[2]], 'k-', lw=1.5)

    # Evita box_aspect com zeros (e.g. estrutura planar)
    spans = (np.ptp(nodes[:, 0]), np.ptp(nodes[:, 1]), np.ptp(nodes[:, 2]))
    spans = tuple(s if s > 0 else 1.0 for s in spans)
    ax.set_box_aspect(spans)

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(title)
    
    if view is not None:
        ax.view_init(elev=float(view[0]), azim=float(view[1]))

    if savepath:
        fig.savefig(savepath, bbox_inches='tight')
    if show:
        plt.show()

def plotar_estrutura_deformada(nodes, elements, U, escala_deformacao, props_IDs, biblioteca_perfis):

    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection='3d')
    deslocamentos = U.reshape((len(nodes), 6))[:, :3]
    nodes_deformados = nodes + deslocamentos * escala_deformacao

    # Criar um mapeamento de cores dinâmico para os perfis
    cores_disponiveis =['red', 'blue', 'green', 'orange', 'purple', 'cyan', 'magenta']
    mapa_cores = {}
    for idx, perfil in enumerate(biblioteca_perfis):
        # Associa cada ID a uma cor da lista (se houver muitos perfis, as cores se repetem)
        mapa_cores[perfil['ID']] = cores_disponiveis[idx % len(cores_disponiveis)]

    # Conjunto para controlar quais legendas já foram adicionadas (evita repetição)
    legendas_adicionadas = set()

    # Linhas originais (Mantemos em cinza tracejado)
    for i, elem in enumerate(elements):
        n1, n2 = nodes[elem[0]], nodes[elem[1]]
        
        label_orig = 'Original'
        if label_orig not in legendas_adicionadas:
            ax.plot([n1[0], n2[0]], [n1[1], n2[1]], [n1[2], n2[2]],
                    '--', lw=1.5, color='gray', label=label_orig)
            legendas_adicionadas.add(label_orig)
        else:
            ax.plot([n1[0], n2[0]], [n1[1], n2[1]],[n1[2], n2[2]],
                    '--', lw=1.5, color='gray')

    # Linhas deformadas (Coloridas de acordo com o ID do perfil)
    for i, elem in enumerate(elements):
        n1, n2 = nodes_deformados[elem[0]], nodes_deformados[elem[1]]
        
        # Identifica qual o ID do elemento atual e pega sua cor correspondente
        id_elemento = props_IDs[i]
        cor_elemento = mapa_cores[id_elemento]
        
        label_def = f'Deformada (Perfil ID: {id_elemento})'
        
        # Adiciona a legenda apenas na primeira vez que esse Perfil aparecer
        if label_def not in legendas_adicionadas:
            ax.plot([n1[0], n2[0]], [n1[1], n2[1]], [n1[2], n2[2]],
                    '-', lw=2.5, color=cor_elemento, label=label_def)
            legendas_adicionadas.add(label_def)
        else:
            ax.plot([n1[0], n2[0]], [n1[1], n2[1]], [n1[2], n2[2]],
                    '-', lw=2.5, color=cor_elemento)

    # Ajuste da proporção dos eixos 3D
    ax.set_box_aspect((np.ptp(nodes[:,0]),
                       np.ptp(nodes[:,1]),
                       np.ptp(nodes[:,2])))
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    
    # Adicionando a legenda final
    ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1.0))
    ax.set_title(f'Estrutura Original vs. Deformada (Escala: {escala_deformacao}x)')

    # ======== Vista no plano XZ (modo 3D) ========
    ax.view_init(elev=0, azim=-90)
    #ax.view_init(elev=90, azim=-90)

    plt.grid(True)
    plt.savefig('plots/trelica_deformada_exemplo', bbox_inches='tight') # bbox_inches evita cortar a legenda
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
    ax.view_init(elev=25., azim=-100); plt.grid(True)
    plt.savefig('plots/trelica_tensoes_exemplo')
    plt.show()

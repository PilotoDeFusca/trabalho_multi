import matplotlib.pyplot as plt
import numpy as np

def plotar_estrutura(nodes, elements):
  fig = plt.figure(figsize=(15, 10)); ax = fig.add_subplot(111, projection='3d')
  for elem in elements:
    n1, n2 = nodes[elem[0]], nodes[elem[1]]
    ax.plot([n1[0], n2[0]], [n1[1], n2[1]], [n1[2], n2[2]],'k-', lw=1.5)
  ax.set_box_aspect((np.ptp(nodes[:,0]), np.ptp(nodes[:,1]), np.ptp(nodes[:,2])))
  ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)'); ax.set_zlabel('Z (m)')
  ax.set_title('Estrutura Original (Não Deformada)'); ax.view_init(elev=25., azim=-100)
  plt.grid(True)
  plt.savefig('plots/trelica_exemplo')
  plt.show()

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
    plt.savefig('plots/trelica_deformada_exemplo')
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
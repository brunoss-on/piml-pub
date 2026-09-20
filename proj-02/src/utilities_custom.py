import time
import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import ArrayLike

def show_one_iter_dm_custom(
    grad_fwi, dm1, grad_pred, dm, update, network_loss,
    iteration=0, *,
    cmap='jet',
    percentile: tuple[float, float] = (2, 98),
    extent=None, xlim=None, ylim=None,
    figsize=(12, 12), save_path=None,
):
    """ Adaptação em relação à função original `show_one_iter_dm` 

    Cada um dos 4 primeiros painéis agora usa sua PRÓPRIA escala de cor,
    calculada individualmente via percentil (default 2-98), em vez de um
    único `scale` global compartilhado entre os 4 arrays. Isso evita que um
    painel de amplitude muito maior "apague" o contraste dos outros 3. O
    valor numérico da escala usada é exibido no título de cada painel.
    Painéis 5 (Updated Model) e 6 (Network loss) permanecem como no original.
    """
    r = 0.81
    fig, axs = plt.subplots(3, 2, figsize=figsize)

    p_low, p_high = percentile

    panels = [
        (axs[0,0], grad_fwi, 'FWI Gradient'),
        (axs[0,1], dm1, r'$\delta m_1$'),
        (axs[1,0], grad_pred, r'$f_{\theta}(\delta m_1)$'),
        (axs[1,1], dm, r'$f_{\theta}(gradient)$'),
    ]

    for ax, data, title in panels:
        # Escala individual para este painel específico
        p_min, p_max = np.percentile(data, [p_low, p_high])
        scale = np.abs([p_min, p_max]).max()

        img = ax.imshow(data, vmin=-scale, vmax=scale, cmap=cmap,
                         interpolation='bilinear', extent=extent)
        plt.colorbar(img, ax=ax, pad=0.02, shrink=r)
        ax.set_title(f'{title}\n(scale: ±{scale:.2e}, p{p_low:.0f}-{p_high:.0f})', fontsize=12)
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Z [m]')
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

    img5 = axs[2,0].imshow(update, cmap='jet', interpolation='bilinear', extent=extent)
    plt.colorbar(img5, ax=axs[2,0], pad=0.02, shrink=r)
    axs[2,0].set_title('Updated Model', fontsize=14)
    axs[2,0].set_xlabel('X [m]')
    axs[2,0].set_ylabel('Z [m]')
    axs[2,0].set_xlim(xlim)
    axs[2,0].set_ylim(ylim)

    axs[2,1].plot(network_loss)
    axs[2,1].set_title('Network loss')
    axs[2,1].set_xlabel('Iteration')

    plt.suptitle(f'Iteration {iteration+1}', x=0.5, y=0.92, fontsize=16)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(f'{save_path}/dms_iter_{iteration+1}.png', bbox_inches='tight', dpi=300)
    plt.show()

def track_iter_time(start_time: float) -> tuple[float, float]:
    """
    Calcula o tempo gasto desde `start_time` e retorna também o novo tempo de referência.

    Parameters
    -----------
    start_time : float
        Timestamp inicial (ex.: obtido via time.time()) a partir do qual o tempo decorrido é medido.

    Returns
    -------
    elapsed : float
        Tempo decorrido em segundos desde start_time.
    new_start : float
        Novo timestamp, para ser usado como referência na próxima chamada.
    """
    new_start = time.time()
    elapsed = new_start - start_time
    return elapsed, new_start
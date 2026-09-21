import time
import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import ArrayLike
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import ScalarFormatter

def add_colorbar(fig, ax, img, label=None):
    """Colorbar com a mesma altura do painel e expoente único no topo (ex.: 1e-1)."""
    cax = make_axes_locatable(ax).append_axes('right', size='3%', pad=0.1)
    cbar = fig.colorbar(img, cax=cax)

    fmt = ScalarFormatter()           
    fmt.set_powerlimits((0, 0))        # sempre usa expoente
    cbar.formatter = fmt
    cbar.update_ticks()

    if label:
        cbar.set_label(label, fontsize=8)
    return cbar

def show_one_iter_fwi_custom(
    grad,
    model,
    iteration=0,
    *,
    cmap='jet',
    percentile=(2, 98),
    vmin=None,
    vmax=None,
    extent=None,
    figsize=(12, 5),
    save_path=None,
):
    """Plota o Gradiente (com escala simétrica p2-p98) e o Modelo de Velocidade numa grade 1x2."""
    fig, axs = plt.subplots(1, 2, figsize=figsize)
    p_low, p_high = percentile

    # Garante conversão dos tensores para matrizes NumPy
    if hasattr(grad, 'detach'):
        grad = grad.detach().cpu().numpy()
    if hasattr(model, 'detach'):
        model = model.detach().cpu().numpy()

    # Painel 1: Gradiente FWI (Escala simétrica baseada nos percentis 2% e 98%)
    p_min, p_max = np.percentile(grad, [p_low, p_high])
    scale = np.abs([p_min, p_max]).max()

    img1 = axs[0].imshow(
        grad,
        vmin=-scale,
        vmax=scale,
        cmap=cmap,
        interpolation='bilinear',
        extent=extent,
    )
    add_colorbar(fig, axs[0], img1)
    axs[0].set_title(f'Gradiente FWI (Iteração {iteration+1})', fontsize=12)
    axs[0].set_xlabel('X [m]')
    axs[0].set_ylabel('Z [m]')

    # Painel 2: Modelo Atualizado (Usa vmin/vmax se passados, ou ajusta via percentil)
    if vmin is None or vmax is None:
        u_pmin, u_pmax = np.percentile(model, [p_low, p_high])
        v_min_plot = vmin if vmin is not None else u_pmin
        v_max_plot = vmax if vmax is not None else u_pmax
    else:
        v_min_plot, v_max_plot = vmin, vmax

    img2 = axs[1].imshow(
        model,
        vmin=v_min_plot,
        vmax=v_max_plot,
        cmap=cmap,
        interpolation='bilinear',
        extent=extent,
    )
    add_colorbar(fig, axs[1], img2)
    axs[1].set_title(f'Modelo Atualizado (Iteração {iteration+1})', fontsize=12)
    axs[1].set_xlabel('X [m]')
    axs[1].set_ylabel('Z [m]')

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(
            f'{save_path}/fwi_iter_{iteration+1}.png',
            bbox_inches='tight',
            dpi=300,
        )
    plt.show()


def show_one_iter_dm_custom(
    grad_fwi,
    dm1,
    grad_pred,
    dm,
    update,
    network_loss,
    iteration=0,
    *,
    cmap='jet',
    percentile: tuple[float, float] = (2, 98),
    vmin: float | None = None,
    vmax: float | None = None,
    extent=None,
    xlim=None,
    ylim=None,
    figsize=(12, 12),
    save_path=None,
    show_scale=False,
):
    """Visualiza os resultados de uma iteração da inversão FWI (grade 3x2).

    Painéis 1 a 4: Gradientes com escala simétrica individual (p2/p98).
    Painel 5: Modelo atualizado com escala ajustada (vmin/vmax ou p2/p98).
    Painel 6: Curva de perda (loss) com notação científica no eixo Y.
    """
    fig, axs = plt.subplots(3, 2, figsize=figsize)
    p_low, p_high = percentile

    # Painéis 1 a 4: Perturbações (escala simétrica por percentil individual)
    panels = [
        (axs[0, 0], grad_fwi, 'FWI Gradient'),
        (axs[0, 1], dm1, r'$\delta m_1$'),
        (axs[1, 0], grad_pred, r'$f_{\theta}(\delta m_1)$'),
        (axs[1, 1], dm, r'$f_{\theta}(gradient)$'),
    ]

    for ax, data, title in panels:
        p_min, p_max = np.percentile(data, [p_low, p_high])
        scale = np.abs([p_min, p_max]).max()

        img = ax.imshow(
            data,
            vmin=-scale,
            vmax=scale,
            cmap=cmap,
            interpolation='bilinear',
            extent=extent,
        )
        add_colorbar(fig, ax, img)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Z [m]')
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

    # Painel 5: Updated Model (Valores de velocidade por percentil 2-98 ou limites manuais)
    if vmin is None or vmax is None:
        u_pmin, u_pmax = np.percentile(update, [p_low, p_high])
        v_min_plot = vmin if vmin is not None else u_pmin
        v_max_plot = vmax if vmax is not None else u_pmax
    else:
        v_min_plot = vmin
        v_max_plot = vmax

    img5 = axs[2, 0].imshow(
        update,
        vmin=v_min_plot,
        vmax=v_max_plot,
        cmap='jet',
        interpolation='bilinear',
        extent=extent,
    )
    add_colorbar(fig, axs[2, 0], img5)
    axs[2, 0].set_title('Updated Model', fontsize=12)
    axs[2, 0].set_xlabel('X [m]')
    axs[2, 0].set_ylabel('Z [m]')
    axs[2, 0].set_xlim(xlim)
    axs[2, 0].set_ylim(ylim)

    # Painel 6: Network Loss (Formatado com Notação Científica no Eixo Y)
    axs[2, 1].plot(network_loss)
    axs[2, 1].set_title('Network loss', fontsize=12)
    axs[2, 1].set_xlabel('Iteration')
    axs[2, 1].set_ylabel('Loss')

    # Força a notação científica igual à da barra de cores 
    fmt_y = ScalarFormatter()
    fmt_y.set_powerlimits((0, 0))
    axs[2, 1].yaxis.set_major_formatter(fmt_y)

    fig.suptitle(f'Iteration {iteration+1}', fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path is not None:
        fig.savefig(
            f'{save_path}/dms_iter_{iteration+1}.png',
            bbox_inches='tight',
            dpi=300,
        )
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
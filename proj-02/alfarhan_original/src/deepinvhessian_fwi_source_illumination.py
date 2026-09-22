import torch
from tqdm import tqdm
import deepwave

def custom_source_illumination(model: torch.Tensor, source: torch.Tensor, dx: float, dt: float, 
                              x_s: torch.Tensor, device: str) -> torch.Tensor:
    """
    Versão corrigida da função source_illumination do autor.
    Garante que o grid de receptores virtuais (x_snap) seja composto por índices inteiros (long).
    """
    nz, nx = model.shape
    num_shots, _, nt = source.shape
    num_batches = num_shots

    x = torch.arange(nx, dtype=torch.float32) * dx
    z = torch.arange(nz, dtype=torch.float32) * dx
    x, z = torch.meshgrid(x, z, indexing='ij')
    x, z = x.flatten(), z.flatten()
    
    # 'Receivers' at every point in the model
    # CORREÇÃO AQUI: Forçamos o arredondamento e conversão para inteiros (.long())
    x_snap = torch.stack([z, x], dim=-1) / dx
    x_snap = torch.round(x_snap).long() 
    
    x_snap = x_snap.unsqueeze(0).expand(num_shots, -1, -1)

    # Move the snapshot grid to the device
    x_snap = x_snap.to(device)

    result = torch.zeros((1, nz*nx, nt), device=device)

    # Simulate the wavefield for each batch and sum up the energy
    for it in tqdm(range(num_shots)):
        source_wavefield = deepwave.scalar(
            model.to(device), 
            dx, 
            dt,
            source_amplitudes=source[it::num_batches].to(device),
            # Segurança extra: garante que x_s também seja inteiro aqui dentro
            source_locations=torch.round(x_s[it::num_batches]).long().to(device),
            receiver_locations=x_snap[it::num_batches].to(device),
            pml_width=[20, 20, 20, 20],
            accuracy=8,
        )[-1].squeeze()

        # Sum the squared wavefield to result
        result += source_wavefield ** 2
    
    # Sum over time to get the final source illumination
    src_illum = result.sum(dim=-1)
    src_illum = src_illum.squeeze().reshape(nx, nz).T

    return src_illum.to(device)
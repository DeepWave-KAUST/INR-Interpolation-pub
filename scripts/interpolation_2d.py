from configure import *
from util import *

import math, tqdm
import torch
import numpy as np
import matplotlib.pyplot as plt
from network import *
import time

import warnings
warnings.filterwarnings('ignore')
torch.manual_seed(888)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

std = 250
normalized = False
n_epochs = 301
save_interval = 50
lr = 1e-4

use_hash = True
use_fourier = True
num_branches = 0

mode = 'hybrid' # hybrid, fourier_only, hash_only
network = 'HybridEncodingMLP'
savepath = ensure_dir(
    output_path(
        f'results_{network}',
        f'epoch={n_epochs}_mode={mode}_test_hash_differenct_config2',
    )
)
# Update Configuration for HashGrid
per_level_scale = math.exp((math.log(hash_config['finest_resolution']) - math.log(hash_config['base_resolution'])) / (hash_config['n_levels'] - 1))
hash_config['per_level_scale'] = per_level_scale
hash_config.update({"otype": "HashGrid",
                    "per_level_scale": per_level_scale,
                    "interpolation": "linear"})


if network == 'CAFE_Net':
    nnmlp = CAFE_Net().to(device)
elif network == 'SineMLP':
    nnmlp = SineMLP(layers=layers,
                 features=features,
                 use_bias=True).to(device)
else:
    nnmlp = HybridEncodingMLP(layers=layers,
                    features=features,
                    omega=omega,
                    use_bias=False,
                    use_hash=use_hash,
                    hash_config=hash_config,
                    use_fourier=use_fourier).to(device)


# Count number of parameters
total_params = sum(p.numel() for p in nnmlp.parameters() if p.requires_grad)
print(f'The network has {total_params} trainable parameters')

def train():
    """Train the INR on sparse 2D DAS data and periodically save reconstructions.

    Loads the sparse and full DAS arrays, fits the network on the non-zero columns
    with an MSE loss while annealing the hash/Fourier blend weight, and writes a
    checkpoint, loss curve and reconstruction figures to ``savepath``.
    """
    ### load data
    original_data = torch.tensor(np.load(data_path('Sparse_Data_DAS.npy')), dtype=torch.float32).to(device)
    full_data = np.load(data_path('Full_Data_DAS.npy'))
    nonzero_cols = np.load(data_path('non-zero_index.npy'))

    mesh = grid_init(original_data.shape, 0, 1)
    shape = original_data.shape

    opt = torch.optim.Adam(nnmlp.parameters(), lr=lr)
    d_min, d_max, d_std = torch.min(original_data), torch.max(original_data), std

    criterion = torch.nn.MSELoss()
    losses = []
    cmap = 'seismic'

    total_coords = mesh.shape[0]
    hash_weights = []
    f_weights = []

    # Training loop
    start_time = time.time()
    for epoch in tqdm.trange(n_epochs):
        opt.zero_grad()

        if network == 'CAFE_Net' or network == 'SineMLP':
            reconstruct_data = nnmlp(torch.from_numpy(mesh).to(device))
            memory = torch.cuda.max_memory_allocated() / 1024**2
            print("Memory cost: %.2f MB" % memory)
        else:
            # Eqs. (20)-(21): tau(e) = min(e / E_s, 1) with E_s = n_epochs / 2,
            # w_H(e) = 0.5 * tau(e) and w_F(e) = 1 - w_H(e).
            w = min(0.5, epoch / (n_epochs * 0.5)* 0.5)
            w_hash = w
            hash_weights.append(w_hash)
            f_weights.append(1 - w_hash)
            reconstruct_data, hash_enc, f_enc = nnmlp(torch.from_numpy(mesh).to(device), w_hash=w_hash, mode=mode)

        reconstruct_data = reconstruct_data.view(shape) * d_std

        loss = criterion(reconstruct_data[:, nonzero_cols], original_data[:, nonzero_cols])

        print(f'Epoch: {epoch}, loss: {loss.item()}')

        losses.append(loss.item())
        loss.backward()

        if epoch == (n_epochs - 1):
            checkpoint = {
                "model": nnmlp.state_dict(),
                'optimizer': opt.state_dict(),
                "epoch": epoch,
            }
            torch.save(checkpoint, savepath / 'checkpoint.pth')

            plt.plot(losses)
            plt.savefig(savepath / 'loss.png')
            plt.close()
            np.save(savepath / 'loss.npy', losses)
            print('save checkpoint!!')

        if epoch % save_interval == 0 or epoch == n_epochs - 1:
            reconstruct_data_cpu = reconstruct_data.detach().cpu().numpy()
            original_data_cpu = original_data.detach().cpu().numpy()

            vmin = np.percentile(full_data, 1)
            vmax = np.percentile(full_data, 99)

            plt.figure(figsize=(10, 8))
            plt.subplot(221)
            plt.imshow(original_data_cpu, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)
            plt.title('Sparse Data')
            plt.colorbar()
            plt.subplot(222)
            plt.imshow(full_data, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)
            plt.title('Full Data')
            plt.colorbar()
            plt.subplot(223)
            plt.imshow(reconstruct_data_cpu, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)
            plt.title('Reconstructed Data')
            plt.colorbar()
            plt.subplot(224)
            plt.imshow(full_data - reconstruct_data_cpu, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)
            plt.title('Residual')
            plt.colorbar()
            plt.savefig(savepath / f'reconstruct_{epoch:04d}.png', bbox_inches='tight')
            np.save(savepath / f'reconstruct_{epoch:04d}.npy', reconstruct_data_cpu)
            plt.close()


            if epoch == n_epochs - 1:
                plt.plot(hash_weights, color='r', label='Hash Weight')
                plt.plot(f_weights, color='b', label='Fourier Weight')
                plt.legend()
                plt.savefig(savepath / 'weights.png', bbox_inches='tight', dpi=300)
                plt.close()

        opt.step()

    end_time = time.time()
    print('cost time: ', end_time - start_time)


if __name__ == '__main__':
    train()

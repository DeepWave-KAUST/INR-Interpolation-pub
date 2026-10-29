from configure import *
from util import *

import math, tqdm
import torch
import numpy as np
import matplotlib.pyplot as plt
from network import *
from pyseistr import plot3d

import warnings
warnings.filterwarnings('ignore')
# NumPy seeds the sampling mask, torch seeds the network initialisation; both are needed
# for a reproducible run, as in interpolation_2d.py.
np.random.seed(888)
torch.manual_seed(888)

use_cuda = torch.cuda.is_available()
ngpus = torch.cuda.device_count() if use_cuda else 0
device = torch.device('cuda' if use_cuda else 'cpu')
print(f"Using device: {device}, num gpus: {ngpus}")



std = 5
normalized = False
n_epochs = 401
save_interval = 50
lr = 1e-4
reg = None  # TV, L2, Smooth, Kernel
lambda_reg = 1e-4
net = 'hybird'   # Sine, Finer, FR_INR, hybird
miss_rate = 0.8

omega = 30 # Frequency of each sine activation
layers = 2
features = 128 # Number of neurons per layer
n_levels = 8
# Section 4.1: for the volumes the finest resolution is N_max = 64, chosen so that the
# finest level still fits within the hash table, since in three dimensions the number of
# grid vertices grows as (N_l + 1)^3. This gives 1254465 trainable parameters (Table 1).
finest_resolution = 64
hash_config = dict(
    n_levels=n_levels,
    n_features_per_level=2,
    log2_hashmap_size=19,
    base_resolution=16,
    finest_resolution=finest_resolution
)

use_hash = True
savepath = ensure_dir(
    output_path(
        'results_field3D',
        f'net={net}_miss_rate={miss_rate}_levels={n_levels}_{finest_resolution}_{lr}_std={std}_layers={layers}_features={features}_epoch={n_epochs}_different_config',
    )
)

# Update Configuration for HashGrid
per_level_scale = math.exp((math.log(hash_config['finest_resolution']) - math.log(hash_config['base_resolution'])) / (hash_config['n_levels'] - 1))
hash_config['per_level_scale'] = per_level_scale
hash_config.update({"otype": "HashGrid",
                    "per_level_scale": per_level_scale,
                    "interpolation": "linear"})

if net == 'SineMLP':
    use_fourier = False
    nnmlp = SineMLP(layers=layers,
                    in_dim=3,
                    features=features,
                    omega=omega,
                    use_bias=False,
                    use_hash=use_hash,
                    hash_config=hash_config).to(device)
else:
    use_fourier = True
    nnmlp = HybridEncodingMLP(layers=layers,
                    features=features,
                    in_dim=3,
                    omega=omega,
                    use_bias=False,
                    use_hash=use_hash,
                    hash_config=hash_config,
                    use_fourier=use_fourier).to(device)

# The experiments of the paper use a single GPU and the training loop calls ``nnmlp``
# directly, so the model is not wrapped for data parallelism.

# Count number of parameters
total_params = sum(p.numel() for p in nnmlp.parameters() if p.requires_grad)
print(f'The network has {total_params} trainable parameters')

def train():
    """Train the INR to reconstruct a randomly sub-sampled 3D seismic volume.

    Loads the full volume, builds a random trace mask at ``miss_rate``, fits the network
    on the observed samples with an MSE loss while annealing the hash/Fourier blend
    weight, and saves checkpoints, inline/crossline slices and 3D renderings to ``savepath``.
    """
    print('training missing data')

    full_data = np.load(data_path('image3dz_400x100x200.npy'))

    full_data = torch.tensor(full_data, dtype=torch.float32).to(device)

    shape = full_data.shape

    mesh = grid_init(shape, 0, 1)

    D1, D2, D3 = full_data.shape
    mask = (np.random.rand(D2, D3) >= miss_rate).astype(np.float32)
    mask = torch.tensor(mask[None, :, :]).to(device)

    original_data = full_data * mask

    opt = torch.optim.Adam(nnmlp.parameters(), lr=lr)

    criterion = torch.nn.MSELoss()
    losses = []
    cmap = 'seismic'

    total_coords = mesh.shape[0]
    hash_weights = []
    f_weights = []

    for epoch in tqdm.trange(n_epochs):
        opt.zero_grad()

        if net == 'hybird':
            w = min(0.5, epoch / (n_epochs * 0.5)* 0.5)
            w_hash = w
            hash_weights.append(w_hash)
            f_weights.append(1 - w_hash)

            print(f"w_hash={w_hash:.6f}, w_ff={(1-w_hash):.6f}")
            reconstruct_data, hash_enc, f_enc = nnmlp(torch.from_numpy(mesh).to(device), w_hash=w_hash)
        else:
            reconstruct_data, hash_enc = nnmlp(torch.from_numpy(mesh).to(device))

        reconstruct_data = reconstruct_data.view(shape) * std

        loss = criterion(reconstruct_data * mask, original_data)

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
        scale = [1, 99]
        if epoch % 50 == 0 or epoch == n_epochs - 1:
            reconstruct_data_cpu = reconstruct_data.detach().cpu().numpy()
            original_data_cpu = original_data.detach().cpu().numpy()
            full_data_cpu = full_data.detach().cpu().numpy()

            np.save(savepath / f'reconstruct_data_{epoch}.npy', reconstruct_data_cpu)

            i = np.random.permutation(shape[2])[:1]
            residual = full_data_cpu[:, :, i][:, :, 0] - reconstruct_data_cpu[:, :, i][:, :, 0]

            vmin2, vmax2 = np.percentile(full_data_cpu[:, :, i], scale)

            plt.subplot(221)
            plt.imshow(original_data_cpu[:, :, i][:, :, 0], cmap="seismic", aspect="auto", vmin=vmin2, vmax=vmax2)
            plt.colorbar()
            plt.subplot(222)
            plt.imshow(reconstruct_data_cpu[:, :, i][:, :, 0], cmap="seismic", aspect="auto", vmin=vmin2, vmax=vmax2)
            plt.colorbar()
            plt.subplot(223)
            plt.imshow(full_data_cpu[:, :, i][:, :, 0], cmap="seismic", aspect="auto", vmin=vmin2, vmax=vmax2)
            plt.colorbar()
            plt.subplot(224)
            plt.imshow(residual, cmap="seismic", aspect="auto", vmin=vmin2, vmax=vmax2)
            plt.colorbar()
            plt.savefig(savepath / f'Inline epoch {epoch}.png')
            plt.close()

            zero_rows_D1 = np.where(np.all(original_data_cpu[:, :, i][:, :, 0] == 0, axis=0))[0][0]

            plt.plot(original_data_cpu[:, :, i][:, :, 0][:, zero_rows_D1], color='g', label='missing')
            plt.plot(reconstruct_data_cpu[:, :, i][:, :, 0][:, zero_rows_D1], color='b', label='pred')
            plt.plot(full_data_cpu[:, :, i][:, :, 0][:, zero_rows_D1], color='r', label='true')
            plt.legend()
            plt.savefig(savepath / f'Inline epoch {epoch} trace {zero_rows_D1}.png')
            plt.close()

            j = np.random.permutation(shape[1])[:1]

            residual = full_data_cpu[:, j, :][:, 0, :] - reconstruct_data_cpu[:, j, :][:, 0, :]

            vmin2, vmax2 = np.percentile(full_data_cpu[:, j, :], scale)

            plt.subplot(221)
            plt.imshow(original_data_cpu[:, j, :][:, 0, :], cmap="seismic", aspect="auto", vmin=vmin2, vmax=vmax2)
            plt.colorbar()
            plt.subplot(222)
            plt.imshow(reconstruct_data_cpu[:, j, :][:, 0, :], cmap="seismic", aspect="auto", vmin=vmin2, vmax=vmax2)
            plt.colorbar()
            plt.subplot(223)
            plt.imshow(full_data_cpu[:, j, :][:, 0, :], cmap="seismic", aspect="auto", vmin=vmin2, vmax=vmax2)
            plt.colorbar()
            plt.subplot(224)
            plt.imshow(residual, cmap="seismic", aspect="auto", vmin=vmin2, vmax=vmax2)
            plt.colorbar()
            plt.savefig(savepath / f'Crossline epoch {epoch}.png')
            plt.close()

            zero_rows_D2 = np.where(np.all(original_data_cpu[:, j, :][:, 0, :] == 0, axis=0))[0][0]

            plt.plot(original_data_cpu[:, j, :][:, 0, :][:, zero_rows_D2], color='g', label='missing')
            plt.plot(reconstruct_data_cpu[:, j, :][:, 0, :][:, zero_rows_D2], color='b', label='pred')
            plt.plot(full_data_cpu[:, j, :][:, 0, :][:, zero_rows_D2], color='r', label='true')
            plt.legend()
            plt.savefig(savepath / f'Crossline epoch {epoch} trace {zero_rows_D2}.png')
            plt.close()


        opt.step()

    reconstruct_data_cpu = reconstruct_data.detach().cpu().numpy()
    original_data_cpu = original_data.detach().cpu().numpy()
    full_data_cpu = full_data.detach().cpu().numpy()

    np.save(savepath / 'missing_data.npy', original_data_cpu)
    np.save(savepath / 'full_data.npy', full_data_cpu)

    nz=D1
    nx=D2
    ny=D3

    plot3d(full_data_cpu,z=np.arange(nz),x=np.arange(nx),y=np.arange(ny),showf=False,close=False)
    plt.gca().set_xlabel("X (m)",fontsize='large', fontweight='normal')
    plt.gca().set_ylabel("Y (m)",fontsize='large', fontweight='normal')
    plt.gca().set_zlabel("Time (s)",fontsize='large', fontweight='normal')
    plt.title('3D full data')
    plt.savefig(fname=savepath / 'full_data.png', format='png', dpi=300)
    print('save full_data')

    plot3d(original_data_cpu,z=np.arange(nz),x=np.arange(nx),y=np.arange(ny),showf=False,close=False)
    plt.gca().set_xlabel("X (m)",fontsize='large', fontweight='normal')
    plt.gca().set_ylabel("Y (m)",fontsize='large', fontweight='normal')
    plt.gca().set_zlabel("Time (s)",fontsize='large', fontweight='normal')
    plt.title('3D missing data')
    plt.savefig(fname=savepath / 'missing_data.png', format='png', dpi=300)
    print('save missing_data')

    plot3d(reconstruct_data_cpu,z=np.arange(nz),x=np.arange(nx),y=np.arange(ny),showf=False,close=False)
    plt.gca().set_xlabel("X (m)",fontsize='large', fontweight='normal')
    plt.gca().set_ylabel("Y (m)",fontsize='large', fontweight='normal')
    plt.gca().set_zlabel("Time (s)",fontsize='large', fontweight='normal')
    plt.title('3D reconstruct data')
    plt.savefig(fname=savepath / 'reconstruct_data.png', format='png', dpi=300)
    print('save reconstruct_data')

    plot3d(full_data_cpu - reconstruct_data_cpu,z=np.arange(nz),x=np.arange(nx),y=np.arange(ny),showf=False,close=False)
    plt.gca().set_xlabel("X (m)",fontsize='large', fontweight='normal')
    plt.gca().set_ylabel("Y (m)",fontsize='large', fontweight='normal')
    plt.gca().set_zlabel("Time (s)",fontsize='large', fontweight='normal')
    plt.title('3D residual data')
    plt.savefig(fname=savepath / 'residual_data.png', format='png', dpi=300)
    print('save residual_data')




if __name__ == '__main__':
    train()

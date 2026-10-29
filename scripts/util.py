from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"


def project_path(*parts):
    """Build a path relative to the project root.

    Args:
        *parts: Path components to append to the project root.

    Returns:
        Path: The joined path under the project root.
    """
    return PROJECT_ROOT.joinpath(*parts)


def data_path(*parts):
    """Build a path relative to the project ``data`` directory.

    Args:
        *parts: Path components to append to the data directory.

    Returns:
        Path: The joined path under the data directory.
    """
    return DATA_DIR.joinpath(*parts)


def output_path(*parts):
    """Build a path relative to the project ``outputs`` directory.

    Args:
        *parts: Path components to append to the outputs directory.

    Returns:
        Path: The joined path under the outputs directory.
    """
    return OUTPUTS_DIR.joinpath(*parts)


def ensure_dir(path):
    """Create a directory (and any missing parents) if it does not exist.

    Args:
        path: Directory path to create.

    Returns:
        Path: The created (or already existing) directory.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def ricker(t, f=10.):
    """Ricker wavelet.

    Args:
        t (Array): Discrete time.
        f (float, optional): Dominant frequency. Defaults to 10..

    Returns:
        Array: Ricker wavelet series.
    """
    r = (1 - 2 * (np.pi * f * t) ** 2) * np.exp(-(np.pi * f * t) ** 2)
    return r.astype(np.float32)


def grid_init(shape, vmin=0., vmax=1.):
    """Initialize the mesh of the domain

    Args:
        shape (tuple): The shape of the domain.
        vmin (float): The minimum value of the domain.
        vmax (float): The maximum value of the domain.

    Returns:
        Array : The mesh of the domain with shape (*shape, 2).
    """
    coord_axis = [np.linspace(vmin, vmax, d, dtype=np.float32) for d in shape]
    grid = np.stack(np.meshgrid(*coord_axis, indexing='ij'), -1)

    ndim = len(shape)

    return grid.reshape(-1, ndim).astype(np.float32)

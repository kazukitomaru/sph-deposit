import numpy as np
from numba import njit

from .grid_2d import _index
from .kernels import KERNELS, kernel

N_SUB_PER_H = 4  # sub-samples per smoothing length, along each axis
MAX_N_SUB = 16  # sub-samples per cell, along each axis


def deposit_grid(
    x, y, z, h, quantities, center, width, n_cells, kernel, cell_array=None
):
    if cell_array is None:
        cell_array = np.zeros((*n_cells, quantities.shape[1]))
    lower = center - width / 2
    cell_size = width / n_cells
    _deposit_grid(
        cell_array, x, y, z, h, quantities, *lower, *cell_size, KERNELS[kernel]
    )
    return cell_array


@njit(cache=True)
def _deposit_grid(
    cell_array, x, y, z, h, quantities, x_min, y_min, z_min, dx, dy, dz, kernel_id
):
    n_x, n_y, n_z, n_q = cell_array.shape
    weights = np.empty((n_x, n_y, n_z))
    for k in range(x.size):
        n_sub = max(1, int(np.ceil(N_SUB_PER_H * max(dx, dy, dz) / h[k])))
        # cells under the kernel, including those outside the grid
        ix0, ix1 = _index(x[k] - h[k], x_min, dx), _index(x[k] + h[k], x_min, dx)
        iy0, iy1 = _index(y[k] - h[k], y_min, dy), _index(y[k] + h[k], y_min, dy)
        iz0, iz1 = _index(z[k] - h[k], z_min, dz), _index(z[k] + h[k], z_min, dz)

        # the sum includes cells outside the grid, so what falls outside is lost
        w_sum = 0.0
        # a kernel much smaller than a cell is not sampled
        if n_sub <= MAX_N_SUB:
            for ix in range(ix0, ix1 + 1):
                for iy in range(iy0, iy1 + 1):
                    for iz in range(iz0, iz1 + 1):
                        x0 = x_min + ix * dx
                        y0 = y_min + iy * dy
                        z0 = z_min + iz * dz
                        w = _cell_weight(
                            x[k],
                            y[k],
                            z[k],
                            h[k],
                            x0,
                            y0,
                            z0,
                            dx,
                            dy,
                            dz,
                            n_sub,
                            kernel_id,
                        )
                        w_sum += w
                        if 0 <= ix < n_x and 0 <= iy < n_y and 0 <= iz < n_z:
                            weights[ix, iy, iz] = w

        # a kernel much smaller than a cell gives everything to the cell holding the particle
        if w_sum == 0.0:
            ix = _index(x[k], x_min, dx)
            iy = _index(y[k], y_min, dy)
            iz = _index(z[k], z_min, dz)
            if 0 <= ix < n_x and 0 <= iy < n_y and 0 <= iz < n_z:
                for iq in range(n_q):
                    cell_array[ix, iy, iz, iq] += quantities[k, iq]
            continue

        for ix in range(max(ix0, 0), min(ix1, n_x - 1) + 1):
            for iy in range(max(iy0, 0), min(iy1, n_y - 1) + 1):
                for iz in range(max(iz0, 0), min(iz1, n_z - 1) + 1):
                    w_normalized = weights[ix, iy, iz] / w_sum
                    for iq in range(n_q):
                        cell_array[ix, iy, iz, iq] += w_normalized * quantities[k, iq]


@njit(cache=True)
def _cell_weight(x, y, z, h, x0, y0, z0, dx, dy, dz, n_sub, kernel_id):
    w = 0.0
    for jz in range(n_sub):
        ddz = z0 + (jz + 0.5) * dz / n_sub - z
        for jy in range(n_sub):
            ddy = y0 + (jy + 0.5) * dy / n_sub - y
            for jx in range(n_sub):
                ddx = x0 + (jx + 0.5) * dx / n_sub - x
                w += kernel(np.sqrt(ddx * ddx + ddy * ddy + ddz * ddz) / h, kernel_id)
    return w

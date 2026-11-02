import numpy as np
from numba import njit

from .kernels import KERNELS, kernel, projected_kernel, sliced_kernel

N_SUB_PER_H = 2  # sub-samples per smoothing length, along each axis
MAX_N_SUB = 8  # sub-samples per cell, along each axis


def deposit_projection(
    x, y, h, quantities, center, width, n_cells, kernel, cell_array=None
):
    if cell_array is None:
        cell_array = np.zeros((*n_cells, quantities.shape[1]))
    lower = center - width / 2
    cell_size = width / n_cells
    _deposit_projection(
        cell_array, x, y, h, quantities, *lower, *cell_size, KERNELS[kernel]
    )
    return cell_array


def deposit_slice(
    x, y, z, h, volume, quantities, center, width, n_cells, kernel, cell_array=None
):
    if cell_array is None:
        cell_array = np.zeros((*n_cells, quantities.shape[1]))
    # the slice is the plane z = center[2]
    lower = center[:2] - width / 2
    cell_size = width / n_cells
    _deposit_slice(
        cell_array,
        x,
        y,
        z,
        h,
        volume,
        quantities,
        *lower,
        *cell_size,
        center[2],
        KERNELS[kernel],
    )
    return cell_array


@njit(cache=True)
def _deposit_projection(
    cell_array, x, y, h, quantities, x_min, y_min, dx, dy, kernel_id
):
    n_x, n_y, n_q = cell_array.shape
    weights = np.empty((n_x, n_y))
    for k in range(x.size):
        n_sub = max(1, int(np.ceil(N_SUB_PER_H * max(dx, dy) / h[k])))
        # cells under the kernel, including those outside the grid
        ix0, ix1 = _index(x[k] - h[k], x_min, dx), _index(x[k] + h[k], x_min, dx)
        iy0, iy1 = _index(y[k] - h[k], y_min, dy), _index(y[k] + h[k], y_min, dy)

        # the sum includes cells outside the grid, so what falls outside is lost
        w_sum = 0.0
        # a kernel much smaller than a cell is not sampled
        if n_sub <= MAX_N_SUB:
            for ix in range(ix0, ix1 + 1):
                for iy in range(iy0, iy1 + 1):
                    x0 = x_min + ix * dx
                    y0 = y_min + iy * dy
                    w = _cell_projection_weight(
                        x[k], y[k], h[k], x0, y0, dx, dy, n_sub, kernel_id
                    )
                    w_sum += w
                    if 0 <= ix < n_x and 0 <= iy < n_y:
                        weights[ix, iy] = w

        # a kernel much smaller than a cell gives everything to the cell holding the particle
        if w_sum == 0.0:
            ix, iy = _index(x[k], x_min, dx), _index(y[k], y_min, dy)
            if 0 <= ix < n_x and 0 <= iy < n_y:
                for iq in range(n_q):
                    cell_array[ix, iy, iq] += quantities[k, iq]
            continue

        for ix in range(max(ix0, 0), min(ix1, n_x - 1) + 1):
            for iy in range(max(iy0, 0), min(iy1, n_y - 1) + 1):
                w_normalized = weights[ix, iy] / w_sum
                for iq in range(n_q):
                    cell_array[ix, iy, iq] += w_normalized * quantities[k, iq]


@njit(cache=True)
def _deposit_slice(
    cell_array, x, y, z, h, volume, quantities, x_min, y_min, dx, dy, z_slice, kernel_id
):
    n_x, n_y, n_q = cell_array.shape
    for k in range(x.size):
        d = z[k] - z_slice
        # a kernel not reaching the slice adds nothing
        if abs(d) >= h[k]:
            continue
        # radius of the kernel on the slice
        r = np.sqrt(h[k] * h[k] - d * d)
        n_sub = max(1, int(np.ceil(N_SUB_PER_H * max(dx, dy) / r)))

        # a kernel much smaller than a cell gives its cell the kernel integrated
        # over the slice, divided by the cell area
        if n_sub > MAX_N_SUB:
            ix, iy = _index(x[k], x_min, dx), _index(y[k], y_min, dy)
            if 0 <= ix < n_x and 0 <= iy < n_y:
                w = (
                    volume[k]
                    * sliced_kernel(abs(d) / h[k], kernel_id)
                    / (h[k] * dx * dy)
                )
                for iq in range(n_q):
                    cell_array[ix, iy, iq] += w * quantities[k, iq]
            continue

        ix0 = max(0, _index(x[k] - r, x_min, dx))
        ix1 = min(n_x - 1, _index(x[k] + r, x_min, dx))
        iy0 = max(0, _index(y[k] - r, y_min, dy))
        iy1 = min(n_y - 1, _index(y[k] + r, y_min, dy))
        for ix in range(ix0, ix1 + 1):
            for iy in range(iy0, iy1 + 1):
                x0 = x_min + ix * dx
                y0 = y_min + iy * dy
                w = volume[k] * _cell_slice_weight(
                    x[k], y[k], d, h[k], x0, y0, dx, dy, n_sub, kernel_id
                )
                if w > 0.0:
                    for iq in range(n_q):
                        cell_array[ix, iy, iq] += w * quantities[k, iq]


@njit(cache=True)
def _index(x, x_min, dx):
    return int(np.floor((x - x_min) / dx))


@njit(cache=True)
def _cell_projection_weight(x, y, h, x0, y0, dx, dy, n_sub, kernel_id):
    w = 0.0
    for jy in range(n_sub):
        ddy = y0 + (jy + 0.5) * dy / n_sub - y
        for jx in range(n_sub):
            ddx = x0 + (jx + 0.5) * dx / n_sub - x
            w += projected_kernel(np.sqrt(ddx * ddx + ddy * ddy) / h, kernel_id)
    return w


@njit(cache=True)
def _cell_slice_weight(x, y, d, h, x0, y0, dx, dy, n_sub, kernel_id):
    w = 0.0
    for jy in range(n_sub):
        ddy = y0 + (jy + 0.5) * dy / n_sub - y
        for jx in range(n_sub):
            ddx = x0 + (jx + 0.5) * dx / n_sub - x
            w += kernel(np.sqrt(ddx * ddx + ddy * ddy + d * d) / h, kernel_id)
    return w / (n_sub * n_sub * h**3)


def regrid_cells(x, y, size, quantities, center, width, n_cells):
    cell_array = np.zeros((*n_cells, quantities.shape[1]))
    lower = center - width / 2
    cell_size = width / n_cells
    _regrid_cells(cell_array, x, y, size, quantities, *lower, *cell_size)
    return cell_array


@njit(cache=True)
def _regrid_cells(cell_array, x, y, size, quantities, x_min, y_min, dx, dy):
    n_x, n_y, n_q = cell_array.shape
    for k in range(x.size):
        half_size = size[k] / 2
        x_lower, x_upper = x[k] - half_size, x[k] + half_size
        y_lower, y_upper = y[k] - half_size, y[k] + half_size
        ix0 = max(0, _index(x_lower, x_min, dx))
        ix1 = min(n_x - 1, _index(x_upper, x_min, dx))
        iy0 = max(0, _index(y_lower, y_min, dy))
        iy1 = min(n_y - 1, _index(y_upper, y_min, dy))
        for ix in range(ix0, ix1 + 1):
            overlap_x = min(x_upper, x_min + (ix + 1) * dx) - max(
                x_lower, x_min + ix * dx
            )
            if overlap_x <= 0.0:
                continue
            for iy in range(iy0, iy1 + 1):
                overlap_y = min(y_upper, y_min + (iy + 1) * dy) - max(
                    y_lower, y_min + iy * dy
                )
                if overlap_y <= 0.0:
                    continue
                w = overlap_x * overlap_y / (size[k] * size[k])
                for iq in range(n_q):
                    cell_array[ix, iy, iq] += w * quantities[k, iq]

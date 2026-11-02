import numpy as np
from numba import njit
from scipy.integrate import quad

CUBIC_SPLINE = 0
WENDLAND_C4 = 1
KERNELS = {"cubic_spline": CUBIC_SPLINE, "wendland_c4": WENDLAND_C4}


@njit(cache=True)
def cubic_spline(q):
    if q < 0.5:
        return 8.0 / np.pi * (1.0 - 6.0 * q * q + 6.0 * q * q * q)
    return 16.0 / np.pi * (1.0 - q) ** 3


@njit(cache=True)
def wendland_c4(q):
    return (
        495.0 / (32.0 * np.pi) * (1.0 - q) ** 6 * (1.0 + 6.0 * q + 35.0 / 3.0 * q * q)
    )


@njit(cache=True)
def kernel(q, kernel_id):
    """Kernel at q = r / h, where h is the compact support radius."""
    if q >= 1.0:
        return 0.0
    if kernel_id == WENDLAND_C4:
        return wendland_c4(q)
    return cubic_spline(q)


def _integrate_projected_kernel(r, kernel_id):
    z_max = np.sqrt(1.0 - r * r)
    return 2.0 * quad(lambda z: kernel(np.hypot(r, z), kernel_id), 0.0, z_max)[0]


def _integrate_sliced_kernel(d, kernel_id):
    # over the plane in rings of radius r, with q^2 = r^2 + d^2 so that r dr = q dq
    return 2.0 * np.pi * quad(lambda q: kernel(q, kernel_id) * q, d, 1.0)[0]


N_TABLE = 1024
PROJECTED_KERNEL_TABLE = np.array(
    [
        [
            _integrate_projected_kernel(r, kernel_id)
            for r in np.linspace(0.0, 1.0, N_TABLE)
        ]
        for kernel_id in range(len(KERNELS))
    ]
)
SLICED_KERNEL_TABLE = np.array(
    [
        [_integrate_sliced_kernel(d, kernel_id) for d in np.linspace(0.0, 1.0, N_TABLE)]
        for kernel_id in range(len(KERNELS))
    ]
)


@njit(cache=True)
def _interpolate(table, x):
    if x >= 1.0:
        return 0.0
    pos = x * (N_TABLE - 1)
    i = int(pos)
    frac = pos - i
    return table[i] * (1.0 - frac) + table[i + 1] * frac


@njit(cache=True)
def projected_kernel(r, kernel_id):
    return _interpolate(PROJECTED_KERNEL_TABLE[kernel_id], r)


@njit(cache=True)
def sliced_kernel(d, kernel_id):
    return _interpolate(SLICED_KERNEL_TABLE[kernel_id], d)

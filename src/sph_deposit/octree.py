from dataclasses import dataclass

import numpy as np
from numba import njit

from .grid_2d import _index
from .grid_3d import MAX_N_SUB, N_SUB_PER_H
from .kernels import KERNELS, kernel


@dataclass
class Octree:
    root_center: np.ndarray  # (3,)
    root_size: float
    node_level: np.ndarray  # (n_nodes,), 0 for the root
    node_index: np.ndarray  # (n_nodes, 3), integer position within the level
    first_child: np.ndarray  # (n_nodes,), the 8 children follow it; -1 for leaves

    @property
    def n_levels(self):
        return self.node_level.max() + 1

    @property
    def n_leaves(self):
        return np.count_nonzero(self.first_child < 0)

    @property
    def node_lower(self):
        return (
            self.root_center
            - self.root_size / 2
            + self.node_index * self.node_size[:, None]
        )

    @property
    def node_size(self):
        return self.root_size / 2.0**self.node_level

    @property
    def leaf_lower(self):
        return self.node_lower[self.leaf_to_node]

    @property
    def leaf_size(self):
        return self.node_size[self.leaf_to_node]

    @property
    def leaf_to_node(self):
        return np.flatnonzero(self.first_child < 0)

    @property
    def node_to_leaf(self):
        node_to_leaf = np.full(len(self.first_child), -1)
        node_to_leaf[self.leaf_to_node] = np.arange(self.n_leaves)
        return node_to_leaf


def build_octree(
    x,
    y,
    z,
    h,
    root_center,
    root_size,
    min_leaf_size,
    max_leaf_size,
    n_leaves_per_h,
):
    if min_leaf_size <= 0:
        raise ValueError("min_leaf_size must be positive")
    root_lower = root_center - root_size / 2
    pos = np.column_stack([x, y, z])
    in_root = np.all((pos >= root_lower) & (pos < root_lower + root_size), axis=1)
    pos_in_root = (pos[in_root] - root_lower) / root_size
    h_in_root = h[in_root]

    node_index_by_level = [np.zeros((1, 3), dtype=np.int64)]
    first_child_by_level = []
    particle_to_node = np.zeros(len(pos_in_root), dtype=np.int64)

    max_level = 0
    while root_size / 2**max_level > min_leaf_size:
        max_level += 1
    for level in range(max_level + 1):
        # split the nodes
        node_index_at_level = node_index_by_level[level]
        node_size_at_level = root_size / 2**level
        h_min_in_node = np.full(len(node_index_at_level), np.inf)
        np.minimum.at(h_min_in_node, particle_to_node, h_in_root)
        # a node is split while larger than max_leaf_size, or than min_leaf_size and
        # the smallest h in it divided by n_leaves_per_h
        is_split_node = (node_size_at_level > max_leaf_size) | (
            (node_size_at_level > min_leaf_size)
            & (node_size_at_level > h_min_in_node / n_leaves_per_h)
        )
        n_nodes_so_far = sum(map(len, node_index_by_level))
        first_child_offset = 8 * np.arange(np.count_nonzero(is_split_node))
        first_child_at_level = np.full(len(node_index_at_level), -1, dtype=np.int64)
        first_child_at_level[is_split_node] = n_nodes_so_far + first_child_offset
        first_child_by_level.append(first_child_at_level)
        if not is_split_node.any():
            break

        # make the children; the 8 children of each split node follow one another,
        # and along each axis, a child's index is twice its parent's, plus bit x, y,
        # or z of its offset from the first child
        child_index = (
            2 * node_index_at_level[is_split_node][:, None, :]
            + ((np.arange(8)[:, None] >> np.arange(3)) & 1)
        ).reshape(-1, 3)
        node_index_by_level.append(child_index)

        # move the particles into the children
        in_split_node = is_split_node[particle_to_node]
        # the child holding each particle: along each axis, whether its index at the next
        # level is odd gives bit x, y, or z of its offset from the first child
        child_offset = (
            np.floor(pos_in_root[in_split_node] * 2 ** (level + 1)).astype(np.int64) % 2
        ) @ np.array([1, 2, 4])
        particle_to_node = (
            first_child_at_level[particle_to_node[in_split_node]]
            - n_nodes_so_far
            + child_offset
        )
        pos_in_root = pos_in_root[in_split_node]
        h_in_root = h_in_root[in_split_node]
    n_nodes_by_level = list(map(len, node_index_by_level))
    node_level = np.repeat(np.arange(len(n_nodes_by_level)), n_nodes_by_level)
    return Octree(
        root_center,
        root_size,
        node_level,
        np.concatenate(node_index_by_level),
        np.concatenate(first_child_by_level),
    )


def deposit_octree(x, y, z, h, quantities, octree, kernel, leaf_array=None):
    if leaf_array is None:
        leaf_array = np.zeros((octree.n_leaves, quantities.shape[1]))
    _deposit_octree(
        leaf_array,
        x,
        y,
        z,
        h,
        quantities,
        octree.n_levels,
        octree.node_lower,
        octree.node_size,
        octree.first_child,
        octree.node_to_leaf,
        KERNELS[kernel],
    )
    return leaf_array


@njit(cache=True)
def _deposit_octree(
    leaf_array,
    x,
    y,
    z,
    h,
    quantities,
    n_levels,
    node_lower,
    node_size,
    first_child,
    node_to_leaf,
    kernel_id,
):
    root_lower = node_lower[0]
    root_upper = root_lower + node_size[0]
    nodes_under_kernel_buffer = np.empty(len(node_size), dtype=np.int64)
    weight_buffer = np.empty(len(node_size))
    for k in range(x.size):
        in_root = _is_inside(x[k], y[k], z[k], 0.0, root_lower, root_upper)
        if in_root:
            particle_node = _leaf_at(
                x[k], y[k], z[k], node_lower, node_size, first_child
            )
            # a kernel much smaller than its leaf gives everything to the leaf
            if N_SUB_PER_H * node_size[particle_node] / h[k] > MAX_N_SUB:
                leaf_array[node_to_leaf[particle_node]] += quantities[k]
                continue
        nodes_under_kernel = _leaves_under_kernel(
            nodes_under_kernel_buffer,
            x[k],
            y[k],
            z[k],
            h[k],
            n_levels,
            node_lower,
            node_size,
            first_child,
        )
        weights = weight_buffer[: len(nodes_under_kernel)]
        for i, node in enumerate(nodes_under_kernel):
            weights[i] = _leaf_weight(
                x[k],
                y[k],
                z[k],
                h[k],
                node_lower[node, 0],
                node_lower[node, 1],
                node_lower[node, 2],
                node_size[node],
                kernel_id,
            )
        w_sum = weights.sum()
        # a kernel reaching no sample in the root adds nothing
        if w_sum == 0.0:
            continue
        # a kernel inside the root is normalized by the sum of its weights; one reaching
        # outside by its exact integral, 1, so what falls outside is lost
        is_kernel_in_root = _is_inside(x[k], y[k], z[k], h[k], root_lower, root_upper)
        norm = w_sum if is_kernel_in_root else 1.0
        for i, node in enumerate(nodes_under_kernel):
            w_normalized = weights[i] / norm
            leaf_array[node_to_leaf[node]] += w_normalized * quantities[k]


@njit(cache=True)
def _is_inside(x, y, z, margin, lower, upper):
    return (
        lower[0] <= x - margin
        and x + margin < upper[0]
        and lower[1] <= y - margin
        and y + margin < upper[1]
        and lower[2] <= z - margin
        and z + margin < upper[2]
    )


@njit(cache=True)
def _leaf_at(x, y, z, node_lower, node_size, first_child):
    node = 0
    while first_child[node] >= 0:
        half_size = node_size[node] / 2
        child_offset = (
            int(x >= node_lower[node, 0] + half_size)
            + 2 * int(y >= node_lower[node, 1] + half_size)
            + 4 * int(z >= node_lower[node, 2] + half_size)
        )
        node = first_child[node] + child_offset
    return node


@njit(cache=True)
def _leaves_under_kernel(
    nodes_under_kernel_buffer,
    x,
    y,
    z,
    h,
    n_levels,
    node_lower,
    node_size,
    first_child,
):
    # depth first; each level leaves at most 7 siblings waiting
    node_stack = np.empty(8 * n_levels, dtype=np.int64)
    node_stack[0] = 0
    n_stacked = 1
    n_nodes_under_kernel = 0
    while n_stacked > 0:
        n_stacked -= 1
        node = node_stack[n_stacked]
        x0, y0, z0 = node_lower[node, 0], node_lower[node, 1], node_lower[node, 2]
        size = node_size[node]
        if (
            x0 >= x + h
            or x0 + size <= x - h
            or y0 >= y + h
            or y0 + size <= y - h
            or z0 >= z + h
            or z0 + size <= z - h
        ):
            continue
        if first_child[node] >= 0:
            for child_offset in range(8):
                node_stack[n_stacked] = first_child[node] + child_offset
                n_stacked += 1
            continue
        nodes_under_kernel_buffer[n_nodes_under_kernel] = node
        n_nodes_under_kernel += 1
    return nodes_under_kernel_buffer[:n_nodes_under_kernel]


@njit(cache=True)
def _leaf_weight(x, y, z, h, x0, y0, z0, size, kernel_id):
    n_sub = max(1, int(np.ceil(N_SUB_PER_H * size / h)))
    sub_size = size / n_sub
    jx0 = max(0, _index(x - h, x0, sub_size))
    jx1 = min(n_sub - 1, _index(x + h, x0, sub_size))
    jy0 = max(0, _index(y - h, y0, sub_size))
    jy1 = min(n_sub - 1, _index(y + h, y0, sub_size))
    jz0 = max(0, _index(z - h, z0, sub_size))
    jz1 = min(n_sub - 1, _index(z + h, z0, sub_size))
    w = 0.0
    for jz in range(jz0, jz1 + 1):
        ddz = z0 + (jz + 0.5) * sub_size - z
        for jy in range(jy0, jy1 + 1):
            ddy = y0 + (jy + 0.5) * sub_size - y
            for jx in range(jx0, jx1 + 1):
                ddx = x0 + (jx + 0.5) * sub_size - x
                w += kernel(np.sqrt(ddx * ddx + ddy * ddy + ddz * ddz) / h, kernel_id)
    # leaves differ in size, so the samples are weighted by their volume to give the
    # kernel integrated over the leaf
    return w * (sub_size / h) ** 3

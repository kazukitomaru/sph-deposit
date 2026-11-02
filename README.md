# sph_deposit

Deposit SPH particles onto 2D grids, 3D grids, and octrees with numba.

Arrays are float64 ndarrays. `h` is the compact support radius of the kernel. `kernel` is `"cubic_spline"` or `"wendland_c4"`. `quantities` is `(n, n_q)`. A grid is given by `center`, `width`, and `n_cells`, one entry per axis in the order x, y(, z), and results are in the same order.

## 2D

### deposit_projection

```python
from sph_deposit import deposit_projection

deposit_projection(x, y, h, quantities, center, width, n_cells, kernel, cell_array=None)
```

Projects the particles along z onto a 2D grid.

- `x, y, h`: `(n,)`
- `quantities`: `(n, n_q)`, the total of each quantity in a particle (mass, not density)
- `center, width, n_cells`: `(2,)`
- `cell_array`: `(n_x, n_y, n_q)`, added into if given
- returns `(n_x, n_y, n_q)`, the total in each cell

### deposit_slice

```python
from sph_deposit import deposit_slice

deposit_slice(
    x, y, z, h, volume, quantities, center, width, n_cells, kernel, cell_array=None
)
```

Interpolates the quantities on the plane z = `center[2]`.

- `x, y, z, h`: `(n,)`
- `volume`: `(n,)`, the volume of each particle, such as mass / density
- `quantities`: `(n, n_q)`, the value of each quantity at a particle (density, temperature); ones give the kernel normalization
- `center`: `(3,)`; `width, n_cells`: `(2,)`
- `cell_array`: `(n_x, n_y, n_q)`, added into if given
- returns `(n_x, n_y, n_q)`, the interpolated value in each cell

## 3D grid

### deposit_grid

```python
from sph_deposit import deposit_grid

deposit_grid(x, y, z, h, quantities, center, width, n_cells, kernel, cell_array=None)
```

Deposits the particles onto a 3D grid.

- `x, y, z, h`: `(n,)`
- `quantities`: `(n, n_q)`, the total of each quantity in a particle
- `center, width, n_cells`: `(3,)`
- `cell_array`: `(n_x, n_y, n_z, n_q)`, added into if given
- returns `(n_x, n_y, n_z, n_q)`, the total in each cell

## Octree

### Octree

```python
from sph_deposit import Octree

Octree(root_center, root_size, node_level, node_index, first_child)
```

A dataclass holding a cubic root split into 8 equal children again and again, down to the leaves. Nodes are numbered level by level from the root (node 0), and leaves are numbered separately, in the order of the nodes. Made by `build_octree`.

Fields:

- `root_center`: `(3,)`; `root_size`: the side of the root
- `node_level`: `(n_nodes,)`, 0 for the root
- `node_index`: `(n_nodes, 3)`, integer position within the level
- `first_child`: `(n_nodes,)`, the 8 children follow it; -1 for leaves

Properties:

- `n_levels`: the number of levels, the root included
- `n_leaves`: the number of leaves
- `node_lower`: `(n_nodes, 3)`, the lower corner of each node
- `node_size`: `(n_nodes,)`, the side of each node
- `leaf_lower`: `(n_leaves, 3)`, the lower corner of each leaf
- `leaf_size`: `(n_leaves,)`, the side of each leaf
- `leaf_to_node`: `(n_leaves,)`, the node number of each leaf
- `node_to_leaf`: `(n_nodes,)`, the leaf number of each node; -1 for nodes that are not leaves

### build_octree

```python
from sph_deposit import build_octree

build_octree(
    x, y, z, h, root_center, root_size, min_leaf_size, max_leaf_size, n_leaves_per_h
)
```

Builds an octree whose leaves follow h. A node is split while larger than `max_leaf_size`, or larger than `min_leaf_size` and than the smallest h in it divided by `n_leaves_per_h`.

- `x, y, z, h`: `(n,)`; particles outside the root are ignored
- `root_center`: `(3,)`; `root_size`: the side of the root
- `min_leaf_size`: positive
- returns an `Octree`

### deposit_octree

```python
from sph_deposit import deposit_octree

deposit_octree(x, y, z, h, quantities, octree, kernel, leaf_array=None)
```

Deposits the particles onto the leaves of an octree.

- `x, y, z, h`: `(n,)`
- `quantities`: `(n, n_q)`, the total of each quantity in a particle
- `octree`: from `build_octree`
- `leaf_array`: `(n_leaves, n_q)`, added into if given
- returns `(n_leaves, n_q)`, the total in each leaf, in the order of `leaf_lower` and `leaf_size`

## Regridding

### regrid_cells

```python
from sph_deposit import regrid_cells

regrid_cells(x, y, size, quantities, center, width, n_cells)
```

Projects cubic cells along z onto a 2D grid, by area of overlap.

- `x, y`: `(n,)`, the centers of the cells
- `size`: `(n,)`, their sides
- `quantities`: `(n, n_q)`, the total of each quantity in a cell
- `center, width, n_cells`: `(2,)`
- returns `(n_x, n_y, n_q)`, the total in each cell of the grid

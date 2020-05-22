import numpy as np
from numba import int32, experimental


@experimental.jitclass(spec=[
    ('_size', int32[:]),
    ('_parent', int32[:]),
])
class UnionFind:
    def __init__(self, size: int):
        self._size = np.array([1] * size, dtype=np.int32)  # size of the component - correct only for roots
        self._parent = np.array(list(range(size)), dtype=np.int32)  # parent: for the internal tree structure

    def find(self, x: int) -> int:
        while x != self._parent[x]:
            # path compression
            q = self._parent[x]
            self._parent[x] = self._parent[q]
            x = q
        return x

    def union(self, x: int, y: int) -> int:
        xroot = self.find(x)
        yroot = self.find(y)

        if xroot == yroot:
            return xroot

        if self._size[xroot] < self._size[yroot]:
            self._parent[xroot] = yroot
            self._size[yroot] += self._size[xroot]
        else:
            self._parent[yroot] = xroot
            self._size[xroot] += self._size[yroot]

        return self._parent[xroot]

    def connected(self, x: int, y: int) -> bool:
        return self.find(x) == self.find(y)

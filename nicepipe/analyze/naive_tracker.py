"""
A naive implementation of tracker using cosine similarity of vectors,
so simple I can do it from memory.
In this case, keeping feature history would actually hurt performance,
but given tracking drop-out is not fatal... not is the ID really that
impt for anything besides tracking only visible instances that are
standing still...
yeah this shit good enough
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class NaiveTracker:
    _prev = None

    def sort(self, objs: np.ndarray):
        """expects shape of (N, ...)"""
        if self._prev is None:
            self._prev = objs
            # must index with list, not tuple, else it will reduce dimension
            return list(range(len(objs)))

        inds = []
        # below code ensures the same pose isnt assigned more than once
        # but its inefficient & rarely happens anyways
        for n in range(min(len(objs), len(self._prev))):
            ind = np.argmin(
                tuple(
                    np.Inf if i in inds else np.linalg.norm(a - self._prev[n])
                    for i, a in enumerate(objs)
                )
            )
            inds.append(ind)
        if len(self._prev) < len(objs):
            inds += list(set(range(len(objs))).difference(inds))

        self._prev = objs[inds]
        return inds  # list not tuple, else dimension reduces

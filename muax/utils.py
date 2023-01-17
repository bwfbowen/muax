from collections import deque
from itertools import islice
import jax


class sliceable_deque(deque):
    r"""A class implemented slice for collections.deque
    
    Reference: https://stackoverflow.com/questions/10003143/how-to-slice-a-deque
    """
    def __getitem__(self, index):
        if isinstance(index, slice):
            return type(self)(islice(self, index.start,
                                               index.stop, index.step))
        return deque.__getitem__(self, index)

jax.tree_util.register_pytree_node(
  sliceable_deque,
  flatten_func=lambda sd: (sd, None),
  unflatten_func=lambda treedef, leaves: sliceable_deque(leaves)
)

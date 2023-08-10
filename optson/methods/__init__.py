"""Please note that some methods that rely on line-search
or trust-region approaches can only be used either without using mini-batches or with overlapping batches,
such as provided by the `ControlGroupBatchManager`.

"""
# This simplifies imports of different methods, e.g.
# from optson.methods import *
# from optson.methods import AdamUpdate, QuasiNewtonUpdate, SteepestDescentUpdate, BasicTRUpdate

from .adam import AdamUpdate
from .quasi_newton import QuasiNewtonUpdate
from .steepest_descent import SteepestDescentUpdate
from .trust_region import BasicTRUpdate

__all__ = ["AdamUpdate", "QuasiNewtonUpdate", "SteepestDescentUpdate", "BasicTRUpdate"]

import numpy as np

from optson.preconditioner import IdentityPreconditioner


def test_identity_preconditioner():
    precond = IdentityPreconditioner()
    g = np.ones(2)
    p = np.ones(2)
    assert np.dot(precond(g), p) == np.dot(g, p)

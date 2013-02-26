""" Taken directly from the scipy 1.7 source code... """



import numpy as np
from scipy import isscalar, sort, array, add


def percentile(a, q, axis=None, out=None, overwrite_input=False):
    a = np.asarray(a)

    if q == 0:
        return a.min(axis=axis, out=out)
    elif q == 100:
        return a.max(axis=axis, out=out)

    if overwrite_input:
        if axis is None:
            sorted = a.ravel()
            sorted.sort()
        else:
            a.sort(axis=axis)
            sorted = a
    else:
        sorted = sort(a, axis=axis)
    if axis is None:
        axis = 0

    return _compute_qth_percentile(sorted, q, axis, out)

# handle sequence of q's without calling sort multiple times
def _compute_qth_percentile(sorted, q, axis, out):
    if not isscalar(q):
        p = [_compute_qth_percentile(sorted, qi, axis, None)
             for qi in q]

        if out is not None:
            out.flat = p

        return p

    q = q / 100.0
    if (q < 0) or (q > 1):
        raise ValueError("percentile must be either in the range [0,100]")

    indexer = [slice(None)] * sorted.ndim
    Nx = sorted.shape[axis]
    index = q * (Nx - 1)
    i = int(index)
    if i == index:
        indexer[axis] = slice(i, i + 1)
        weights = array(1)
        sumval = 1.0
    else:
        indexer[axis] = slice(i, i + 2)
        j = i + 1
        weights = array([(j - index), (index - i)], float)
        wshape = [1] * sorted.ndim
        wshape[axis] = 2
        weights.shape = wshape
        sumval = weights.sum()

    # Use add.reduce in both cases to coerce data type as well as
    # check and use out array.
    return add.reduce(sorted[indexer] * weights, axis=axis, out=out) / sumval

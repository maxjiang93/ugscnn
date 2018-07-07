import numpy as np
import pyigl as igl
import scipy.sparse as sparse


def unique_rows(data, digits=None):
    """
    Returns indices of unique rows. It will return the
    first occurrence of a row that is duplicated:
    [[1,2], [3,4], [1,2]] will return [0,1]
    Parameters
    ---------
    data: (n,m) set of floating point data
    digits: how many digits to consider for the purposes of uniqueness
    Returns
    --------
    unique:  (j) array, index in data which is a unique row
    inverse: (n) length array to reconstruct original
                 example: unique[inverse] == data
    """
    hashes = hashable_rows(data, digits=digits)
    garbage, unique, inverse = np.unique(hashes,
                                         return_index=True,
                                         return_inverse=True)
    return unique, inverse


def hashable_rows(data, digits=None):
    """
    We turn our array into integers based on the precision
    given by digits and then put them in a hashable format.
    Parameters
    ---------
    data:    (n,m) input array
    digits:  how many digits to add to hash, if data is floating point
             If none, TOL_MERGE will be turned into a digit count and used.
    Returns
    ---------
    hashable:  (n) length array of custom data which can be sorted
                or used as hash keys
    """
    # if there is no data return immediatly
    if len(data) == 0:
        return np.array([])

    # get array as integer to precision we care about
    as_int = float_to_int(data, digits=digits)

    # if it is flat integers already, return
    if len(as_int.shape) == 1:
        return as_int

    # if array is 2D and smallish, we can try bitbanging
    # this is signifigantly faster than the custom dtype
    if len(as_int.shape) == 2 and as_int.shape[1] <= 4:
        # time for some righteous bitbanging
        # can we pack the whole row into a single 64 bit integer
        precision = int(np.floor(64 / as_int.shape[1]))
        # if the max value is less than precision we can do this
        if np.abs(as_int).max() < 2**(precision - 1):
            # the resulting package
            hashable = np.zeros(len(as_int), dtype=np.int64)
            # loop through each column and bitwise xor to combine
            # make sure as_int is int64 otherwise bit offset won't work
            for offset, column in enumerate(as_int.astype(np.int64).T):
                # will modify hashable in place
                np.bitwise_xor(hashable,
                               column << (offset * precision),
                               out=hashable)
            return hashable

    # reshape array into magical data type that is weird but hashable
    dtype = np.dtype((np.void, as_int.dtype.itemsize * as_int.shape[1]))
    # make sure result is contiguous and flat
    hashable = np.ascontiguousarray(as_int).view(dtype).reshape(-1)
    return hashable


def float_to_int(data, digits=None, dtype=np.int32):
    """
    Given a numpy array of float/bool/int, return as integers.
    Parameters
    -------------
    data:   (n, d) float, int, or bool data
    digits: float/int precision for float conversion
    dtype:  numpy dtype for result
    Returns
    -------------
    as_int: data, as integers
    """
    # convert to any numpy array
    data = np.asanyarray(data)

    # if data is already an integer or boolean we're done
    # if the data is empty we are also done
    if data.dtype.kind in 'ib' or data.size == 0:
        return data.astype(dtype)

    # populate digits from kwargs
    if digits is None:
        digits = decimal_to_digits(1e-8)
    elif isinstance(digits, float) or isinstance(digits, np.float):
        digits = decimal_to_digits(digits)
    elif not (isinstance(digits, int) or isinstance(digits, np.integer)):
        log.warn('Digits were passed as %s!', digits.__class__.__name__)
        raise ValueError('Digits must be None, int, or float!')

    # data is float so convert to large integers
    data_max = np.abs(data).max() * 10**digits
    # ignore passed dtype if we have something large
    dtype = [np.int32, np.int64][int(data_max > 2**31)]
    # multiply by requested power of ten
    # then subtract small epsilon to avoid "go either way" rounding
    # then do the rounding and convert to integer
    as_int = np.round((data * 10 ** digits) - 1e-6).astype(dtype)

    return as_int


def decimal_to_digits(decimal, min_digits=None):
    """
    Return the number of digits to the first nonzero decimal.
    Parameters
    -----------
    decimal:    float
    min_digits: int, minumum number of digits to return
    Returns
    -----------
    digits: int, number of digits to the first nonzero decimal
    """
    digits = abs(int(np.log10(decimal)))
    if min_digits is not None:
        digits = np.clip(digits, min_digits, 20)
    return digits


def p2e(m):
    if isinstance(m, np.ndarray):
        if not (m.flags['C_CONTIGUOUS'] or m.flags['F_CONTIGUOUS']):
            raise TypeError('p2e support either c-order or f-order')
        if m.dtype.type in [np.int32, np.int64]:
            return igl.eigen.MatrixXi(m.astype(np.int32))
        elif m.dtype.type in [np.float64, np.float32]:
            return igl.eigen.MatrixXd(m.astype(np.float64))
        elif m.dtype.type == np.bool:
            return igl.eigen.MatrixXb(m)
        raise TypeError("p2e only support dtype float64/32, int64/32 and bool")
    if sparse.issparse(m):
        # convert in a dense matrix with triples
        coo = m.tocoo()
        triplets = np.vstack((coo.row, coo.col, coo.data)).T

        triples_eigen_wrapper = igl.eigen.MatrixXd(triplets)

        if m.dtype.type == np.int32:
            t = igl.eigen.SparseMatrixi()
            t.fromcoo(triples_eigen_wrapper)
            return t
        elif m.dtype.type == np.float64:
            t = igl.eigen.SparseMatrixd()
            t.fromCOO(triples_eigen_wrapper)
            return t


    raise TypeError("p2e only support numpy.array or scipy.sparse")


def e2p(m):
    if isinstance(m, igl.eigen.MatrixXd):
        return np.array(m, dtype='float64', order='C')
    elif isinstance(m, igl.eigen.MatrixXi):
        return np.array(m, dtype='int32', order='C')
    elif isinstance(m, igl.eigen.MatrixXb):
        return np.array(m, dtype='bool', order='C')
    elif isinstance(m, igl.eigen.SparseMatrixd):
        coo = np.array(m.toCOO())
        I = coo[:, 0]
        J = coo[:, 1]
        V = coo[:, 2]
        return sparse.coo_matrix((V,(I,J)), shape=(m.rows(),m.cols()), dtype='float64')
    elif isinstance(m, igl.eigen.SparseMatrixi):
        coo = np.array(m.toCOO())
        I = coo[:, 0]
        J = coo[:, 1]
        V = coo[:, 2]
        return sparse.coo_matrix((V,(I,J)), shape=(m.rows(),m.cols()), dtype='int32')
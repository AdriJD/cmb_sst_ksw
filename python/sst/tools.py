'''
Collection of tools used for fisher analysis
'''

import numpy as np
import warnings
import scipy.stats as ss

def combine(self, a, b, c):
    '''
    Combine three int arrays into a single int array
    by packing them bitwise. Only works for
    non-negative ints < 16384
    '''

    # Do some checks because this is a scary function.
    a = np.asarray(a)
    b = np.asarray(b)
    c = np.asarray(c)

    if a.dtype != np.int64:
        raise ValueError('a')

    if b.dtype != np.int64:
        raise ValueError('b')

    if c.dtype != np.int64:
        raise ValueError('c')

    if np.max(a) >= 16384 or np.any(a < 0):
        raise ValueError('a')

    if np.max(b) >= 16384 or np.any(b < 0):
        raise ValueError('b')

    if np.max(c) >= 16384 or np.any(c < 0):
        raise ValueError('c')

    # shift a 28 bits to the left, b 14
    return (a << 28) | (b << 14) | c

def unpack(self, comb):
    '''
    Unpack int array into 3 arrays

    comb: int, array-like
    '''        

    comb = np.asarray(comb)
    if comb.dtype != np.int64:
        raise ValueError('d')

    # shift a 28 bits back to right
    a = comb >> 28
    # shift b 14 bits back to right 
    # and set all above 16383 to zero (i.e. a)
    b = (comb >> 14) & 16383
    # set all above 16383 to zero (i.e. a and b)
    c = (comb & 16383)

    return a, b, c

def binned_statistic(*args, **kwargs):
    ''' scipy.stats.binned_statistics without FutureWarnings'''

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore",category=FutureWarning)
        ret = ss.binned_statistic(*args, **kwargs)
    return ret

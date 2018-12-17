'''
Collection of tools used for fisher analysis
'''

import numpy as np
import warnings
import scipy.stats as ss
from numba import jit

def combine(a, b, c):
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

def unpack(comb):
    '''
    Unpack int array into 3 arrays

    Arguments
    ---------
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

@jit(nopython=True)
def init_bins_jit(bins, idxs_on_rank, num_pass, first_pass, pmod):
    '''
    Compute first and number of good triplets in bins.

    Arguments
    ---------
    bins : array-like
        Left (lower) side of bins.
    bins_outer : array-like
        Indices to bins on rank (ell_1).
    num_pass : array-like
        Modified in-place. Array of shape (bin_outer, bins, bins)
    first_pass : array-like
        Modified in-place. Array of shape (bin_outer, bins, bins, 3)
    pmod : int
        1 if parity is 'odd', 0 for 'even', 2 for 'both'.
    '''

    lmax = bins[-1]

    for bidx1, bidx1_full in enumerate(idxs_on_rank):
        # Note, bidx1 is index to this ranks bin arrays only.
        # bidx1_full = index to full bins array.
        bin1 = bins[bidx1_full]

        if bin1 == lmax:
            bmax1 = lmax
        else:
            bmax1 = bins[bidx1_full+1] - 1

        for bidx2 in xrange(bidx1_full, bins.size):
            bin2 = bins[bidx2]
            if bin2 == lmax:
                bmax2 = lmax
            else:
                bmax2 = bins[bidx2+1] - 1

            for bidx3 in xrange(bidx2, bins.size):
                bin3 = bins[bidx3]
                if bin3 == lmax:
                    bmax3 = lmax
                else:
                    bmax3 = bins[bidx3+1] - 1

                if bin3 > (bmax1 + bmax2):
                    # Triangle condition will always fail.
                    break

                # Loop through bin, again only the l1 <= l2 <= l3 part
                n = 0
                f1 = 0
                f2 = 0
                f3 = 0
                for ell1 in xrange(bin1, bmax1+1):
                    for ell2 in xrange(max(ell1, bin2), bmax2+1):
                        for ell3 in xrange(max(ell2, bin3), bmax3+1):

                            # RHS triangle ineq.
                            if ell3 > (ell1 + ell2):
                                # There will no valid one after this.
                                break                            

                            if ell3 < ell2 or ell3 < ell1:
                                # This takes care of |l1 - l2| <= l3
                                continue
                                
                            # Parity. Pmod = 1 if parity == 0
                            if pmod != 2:
                                if (ell1 + ell2 + ell3) % 2 != pmod:
                                    continue

                            n += 1
                            if n == 1:
                                # First pass.
                                f1 = ell1
                                f2 = ell2
                                f3 = ell3

                # Modify in-place.
                num_pass[bidx1, bidx2, bidx3] = n
                first_pass[bidx1, bidx2, bidx3,0] = f1
                first_pass[bidx1, bidx2, bidx3,1] = f2
                first_pass[bidx1, bidx2, bidx3,2] = f3

    return

def init_bins_numpy(bins, idxs_on_rank, num_pass, first_pass, pmod):
    '''
    Old version of init_bins that is not used anymore, but is kept
    as cross check.

    Arguments
    ---------
    bins : array-like
        Left (lower) side of bins.
    bins_outer : array-like
        Indices to bins on rank (ell_1).
    num_pass : array-like
        Modified in-place. Array of shape (bin_outer, bins, bins)
    first_pass : array-like
        Modified in-place. Array of shape (bin_outer, bins, bins, 3)
    pmod : int
        1 if parity is 'odd', 0 for 'even', 2 for 'both'.
    '''

    for idx1, idx1_full in enumerate(idxs_on_rank):
        # Note, idx1 is index to this ranks bin arrays only.
        # idx1_full = index to full bins array.
        bin1 = bins[idx1_full]

        try:
            lmax1 = bins[idx1_full + 1] - 1
        except IndexError:
            # We are in last bin, use lmax.
            lmax1 = lmax

        for idx2, idx2_full in enumerate(idx[idx1_full:]):
            bin2 = bins[idx2_full]

            try:
                lmax2 = bins[idx2_full + 1] - 1
            except IndexError:
                # We are in last bin, use lmax.
                lmax2 = lmax

            for idx3, idx3_full in enumerate(idx[idx2_full:]):
                bin3 = bins[idx3_full]

                try:
                    lmax3 = bins[idx3_full + 1] - 1
                except IndexError:
                    # We are in last bin, use lmax.
                    lmax3 = lmax

                # Exclude triangle.
                if bin3 > (lmax1 + lmax2):
                    break

                for ell1 in xrange(bin1, lmax1+1):
                    for ell2 in xrange(max(ell1, bin2), lmax2+1):

                        ells3 = np.arange(max(ell2, bin3), lmax3+1)
                        ba = np.ones(ells3.size, dtype=bool)

                        # Exclude parity odd/even.
                        if pmod != 2:
                            if (ell1 + ell2) % 2:
                                # Odd sum, l3 must be even if parity=odd.
                                ba[ells3%2 == pmod] *= False
                            else:
                                # Even sum, l3 must be odd if parity=odd.
                                ba[~(ells3%2 == pmod)] *= False

                        # exclude triangle
                        ba[(abs(ell1 - ell2) > ells3) | (ells3 > (ell1 + ell2))] \
                            *= False

                        # Use boolean index array to determine good ell3s
                        gd_ell3s = ells3[ba]
                        n_pass = np.sum(ba)

                        n_in_bin = num_pass[idx1,idx2_full,idx3_full]

                        if n_pass != 0 and n_in_bin == 0:
                            # No good tuples in this bin yet but we
                            # just found at least one.
                            first_pass[idx1, idx2_full, idx3_full] \
                                = ell1, ell2, gd_ell3s[0]

                        num_pass[idx1,idx2_full,idx3_full] += n_pass

    @jit(nopython=True)
    def contract_bcb(B, C):
        '''
        Calculate B^T C B.
        
        Arguments
        ---------
        B : array-like
        C : array-like
        '''
        
        return np.dot(B, np.dot(C, B))

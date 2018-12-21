'''
Collection of tools used for fisher analysis
'''

import numpy as np
import warnings
import scipy.stats as ss
import numba

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

@numba.jit(nopython=True)
def init_bins_jit(bins, idxs_on_rank, num_pass, first_pass, pmod):
    '''
    Compute first and number of good triplets per bin.

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

@numba.jit(nopython=True)
def area_trapezoid(ell1, lmax):
    '''
    Return the area of the isosceles trapezoid
    that the angle-average bispectrum traces out 
    in l2 x l3.

    Arguments
    ---------
    ell1 : int
    lmax : int

    Returns
    -------
    area : float

    Notes
    -----
              a
        /-----------\
    c  /             \
      /______________ \
              b

    '''

    a = np.sqrt(2 * max((lmax - 2 * ell1), 0) ** 2)
    b = np.sqrt(2 * max((lmax - ell1), 0) ** 2)
    c = min(ell1, lmax - ell1)

    area = 0.5 * (a + b) * np.sqrt(max(c * c - 0.25 * (b - a) * (b - a), 0))

    return area

@numba.vectorize(nopython=True)
def estimate_n_tuples(ell1, lmax):
    '''
    Estimate the number of (l1, l2, l3) triples
    allowed by triangle condition given l1 and lmax.

    Arguments
    ---------
    ell1 : int
    lmax : int            

    Returns
    -------
    estimate : int
        Estimated number of triplets per l1 multipole.

    Notes
    -----
    Returns area of trapezoid spanned by allowed ell2, ell3 
    doublets + perimeter of the square around the trapezoid 
    (to correct for the discrete nature of the multipole triplets).
    '''

    area = area_trapezoid(ell1, lmax) 

    perimeter = 4 * (max(lmax - ell1, 0))

    return area + perimeter

def rank_bins(bins, n_per_ell, ells):
    '''
    Return array of bin indices that will sort `bins` 
    from small to large number of good tuples.

    Arguments
    ---------
    bins : array-like
        Left/lower side of bins. Monotonically increasing.
    n_per_ell : array-like
        Number of valid tuples per multipole.
    ells : array-like
        Multipoles (same lenght as n_tuples_per_ell)

    Returns
    -------
    bidx_sorted : ndarray
        Index array such that bins[bidx_sorted] is sorted.
    n_per_bin : ndarray
        Binned version of n_per_ell (in original order).
    '''

    
    bins_ext = np.empty(len(bins) + 1, dtype=float)
    bins_ext[:-1] = np.asarray(bins, dtype=float)
    bins_ext[-1] = bins[-1] + 0.1
    n_per_bin, _, _ = binned_statistic(ells, n_per_ell,
                                bins=bins_ext, statistic='sum')

    bidx_sorted = np.argsort(n_per_bin)
    
    return bidx_sorted, n_per_bin

def distribute_bins(bidx_sorted, n_per_bin, size, tol=0.1):
    '''
    Find roughly optimal bin indices per MPI rank based on
    uniformity in "n".

    Arguments
    ---------
    bidx_sorted
        Index array such that bins[bidx_sorted] is in increasing order.
    n_per_bin
        Number per bin (in original order).
    size : int
        Number of MPI ranks.

    Keyword arguments
    -----------------
    tol : float
        Fractional tolerance in load difference between ranks.
        (default : 0.1)

    Returns
    -------
    bidx_per_rank : list of ndarrays
        Bin indices per rank. Shape=(size, nbins)

    Notes
    -----
    Idea is that rank 0 gets the bin with the most work
    then rank 1 gets next-to-largest bin + a number of 
    small-load bins etc. This is then repeated after
    all ranks are treated.
    '''

    n_per_bin_sorted = n_per_bin[bidx_sorted] # Smallest first.
    n_cum = np.cumsum(n_per_bin_sorted)

    bidx_per_rank = [[] for i in xrange(size)]
    
    num_bins = bidx_sorted.size

    mask = np.ones(bidx_sorted.size, dtype=bool)
    i = 1 # Counter over sorted bidx array right2left.
    start = 0 # Counter for left2right.
    end = None
    bins_left = True
    ii = 0
    while bins_left:
        
        # Because I hate while loops.
        ii += 1
        if ii > 1000:
            raise ValueError('Cannot find bin distribution')

        for rank in xrange(size):

            if i > num_bins or -i == end:
                bins_left = False
                break

            if rank == 0:
                # Rank 0 receives bin with largest value.                
                if mask[-i]:
                    bidx_per_rank[rank].append(bidx_sorted[-i])
                    n_max = n_per_bin_sorted[-i]
                    mask[-i] = False
                    i += 1
                else:
                    bins_left = False
                    break
            
            if rank > 0:
                # Always give bin index with next largest value.
                if mask[-i]:
                    bidx = bidx_sorted[-i]
                    bidx_per_rank[rank].append(bidx)
                    mask[-i] = False
                else:
                    bins_left = False
                    break

                # Find n for this rank
                n_rank = n_per_bin_sorted[-i]

                # Find more bins until n exceeds n_max.
                target = (1 + tol) * n_max - n_rank
                indices = np.where(n_cum[start:] <= target)[0] 

                # NOTE, search needs to update its starting point...
                if indices.size > 0:
                    indices += start
                    start = indices[-1] + 1
                    end = indices[-1]
                    mask[indices] = False

                bidx2add = bidx_sorted[indices] # Empty if none are.

                for bi in bidx2add:
                    if bi == bidx:
                        # We have met in the middle.
                        bins_left = False
                        break
                    else:
                        bidx_per_rank[rank].append(bi)

                try:
                    if indices[-1] == i + 1:
                        # This index will be considered next.
                        bins_left = False
                except IndexError:
                    # Empty array.
                    pass

                if not bins_left:
                    break

                i += 1
                
    # Turn inner lists in arrays.
    check = []
    for i in xrange(len(bidx_per_rank)):
        check.append(bidx_per_rank[i])
        bidx_per_rank[i] = np.asarray(bidx_per_rank[i], dtype=int)
    check = [item for sublist in check for item in sublist]

    check = np.asarray(check)
    if not np.array_equal(np.unique(check), np.arange(num_bins)):
        raise ValueError('bins not distributed right')

    return bidx_per_rank
                
def get_good_triplets(bmin, bmax, lmax, good_triplets, pmod):
    '''
    Fill array with all valid triplets.

    Arguments
    ---------
    bmin : int
    bmax : int
    lmax : int
    good_triplets : ndarray, dtype=int
        Shape (N, 3)
    pmod : int
    '''

    ret = _get_good_triplets(bmin, bmax, lmax, good_triplets, pmod)
    if ret != 0:
        raise ValueError('Error: {}, good_triplets has wrong size: {}'.format(
            ret, good_triplets.shape))
    else:
        return

@numba.jit(nopython=True)
def _get_good_triplets(bmin, bmax, lmax, good_triplets, pmod):
    '''
    Fill array with all valid triplets.

    Arguments
    ---------
    bmin : int
    bmax : int
    lmax : int
    good_triplets : ndarray, dtype=int
        Shape (N, 3)
    pmod : int
    '''

    input_size = good_triplets.shape[0]

    ii = 0 # Iterate over triplets.
    for ell1 in xrange(bmin, bmax + 1):
        
        for ell2 in xrange(ell1, lmax + 1):            
            for ell3 in xrange(ell2, lmax + 1):

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

                # Error checking. If input arr too small
                # you might get seg. faults etc.
                if ii >= input_size:
                    return -2

                good_triplets[ii,0] = ell1
                good_triplets[ii,1] = ell2
                good_triplets[ii,2] = ell3

                ii += 1

    # Another check, for too large input array.
    if ii != input_size:
        return -1

    return 0
    
@numba.jit(nopython=True)
def contract_bcb(B, C):
    '''
    Calculate B^T C B.

    Arguments
    ---------
    B : array-like
    C : array-like
    '''

    return np.dot(B, np.dot(C, B))

@numba.jit(nopython=True)
def contract_bcb_loop(B, C, ii, jj):
    
    ans = 0.
    for i in xrange(ii):
        Bt = B[i]
        Ct = C[i,:]
        temp = 0
        for j in xrange(jj):
            temp += Bt * Ct[j] * B[j]

        ans += temp
    return ans

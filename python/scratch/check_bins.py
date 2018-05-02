'''
Test binning
'''

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import sys
import os
import numpy as np
from scipy.special import spherical_jn
sys.path.insert(0,'./../')
import fisher

opj = os.path.join

def bin_test(parity, bins=None, lmin=2, lmax=23):
    '''
    Init bins, run over them and test them.
    '''


    F = fisher.Fisher()

    F.init_bins(lmin=lmin, lmax=lmax, parity=parity, bins=bins)

    bins = F.bins
    pint = 1 if parity == 'odd' else 0

    for i1, b1 in enumerate(bins):
        for i2, b2 in enumerate(bins[i1:]):
            i2 += i1
            for i3, b3 in enumerate(bins[i2:]):
                i3 += i2
                ell1, ell2, ell3 = F.first_pass[i1,i2,i3]
                num = F.num_pass[i1,i2,i3]
                try:
                    if num == 0:
                        assert (ell1, ell2, ell3) == (0,0,0) 
                    else:
                        assert (ell1, ell2, ell3) != (0,0,0) 
                        if parity is not None:
                            assert (ell1 + ell2 + ell3) % 2 == pint 
                except:
                    print 'error in bin:'
                    print 'bin_idx: ({},{},{}), bin: ({},{},{}), no. gd_bins: {}, '\
                        'u_ell: ({},{},{})'.format(i1, i2, i3, b1, b2, b3, 
                                                   F.num_pass[i1,i2,i3], 
                                                   ell1, ell2, ell3)
                    raise
                
#                print 'bin_idx: ({},{},{}), bin: ({},{},{}), no. gd_bins: {}, '\
#                    'u_ell: ({},{},{})'.format(i1, i2, i3, b1, b2, b3, 
#                                               F.num_pass[i1,i2,i3], 
#                                               ell1, ell2, ell3)


    print 'bins: ', F.bins
    print 'lmin: ', F.lmin
    print 'lmax: ', F.lmax
    print 'sum num_pass: ', np.sum(F.num_pass)
    print 'unique_ells: ', F.unique_ells
    print 'num bins: ', F.bins.size
    print 'shape num_pass: ', F.num_pass.shape
    print 'shape first_pass: ', F.first_pass.shape, '\n'

if __name__ == '__main__':
    
    lmin = 2
    lmax = 23
    for parity in ['even', 'odd', None]:
        print 'parity: {}, lmin: {}, lmax: {}'.format(parity, lmin, lmax)
        print 'default bins'
        bin_test(lmin=lmin, lmax=lmax, parity=parity, bins=None)
        print 'bins up to lmax'
        bin_test(lmin=lmin, lmax=lmax, parity=parity, bins=[2,3,4,10,lmax])
        print 'bins over lmax'
        bin_test(lmin=lmin, lmax=lmax, parity=parity, bins=[2,3,4,10,lmax+12])

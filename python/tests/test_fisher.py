import unittest
import numpy as np
import os
from sst import Fisher

opj = os.path.join

class TestTools(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        pass

    @classmethod
    def tearDownClass(cls):
        pass

    def test_init(self):
        '''Test whether the class instance is initiated correctly. '''

        pass

    def loop_over_bins(self, lmin, lmax, parity, bins):

        F = Fisher('.')

        F.init_bins(lmin=lmin, lmax=lmax, parity=parity, bins=bins)

        bins = F.bins['bins']
        pint = 1 if parity == 'odd' else 0

        for i1, b1 in enumerate(bins):
            for i2, b2 in enumerate(bins):
                for i3, b3 in enumerate(bins):
                    ell1, ell2, ell3 = F.bins['first_pass'][i1,i2,i3]
                    num = F.bins['num_pass'][i1,i2,i3]
                    try:
                        if b1 > b2:
                            assert num == 0
                        if b2 > b3:
                            assert num == 0

                        if num == 0:
                            assert (ell1, ell2, ell3) == (0,0,0) 
                        else:
                            assert (ell1, ell2, ell3) != (0,0,0) 
                            if parity is not None:
                                assert (ell1 + ell2 + ell3) % 2 == pint 

                            # check if first pass ells fit in bins
                            assert ell1 >= b1
                            assert ell2 >= b2
                            assert ell3 >= b3

                            try:
                                assert ell1 < bins[i1+1]
                            except IndexError:
                                if (i1 + 1) >= bins.size:
                                    pass
                                else:
                                    raise
                            try:
                                assert ell2 < bins[i2+1]
                            except IndexError:
                                if (i2 + 1) >= bins.size:
                                    pass
                                else:
                                    raise
                            try:
                                assert ell3 < bins[i3+1]
                            except IndexError:
                                if (i3 + 1) >= bins.size:
                                    pass
                                else:
                                    raise

                            # Check if first pass matches triangle cond.
                            assert abs(ell1 - ell2) <= ell3
                            assert ell3 <= (ell1 + ell2)

                    except:
                        print 'error in bin:'
                        print 'bin_idx: ({},{},{}), bin: ({},{},{}), no. gd_tuples: {}, '\
                            'u_ell: ({},{},{})'.format(i1, i2, i3, b1, b2, b3, 
                                                       F.bins['num_pass'][i1,i2,i3], 
                                                       ell1, ell2, ell3)
                        raise
                
    def test_bins(self):

        lmin = 2
        lmax = 40
        for parity in ['even', 'odd', None]:
            self.loop_over_bins(lmin=lmin, lmax=lmax, parity=parity, bins=None)
            self.loop_over_bins(lmin=None, lmax=lmax, parity=parity, bins=[2,3,4,10,lmax])
            self.loop_over_bins(lmin=None, lmax=lmax, parity=parity, bins=[2,3,4,10,lmax+12])

    def test_first_num_pass(self):
        
        F = Fisher('.')
        F.init_bins(lmin=10, lmax=30, parity='odd')

        n_gd_bins = np.sum(F.bins['num_pass_full'].astype(bool))
        n_gd_tripl = np.sum(F.bins['first_pass_full'].astype(bool)) / 3.

        assert n_gd_bins == n_gd_tripl

        passed_trpl = F.bins['first_pass_full'][F.bins['num_pass_full'].astype(bool)]
        assert n_gd_bins == np.sum(passed_trpl.astype(bool)) / 3.

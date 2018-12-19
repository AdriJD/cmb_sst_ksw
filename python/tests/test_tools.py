import unittest
import numpy as np
import os
from sst import tools

opj = os.path.join

class TestTools(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        pass

    @classmethod
    def tearDownClass(cls):
        pass

    def test_area_trapezoid(self):

        lmax = 2
        ell1 = 1
        ans = 0.5
        self.assertEqual(tools.area_trapezoid(ell1, lmax), ans)

        lmax = 3000
        ell1 = 1500
        ans = 1.125e6
        self.assertEqual(tools.area_trapezoid(ell1, lmax), ans)

        lmax = 0
        ell1 = 0
        ans = 0. # Single point so zero area, although 1 good tuple.
        self.assertEqual(tools.area_trapezoid(ell1, lmax), ans)

        lmax = 3000
        ell1 = 4000
        ans = 0
        self.assertEqual(tools.area_trapezoid(ell1, lmax), ans)

        lmax = 3000
        ell1 = 3001
        ans = 0
        self.assertEqual(tools.area_trapezoid(ell1, lmax), ans)

    def test_estimate_n_tuples(self):

        lmax = 2
        ell1 = 1
        ans = 0.5 + 4
        self.assertEqual(tools.estimate_n_tuples(ell1, lmax), ans)

        lmax = 0
        ell1 = 0
        ans = 0.
        self.assertEqual(tools.estimate_n_tuples(ell1, lmax), ans)

        lmax = 3000
        ell1 = 3001
        ans = 0
        self.assertEqual(tools.estimate_n_tuples(ell1, lmax), ans)

        lmax = 4
        ell1s = np.arange(3, dtype=int)
        ans = np.asarray([0 + 16, 2.5 + 12, 2 + 8],
                         dtype=float)

        np.testing.assert_array_equal(
            tools.estimate_n_tuples(ell1s, lmax), ans)

    def test_rank_bins(self):
        
        bins = np.asarray([2, 3, 6, 10])
        ells = np.arange(2, 20)
        val_per_ell = np.ones_like(ells)
        bidx_sorted, n_per_bin = tools.rank_bins(bins, val_per_ell, ells)
        
        exp_bidx = np.asarray([0, 3, 1, 2], dtype=int)
        exp_n = np.asarray([1, 3, 4, 1], dtype=int)
        np.testing.assert_array_equal(bidx_sorted, exp_bidx)
        np.testing.assert_array_equal(n_per_bin, exp_n)

        exp_bin_order = np.asarray([2, 10, 3, 6])
        np.testing.assert_array_equal(bins[bidx_sorted],
                                      exp_bin_order)

        exp_n_order = np.asarray([1, 1, 3, 4])
        np.testing.assert_array_equal(n_per_bin[bidx_sorted],
                                      exp_n_order)

    def test_distribute_bins(self):

        # An index array to bins that sorts 
        # n_per_bin from low to high values.
        bidx_sorted = np.asarray([4,1,2,3,0], dtype=int)
        n_per_bin = np.asarray([10, 7, 8, 9, 6])

        size = 1
        bidx_per_rank = tools.distribute_bins(bidx_sorted, n_per_bin,
                                              size=size)
        self.assertTrue(type(bidx_per_rank) == list)
        self.assertTrue(len(bidx_per_rank) == size)

        exp_arr = bidx_sorted[::-1] # I.e. largest first.
        np.testing.assert_array_equal(bidx_per_rank[0], exp_arr)

    def test_distribute_bins2(self):

        bidx_sorted = np.asarray([4,1,2,3,0], dtype=int)
        n_per_bin = np.asarray([10, 7, 8, 9, 6])

        size = 2
        bidx_per_rank = tools.distribute_bins(bidx_sorted, n_per_bin,
                                              size=size)
        self.assertTrue(type(bidx_per_rank) == list)
        self.assertTrue(len(bidx_per_rank) == size)

        exp_arr = np.asarray([0, 2, 4]) # Rank 0.
        np.testing.assert_array_equal(bidx_per_rank[0], exp_arr)

        exp_arr = np.asarray([3, 1]) # Rank 1.
        np.testing.assert_array_equal(bidx_per_rank[1], exp_arr)

    def test_distribute_bins3(self):

        bidx_sorted = np.asarray([4,1,2,3,0], dtype=int)
        n_per_bin = np.asarray([10, 7, 8, 9, 6])

        size = 3
        bidx_per_rank = tools.distribute_bins(bidx_sorted, n_per_bin,
                                              size=size)
        self.assertTrue(type(bidx_per_rank) == list)
        self.assertTrue(len(bidx_per_rank) == size)

        exp_arr = np.asarray([0, 1]) # Rank 0.
        np.testing.assert_array_equal(bidx_per_rank[0], exp_arr)

        exp_arr = np.asarray([3, 4]) # Rank 1.
        np.testing.assert_array_equal(bidx_per_rank[1], exp_arr)

        exp_arr = np.asarray([2]) # Rank 2.
        np.testing.assert_array_equal(bidx_per_rank[2], exp_arr)

    def test_distribute_bins4(self):

        bidx_sorted = np.asarray([4,1,2,3,0], dtype=int)
        n_per_bin = np.asarray([10, 7, 8, 9, 6])

        size = 6
        bidx_per_rank = tools.distribute_bins(bidx_sorted, n_per_bin,
                                              size=size)
        self.assertTrue(type(bidx_per_rank) == list)
        self.assertTrue(len(bidx_per_rank) == size)

        exp_arr = np.asarray([0]) # Rank 0.
        np.testing.assert_array_equal(bidx_per_rank[0], exp_arr)

        exp_arr = np.asarray([3]) # Rank 1.
        np.testing.assert_array_equal(bidx_per_rank[1], exp_arr)

        exp_arr = np.asarray([2]) # Rank 2.
        np.testing.assert_array_equal(bidx_per_rank[2], exp_arr)

        exp_arr = np.asarray([1]) # Rank 3.
        np.testing.assert_array_equal(bidx_per_rank[3], exp_arr)

        exp_arr = np.asarray([4]) # Rank 4.
        np.testing.assert_array_equal(bidx_per_rank[4], exp_arr)

        exp_arr = np.asarray([]) # Rank 5.
        np.testing.assert_array_equal(bidx_per_rank[5], exp_arr)


    def test_distribute_bins5(self):

        bidx_sorted = np.asarray([4,1,2,3,0], dtype=int)
        n_per_bin = np.asarray([100, 7, 8, 9, 6])

        size = 6
        bidx_per_rank = tools.distribute_bins(bidx_sorted, n_per_bin,
                                              size=size)
        self.assertTrue(type(bidx_per_rank) == list)
        self.assertTrue(len(bidx_per_rank) == size)

        exp_arr = np.asarray([0]) # Rank 0.
        np.testing.assert_array_equal(bidx_per_rank[0], exp_arr)

        exp_arr = np.asarray([3, 4, 1, 2]) # Rank 1.
        np.testing.assert_array_equal(bidx_per_rank[1], exp_arr)

        exp_arr = np.asarray([]) # Rank 2.
        np.testing.assert_array_equal(bidx_per_rank[2], exp_arr)

        exp_arr = np.asarray([]) # Rank 3.
        np.testing.assert_array_equal(bidx_per_rank[3], exp_arr)

        exp_arr = np.asarray([]) # Rank 4.
        np.testing.assert_array_equal(bidx_per_rank[4], exp_arr)

        exp_arr = np.asarray([]) # Rank 5.
        np.testing.assert_array_equal(bidx_per_rank[5], exp_arr)



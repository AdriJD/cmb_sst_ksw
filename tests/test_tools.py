import unittest
import numpy as np
import os
from sst import tools
from scipy.interpolate import griddata


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

    def test_distribute_bins_simple(self):
        
        size = 3
        # An index array to bins that sorts 
        # n_per_bin from low to high values.
        bidx_sorted = np.asarray([4,1,2,3,0], dtype=int)
        n_per_bin = np.asarray([10, 7, 8, 9, 6])
        bidx_per_rank = tools.distribute_bins_simple(
            bidx_sorted, n_per_bin, size=size)

        exp_rank0 = np.asarray([0, 3]) # 19
        exp_rank1 = np.asarray([2]) # 8
        exp_rank2 = np.asarray([1, 4]) # 13
        np.testing.assert_array_equal(bidx_per_rank[0], exp_rank0)
        np.testing.assert_array_equal(bidx_per_rank[1], exp_rank1)
        np.testing.assert_array_equal(bidx_per_rank[2], exp_rank2)

    def test_distribute_bins_simple2(self):
        
        size = 6
        bidx_sorted = np.asarray([4,1,2,3,0], dtype=int)
        n_per_bin = np.asarray([10, 7, 8, 9, 6])
        bidx_per_rank = tools.distribute_bins_simple(
            bidx_sorted, n_per_bin, size=size)

        exp_rank0 = np.asarray([0]) # 10
        exp_rank1 = np.asarray([3]) # 9
        exp_rank2 = np.asarray([2]) # 8
        exp_rank3 = np.asarray([]) # -
        exp_rank4 = np.asarray([1]) # 7
        exp_rank5 = np.asarray([4]) # 6

        np.testing.assert_array_equal(bidx_per_rank[0], exp_rank0)
        np.testing.assert_array_equal(bidx_per_rank[1], exp_rank1)
        np.testing.assert_array_equal(bidx_per_rank[2], exp_rank2)
        np.testing.assert_array_equal(bidx_per_rank[3], exp_rank3)
        np.testing.assert_array_equal(bidx_per_rank[4], exp_rank4)
        np.testing.assert_array_equal(bidx_per_rank[5], exp_rank5)

    def test_distribute_bins_simple3(self):
        
        size = 3
        n_per_bin = np.asarray([1, 1, 1, 10, 10, 10, 10, 10, 10, 25, 30, 30])
        bidx_sorted = np.arange(n_per_bin.size)
        bidx_per_rank = tools.distribute_bins_simple(
            bidx_sorted, n_per_bin, size=size)

        exp_rank0 = np.asarray([11, 10]) # 60
        exp_rank1 = np.asarray([9, 8, 7]) # 45
        exp_rank2 = np.asarray([6, 5, 4, 3, 2, 1, 0]) # 53
        np.testing.assert_array_equal(bidx_per_rank[0], exp_rank0)
        np.testing.assert_array_equal(bidx_per_rank[1], exp_rank1)
        np.testing.assert_array_equal(bidx_per_rank[2], exp_rank2)

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

    def test_get_good_triplets(self):

        bmin = 2
        bmax = 4
        lmax = 5
        pmod = 1

        # With these options I expect only the following triplets.
        exp_ans = np.asarray([[2, 2, 3], [2, 3, 4], [2, 4, 5], [3, 3, 3],
                              [3, 3, 5], [3, 4, 4], [3, 5, 5], [4, 4, 5]])
        good_triplets = np.zeros_like(exp_ans)

        tools.get_good_triplets(bmin, bmax, lmax, good_triplets, pmod)

        np.testing.assert_array_equal(exp_ans, good_triplets)

        ### 
        pmod = 0
        # With these options I expect only the following triplets.

        exp_ans2 = np.asarray([[2, 2, 2], [2, 2, 4], [2, 3, 3], [2, 3, 5],
                               [2, 4, 4], [2, 5, 5], [3, 3, 4], [3, 4, 5],
                               [4, 4, 4], [4, 5, 5]])
        good_triplets = np.zeros_like(exp_ans2)

        tools.get_good_triplets(bmin, bmax, lmax, good_triplets, pmod)

        np.testing.assert_array_equal(exp_ans2, good_triplets)
        
        ###
        pmod = 2
        # With these options I expect only the following triplets.
        exp_ans3 = np.asarray([[2, 2, 2], [2, 2, 3], [2, 2, 4], [2, 3, 3],
                               [2, 3, 4], [2, 3, 5], [2, 4, 4], [2, 4, 5],
                               [2, 5, 5], [3, 3, 3], [3, 3, 4], [3, 3, 5],
                               [3, 4, 4], [3, 4, 5], [3, 5, 5], [4, 4, 4],
                               [4, 4, 5], [4, 5, 5]])

        good_triplets = np.zeros_like(exp_ans3)

        tools.get_good_triplets(bmin, bmax, lmax, good_triplets, pmod)

        np.testing.assert_array_equal(exp_ans3, good_triplets)

    def test_get_good_triplets_err(self):

        bmin = 2
        bmax = 4
        lmax = 5
        pmod = 1

        # With these options I expect only the following triplets.
        # ([[2, 2, 3], [2, 3, 4], [2, 4, 5], [3, 3, 3],
        #   [3, 3, 5], [3, 4, 4], [3, 5, 5], [4, 4, 5]])
        # so good_triplets should have shape (8, 3).

        # Give a too large array, should crash.
        good_triplets = np.zeros((9, 3), dtype=int)

        with self.assertRaises(ValueError):
            tools.get_good_triplets(bmin, bmax, lmax, good_triplets, pmod)

        # Give too small array, should crash.
        good_triplets = np.zeros((7, 3), dtype=int)

        with self.assertRaises(ValueError):
            tools.get_good_triplets(bmin, bmax, lmax, good_triplets, pmod)

    def test_interpolate(self):
        
        # 3d box
        points = np.ones((8, 3), dtype=float)
        points[0] = (0, 0, 0)
        points[1] = (1, 0, 0)
        points[2] = (1, 1, 0)
        points[3] = (0, 1, 0)
        points[4] = (0, 0, 1)
        points[5] = (1, 0, 1)
        points[6] = (1, 1, 1)
        points[7] = (0, 1, 1)

        values = np.zeros(8, dtype=float)
        values[4:] = 1.

        # Three points.
        xi = np.ones((3, 3))
        xi[0] = (0.5, 0.5, 0.5)
        xi[1] = (0.5, 0.5, 0.)
        xi[2] = (0.5, 0.5, -5.)

        vertices, weights = tools.get_interp_weights(points, xi)
        
        ans = tools.interpolate(values, vertices, weights)
        exp_ans = np.asarray([0.5, 0., np.nan])
        np.testing.assert_array_equal(exp_ans, ans)

    def test_interpolate2(self):

        # Test if my function agrees with scipy griddata.
        points = np.random.randn(30).reshape(10, 3)
        values = np.random.randn(10)
        xi = np.random.randn(15).reshape(5, 3)

        vertices, weights = tools.get_interp_weights(points, xi)        
        ans = tools.interpolate(values, vertices, weights)

        exp_ans = griddata(points, values, xi, method='linear')        
        np.testing.assert_array_almost_equal(exp_ans, ans)

    def test_interpolate3(self):
        
        # 3d box
        points = np.ones((8, 3), dtype=float)
        points[0] = (0, 0, 0)
        points[1] = (1, 0, 0)
        points[2] = (1, 1, 0)
        points[3] = (0, 1, 0)
        points[4] = (0, 0, 1)
        points[5] = (1, 0, 1)
        points[6] = (1, 1, 1)
        points[7] = (0, 1, 1)

        values = np.zeros(8, dtype=float)
        values[4:] = 1.

        # Three points.
        xi = np.ones((3, 3))
        xi[0] = (0.5, 0.5, 0.5)
        xi[1] = (0.5, 0.5, 0.)
        xi[2] = (0.5, 0.5, -5.)

        vertices, weights = tools.get_interp_weights(points, xi)
        
        ans = tools.interpolate(values, vertices, weights)
        exp_ans = np.asarray([0.5, 0., np.nan])
        np.testing.assert_array_equal(exp_ans, ans)

        # New values.
        values = np.zeros(8, dtype=float)
        values[4:] = 2.

        ans = tools.interpolate(values, vertices, weights)
        exp_ans = np.asarray([1, 0., np.nan])
        np.testing.assert_array_equal(exp_ans, ans)

    def test_interpolate4(self):

        # Testing output shapes
        points = np.random.randn(30).reshape(10, 3)
        values = np.random.randn(10)
        xi = np.random.randn(15).reshape(5, 3)

        vertices, weights = tools.get_interp_weights(points, xi)        
        
        exp_shape = (5, 4) # i.e. 4 numbers per xi point.
        self.assertEqual(weights.shape, exp_shape)
        self.assertEqual(vertices.shape, exp_shape)

        exp_shape = (5,) # i.e. 4 numbers per xi point.
        ans = tools.interpolate(values, vertices, weights).shape
        self.assertEqual(ans, exp_shape)

    def test_interpolate5(self):

        # Test if values on input points are unchanged.
        points = np.random.randn(30).reshape(10, 3)
        values = np.random.randn(10)
        xi = points

        vertices, weights = tools.get_interp_weights(points, xi)        
        
        ans = tools.interpolate(values, vertices, weights)
        
        np.testing.assert_array_almost_equal(ans, values, decimal=7)
        
    def test_has_nan(self):
        
        a = np.arange(30, dtype=float)
        self.assertFalse(tools.has_nan(a))
        
        a[10] = np.nan
        
        self.assertTrue(tools.has_nan(a))

    def test_one_over_delta(self):
        
        self.assertTrue(tools.one_over_delta(1, 2, 3) == 1.)
        self.assertTrue(tools.one_over_delta(2, 2, 2) == 1/6.)
        self.assertTrue(tools.one_over_delta(1, 1, 2) == 0.5)

    def test_contract_bcb(self):
        
        B = np.arange(12, dtype=float)
        C = np.arange(144, dtype=float).reshape(12, 12)

        ans = tools.contract_bcb(B, C)
        exp_ans = np.einsum("i,ij,j", B, C, B)

        self.assertEqual(ans, exp_ans)

    def test_fisher_loop(self):
        
        bispec = np.ones((1, 2))
        triplets = np.asarray([[2, 2, 3]])
        lmin = 2
        ic1 = np.ones((2, 2, 2))
        ic2 = np.ones((2, 2, 2))
        ic3 = np.ones((2, 2, 2))

        f = tools._fisher_loop(bispec, triplets, ic1, ic2, ic3, lmin)
        exp_ans = 2
        self.assertEqual(exp_ans, f)

    def test_fisher_loop2(self):
        
        bispec = np.ones((4, 2))
        triplets = np.asarray([[2, 2, 3], [2, 3, 4], [2, 4, 5], [3, 3, 3]])
        lmin = 2
        ic1 = np.ones((4, 2, 2))
        ic2 = np.ones((4, 2, 2))
        ic3 = np.ones((4, 2, 2))

        f = tools._fisher_loop(bispec, triplets, ic1, ic2, ic3, lmin)
        exp_ans = 10.666666666666666
        self.assertEqual(exp_ans, f)

    def test_fisher_loop3(self):
        
        bispec = np.ones((4, 2))
        triplets = np.asarray([[2, 2, 3], [2, 3, 4], [2, 4, 5], [3, 3, 3]])
        lmin = 2
        ic1 = np.ones((4, 2, 2))
        ic2 = np.ones((4, 2, 2))
        ic3 = np.ones((4, 2, 2))

        ic1 *= 2

        f = tools._fisher_loop(bispec, triplets, ic1, ic2, ic3, lmin)
        exp_ans = 21.333333333333333
        self.assertEqual(exp_ans, f)

    def test_fisher_loop4(self):
        
        bispec = np.ones((1, 2))
        triplets = np.asarray([[2, 3, 4]])
        lmin = 2
        ic1 = np.ones((4, 2, 2))
        ic2 = np.ones((4, 2, 2))
        ic3 = np.ones((4, 2, 2))

        ic1 *= 2
        ic3 *= 2

        f = tools._fisher_loop(bispec, triplets, ic1, ic2, ic3, lmin)
        exp_ans = 16
        self.assertEqual(exp_ans, f)

    def test_fisher_loop5(self):
        
        bispec = np.ones((1, 2))
        triplets = np.asarray([[2, 3, 4]])
        lmin = 2
        ic1 = np.ones((4, 2, 2))
        ic2 = np.ones((4, 2, 2))
        ic3 = np.ones((4, 2, 2))

        ic1 *= 2
        ic2 *= 2
        ic3 *= 2

        f = tools._fisher_loop(bispec, triplets, ic1, ic2, ic3, lmin)
        exp_ans = 32
        self.assertEqual(exp_ans, f)

    def test_fisher_loop6(self):
        
        bispec = np.ones((2, 2))
        triplets = np.asarray([[2, 2, 2], [2, 3, 4]])
        lmin = 2
        ic1 = np.ones((4, 2, 2))
        ic2 = np.ones((4, 2, 2))
        ic3 = np.ones((4, 2, 2))

        ic1 *= 2
        ic2 *= 2
        ic3 *= 2

        f = tools._fisher_loop(bispec, triplets, ic1, ic2, ic3, lmin)
        exp_ans = 37.333333333333333
        self.assertEqual(exp_ans, f)

    def test_get_slice(self):

        # get_slice is supposed to give slice to one bin before 
        # and after provided bin index.

        num_bins = 10
        bidx = 0
        b_start, b_stop = tools.get_slice(bidx, num_bins)
        self.assertEqual(b_start, 0)
        self.assertEqual(b_stop, 2)

        bidx = 1
        b_start, b_stop = tools.get_slice(bidx, num_bins)
        self.assertEqual(b_start, 0)
        self.assertEqual(b_stop, 3)

        # Note that stop index can be at most size of array.
        bidx = num_bins - 1
        b_start, b_stop = tools.get_slice(bidx, num_bins)
        self.assertEqual(b_start, num_bins-2)
        self.assertEqual(b_stop, num_bins)

        with self.assertRaises(ValueError):
            num_bins = -1
            bidx = 1
            b_start, b_stop = tools.get_slice(bidx, num_bins)

        with self.assertRaises(ValueError):
            num_bins = 10
            bidx = 10
            b_start, b_stop = tools.get_slice(bidx, num_bins)

        with self.assertRaises(ValueError):
            num_bins = 0
            bidx = 10
            b_start, b_stop = tools.get_slice(bidx, num_bins)

        with self.assertRaises(ValueError):
            num_bins = -1
            bidx = 10
            b_start, b_stop = tools.get_slice(bidx, num_bins)

    def test_ell2bidx(self):
        
        bins = np.asarray([2, 3, 4, 5, 10, 20])
        
        ell = 2
        self.assertEqual(tools.ell2bidx(ell, bins), 0)

        ell = 4
        self.assertEqual(tools.ell2bidx(ell, bins), 2)

        ell = 6
        self.assertEqual(tools.ell2bidx(ell, bins), 3)

        ell = 20
        self.assertEqual(tools.ell2bidx(ell, bins), 5)

        ell = 21
        with self.assertRaises(ValueError):
            tools.ell2bidx(ell, bins)

        ell = 0
        with self.assertRaises(ValueError):
            tools.ell2bidx(ell, bins)


    

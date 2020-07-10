'''
Calculate Fisher information and all required quantities for a
tensor-scalar-scalar bispectrum.
'''

from __future__ import print_function
import sys
import os
import numpy as np
import pickle
import warnings
import cProfile

from scipy.special import spherical_jn
from scipy.integrate import trapz
from scipy.linalg import inv
from scipy.interpolate import CubicSpline
from scipy.interpolate import griddata
from scipy.signal import convolve
from scipy.spatial import qhull

from sst import camb_tools as ct
from sst import tools
from .mpibase import MPIBase
import pywigxjpf as wig

opj = os.path.join

try:
    xrange
except NameError:
    xrange = range

__all__ = ['PreCalc', 'Template', 'Fisher']

class PreCalc(MPIBase):

    def __init__(self, **kwargs):
        '''

        Keyword arguments
        -----------------
        kwargs : {MPIBase_opts}
        '''

        # Precomputed quantities go in dictionaries to not pollute
        # the namespace too much.
        self.cosmo = {} 
        self.bins = {}
        self.beta = {}
        self.bispec = {}

        self.cosmo['init_cosmo'] = False
        self.bins['init_bins'] = False
        self.beta['init_beta'] = False

        # init MPIBase
        super(PreCalc, self).__init__(**kwargs)

        # Initialize wigner tables for lmax=8000.
        wig.wig_table_init(2 * 8000, 9)
        wig.wig_temp_init(2 * 8000)

    def get_camb_output(self, camb_out_dir='.', high_ell=False,
                        interp_factor=None, **kwargs):
        '''
        Store CAMB ouput (transfer functions, Cls etc.) and store
        in internal dictionaries.

        Keyword Arguments
        -----------------
        camb_out_dir : str
            Path to folder containing CAMB output.
            (default : '.')
        high_ell : boool
            If True, load up additional transfer functions for high
            ell (>4000) that use sparse samples in ell. Is combined
            into single transfer function.
        interp_factor : int, None
            Factor of extra point in k-dimension of tranfer function
            calculated by cubic interpolation.
        kwargs : {get_spectra_opts}

        Notes
        -----
        Transfer functions and Cls are not independent,
        so makes sense to load them together like this.
        '''
        source_dir = camb_out_dir

        lmax = None
        cls = None
        tr = None
        k = None
        ells = None

        # load spectra
        if self.mpi_rank == 0:

            cls, lmax = ct.get_spectra(source_dir, **kwargs)

        cls = self.broadcast_array(cls)
        lmax = self.broadcast(lmax)

        self.cosmo['cls'] = cls
        self.cosmo['cls_lmax'] = lmax

        # load transfer functions, k-arrays and lmax
        for ttype in ['scalar', 'tensor']:

            if self.mpi_rank == 0:
                tr, lmax, k, ells = ct.read_camb_output(source_dir,
                                                  ttype=ttype)

                if high_ell:
                    tr_hl, lmax_hl, k_hl, ells_hl = ct.read_camb_output(
                        source_dir, ttype=ttype, high_ell=True)

                    # disable this for tensors
                    if ttype == 'tensor':
                        tr_hl *= 0.

                    # remove low ell from tr_hl
                    # remember, tr has shape (num_s, ells, ks)
                    lidx_min_hl = np.where(ells_hl > lmax)[0][0]

                    # create ells that combines low and high
                    ells = np.concatenate((np.arange(2, lmax+1),
                                           ells_hl[lidx_min_hl:]))

                    # index where sparse part starts in full ells array
                    lidx_min = np.where(ells > lmax)[0][0]
                    assert ells[lidx_min] == ells_hl[lidx_min_hl]

                    # allocate combined transfer
                    num_s = tr_hl.shape[0]
                    tr_full = np.zeros((num_s,ells.size,k.size))

                    # fill with old transfer
                    tr_full[:,:lmax-1,:] = tr

                    # interpolate high ell transfer to low ell ks
                    print('interpolating high_ell transfer...')
                    for nsidx in xrange(num_s):

                        # only loop over sparse part of ells
                        for lidx, ell in enumerate(ells[lidx_min:]):

                            lidx_hl = lidx + lidx_min_hl # index to ells_hl
                            lidx_f = lidx + lidx_min # index to ells
                            assert ells[lidx_f] == ells_hl[lidx_hl]

                            # cubic interpolation over k samples
                            cs = CubicSpline(k_hl, tr_hl[nsidx,lidx_hl,:])
                            tr_full[nsidx,lidx_f,:] = cs(k)

                    tr = tr_full
                    lmax = lmax_hl
                    print('...done')

            tr = self.broadcast_array(tr)
            k = self.broadcast_array(k)
            lmax = self.broadcast(lmax)
            ells = self.broadcast_array(ells)

            if interp_factor is not None:
                if interp_factor % 1 != 0:
                    raise ValueError("interp_factor has to be integer.")

                if self.mpi_rank == 0:
                    print("Interpolating {} transfer function by factor "
                          "{} ...".format(ttype, interp_factor))

                # Determine new k array.
                k_interp = np.zeros(k.size * interp_factor)
                for i in xrange(interp_factor):
                    k_interp[i::interp_factor] = k
                conv_window = np.ones(interp_factor) / float(interp_factor)
                k_interp = convolve(k_interp, conv_window, mode='valid')

                # Allocate new transfer function.
                num_s = tr.shape[0]
                num_ells = tr.shape[1]
                tr_interp = np.zeros((num_s,num_ells,k_interp.size))

                for nsidx in xrange(num_s):
                    for lidx in xrange(num_ells):

                        if self.mpi_rank == 0:
                            print("num_sources: {}/{}, ell idx: {:04d} \r".format(
                                nsidx+1, num_s, lidx), end='\r')

                        cs = CubicSpline(k, tr[nsidx,lidx,:])
                        tr_interp[nsidx,lidx,:] = cs(k_interp)

                tr = tr_interp
                k = k_interp

                if self.mpi_rank == 0:
                    print("...done")

            self.cosmo[ttype] = {'transfer' : tr,
                                'lmax' : lmax,
                                'k' : k,
                                'ells_camb' : ells}

        # make sure tensor and scalar ks are equal (not
        # guaranteed by CAMB). Needed b/c both T and S
        # transfer functions are integrated over same k
        # when calculating beta
        ks = self.cosmo['scalar']['k']
        kt = self.cosmo['tensor']['k']

        np.testing.assert_allclose(ks, kt)

    def init_cosmo(self, lmax=5200, verbose=True, **kwargs):
        '''
        Run CAMB to get transfer functions, Cls etc. Stores
        output in internal `cosmo` dictionary.

        Arguments
        ---------
        lmax : int
        
        Keyword Arguments
        -----------------
        verbose : bool
        kwargs : {camb_tools.run_camb_opts}

        Notes
        -----
        Populates following keys in internal `cosmo' dictionary on 
        all ranks:
            transfer : dict
            cls : dict
            opts : dict  
            init_cosmo : bool
        '''

        if verbose is True:
            # Also print things in run_camb.
            kwargs.update({'verbose' : True})

        transfer = None
        cls = None
        opts = None

        # Only run CAMB on root.
        if self.mpi_rank == 0:

            transfer, cls, opts = ct.run_camb(lmax, **kwargs)

            if self.mpi and verbose:
                print('Broadcasting...')

        transfer = self.broadcast(transfer)
        cls = self.broadcast(cls)
        opts = self.broadcast(opts)

        if self.mpi == 0 and verbose:
            print('...done')
        
        self.cosmo['transfer'] = transfer
        self.cosmo['cls'] = cls
        self.cosmo['opts'] = opts
        self.cosmo['init_cosmo'] = True
        
        return

    def get_noise_curves(self, cross_noise=False, **kwargs):
        '''
        Load SO TT and pol noise curves and process
        into TT, EE, BB, TE, TB and EB noise curves

        Keyword Arguments
        ---------
        cross_noise : bool
            If set, TE, TB and EB elements get noise contribution
            (sqrt(nl_XX nl_YY) (default : False)
        tt_file : str
            Path to TT file
        pol_file : str
            Path to EE, BB file
        '''

        nls = None
        lmin = None
        lmax = None

        if self.mpi_rank == 0:

            if kwargs.get('sat_file'):

                ell_tt, nl_tt, ell_pol, nl_ee, nl_bb, nl_sat, ell_sat = ct.get_so_noise(
                    **kwargs)

            ell_tt, nl_tt, ell_pol, nl_ee, nl_bb = ct.get_so_noise(
                                                           **kwargs)

            # Make lmin and lmax equal
            lmin_tt = ell_tt[0]
            lmin_pol = ell_pol[0]
            lmax_tt = ell_tt[-1]
            lmax_pol = ell_pol[-1]

            lmin = max(lmin_tt, lmin_pol)
            lmax = min(lmax_tt, lmax_pol)

            # pol has lower lmin and higher lmax
            nl_ee = nl_ee[ell_pol >= lmin]
            nl_bb = nl_bb[ell_pol >= lmin]
            ell_pol = ell_pol[ell_pol >= lmin]

            nl_ee = nl_ee[ell_pol <= lmax]
            nl_bb = nl_bb[ell_pol <= lmax]
            ell_pol = ell_pol[ell_pol <= lmax]

            # Compute noise spectra: TT, EE, BB, TE, TB, EB
            nls = np.ones((6, ell_pol.size))
            nls[0] = nl_tt
            nls[1] = nl_ee
            nls[2] = nl_bb

            if cross_noise:
                nls[3] = np.sqrt(nl_tt * nl_ee)
                nls[4] = np.sqrt(nl_tt * nl_bb)
                nls[5] = np.sqrt(nl_ee * nl_bb)
            else:
                nls[3] = np.zeros_like(nl_bb)
                nls[4] = np.zeros_like(nl_bb)
                nls[5] = np.zeros_like(nl_bb)

        nls = self.broadcast_array(nls)
        lmin = self.broadcast(lmin)
        lmax = self.broadcast(lmax)

        self.cosmo['nls'] = nls
        self.cosmo['nls_lmin'] = int(lmin)
        self.cosmo['nls_lmax'] = int(lmax)

    def get_default_radii(self):
        '''
        Get the radii (in Mpc) used in table 1 of liquori 2007
        '''

        low = np.linspace(0, 9377, num=98, dtype=float, endpoint=False)
        re1 = np.linspace(9377, 10007, num=18, dtype=float, endpoint=False)
        re2 = np.linspace(10007, 12632, num=25, dtype=float, endpoint=False)
        rec = np.linspace(12632, 13682, num=300, dtype=float, endpoint=False)

        radii = np.concatenate((low, re1, re2, rec))

        return radii

    def get_updated_radii(self):
        '''
        Get the radii (in Mpc) that are more suitable for the sst case.
        '''

        low = np.linspace(0, 9377, num=98, dtype=float, endpoint=False)
        re1 = np.linspace(9377, 10007, num=18, dtype=float, endpoint=False)
        re2 = np.linspace(10007, 12632, num=25, dtype=float, endpoint=False)
        rec = np.linspace(12632, 13682, num=50, dtype=float, endpoint=False)
        rec_new = np.linspace(13682, 15500, num=300, dtype=float, endpoint=False)
        rec_extra = np.linspace(15500, 18000, num=10, dtype=float, endpoint=False)

        radii = np.concatenate((low, re1, re2, rec, rec_new, rec_extra))

        return radii

    def get_default_bins(self):
        '''
        Return default bins.

        Returns
        -------
        bins : array-like
            Left/lower side of bins from ell=2 to ell=1e4.
        '''
        f = 1
        bins_0 = np.arange(2, 51, 1)
        bins_1a = np.arange(54, 204, int(4 * f))
        bins_1b = np.arange(212, 512, int(12 * f))
        bins_2 = np.arange(524, 2024, int(24 * f))
        bins_3 = np.arange(2040, 10000, int(40 * f))

        bins = np.concatenate((bins_0, bins_1a, bins_1b,
                               bins_2, bins_3))

        return bins

    def get_default_beta_bins(self):
        '''
        Return multipole bins optimized for calculating
        beta (left/lower edges).

        Returns
        -------
        bins : array-like
            Left/lower side of bins from ell=2 to ell=1e4.
        '''

        # For now, use same bins as bispectrum.
        return self.get_default_bins()

    def _scatter_bins(self):
        ''' 
        Scatter bins and make every rank know
        bins on other ranks. 
        '''

        bins = self.bins['bins']
        idx = np.arange(bins.size)        

        idxs_on_rank = {}
        bins_on_rank = {}

        for rank in xrange(self.mpi_size):
            idxs_on_rank[str(rank)] = idx[rank::self.mpi_size]
            bins_on_rank[str(rank)] = bins[rank::self.mpi_size]

        self.bins['bins_on_rank'] = bins_on_rank
        self.bins['idxs_on_rank'] = idxs_on_rank
        
    def get_bins_on_rank(self, return_idx=False, rank=None):
        '''
        Return bins on rank.
        
        Keyword arguments
        -----------------
        return_idx : bool
            Also return indices to full-sized bins array
        rank : int, None
            If None use rank from which function is called.
            (default : None)

        Returns
        -------
        bins_on_rank : array_like
            Subset of bins, unique per rank
        idxs_on_rank : array_like
            If return_idx is set. Indices of local bins
            in full-sized bins array.       
        '''
        if rank is None:
            rank = self.mpi_rank

        bins_on_rank = self.bins['bins_on_rank'][str(rank)]

        if return_idx is False:
            return bins_on_rank
        else:
            idxs_on_rank = self.bins['idxs_on_rank'][str(rank)]
            return bins_on_rank, idxs_on_rank

    def init_bins(self, lmin=None, lmax=None, bins=None,
                  parity='odd', verbose=False):
        '''
        Create default bins in ell-space. Stores attributes for
        unique_ells, bins, num_pass and first_pass

        Keyword Arguments
        -----------------
        lmin : int
            Value of first bin
        lmax : int
             Create bins up to (but not including lmax)
        bins : array-like
            Lower value of bins in ell. Must be monotonically increasing.
            No repeated values allowed. May be truncated depending on lmin
            lmax. if lmin and lmax are given, bins must encompass them.
        parity : str
            Consider tuples in ell that are parity "odd"
            or "even". If None, use both. (default : "odd")
        verbose : bool, int
            If True or integer, print stuff.

        Notes
        -----
        Sum over ell is l1 <= l2 <= l3

        Stores following internal attributes

        lmin : int
        lmax : int
        ells : array-like
            Unbinned ell values from lmin to lmax
        bins : array-like
            Left (lower) side of bins used in rest of code
        num_pass : array-like
            number of tuples per 3d bin that pass triangle
            condition. Shape : (nbins, nbins, nbins).
        first_pass : array-like
            Lowest sum tuple per 3d bin that passes triangle
            condition. Shape : (nbins, nbins, nbins, 3).
        unique_ells : array-like
            Values of ell that appear in first_pass. Used to only
            precompute Wigner 3j's for values that are used.
        '''

        # Store parity. NOTE, really even or odd ell sum 
        self.bins['parity'] = parity

        # If needed, extract lmin, lmax from provided bins
        if bins is not None:
            bins = np.array(bins)
            if not lmin:
                lmin = bins[0]
            elif lmin < bins[0]:
                raise ValueError("lmin < mininum bin")
            if not lmax:
                lmax = bins[-1]

        elif not (lmin and lmax):
            raise ValueError('Provide lmin and lmax if bins=None')

        if lmax <= lmin:
            raise ValueError('lmax <= lmin')

        ells = np.arange(lmin, lmax+1)
        self.bins['ells'] = ells

        if bins is None:
            bins = self.get_default_bins()

        try:
            # find first occurences
            min_bidx = np.where(bins >= lmin)[0][0]
            max_bidx = np.where(bins > lmax)[0][0]
        except IndexError:
            if lmax == bins[-1]:
                max_bidx = None
            else:
                raise ValueError('lmin or lmax is larger than max bin')

        # Note: we include bin that contains lmax, num_pass takes care of
        # partial bin.
        bins = bins[min_bidx:max_bidx]

        # check for monotonic order and no repeating values.
        np.testing.assert_array_equal(np.unique(bins), bins)

        self.bins['bins'] = bins
        self.bins['lmax'] = bins[-1]
        self.bins['lmin'] = bins[0]

        idx = np.arange(bins.size) # indices to bins
        num_bins = bins.size

        if parity == 'odd':
            pmod = 1
        elif parity == 'even':
            pmod = 0
        else:
            pmod = None

        if self.mpi_rank == 0 and verbose:
            print('initializing bins...')

        self._scatter_bins()
        bins_on_rank, idxs_on_rank = self.get_bins_on_rank(return_idx=True)

        # Allocate these arrays only for bins in rank.
        # allocate space for number of good tuples per 3d bin
        num_pass = np.zeros((bins_on_rank.size, num_bins, num_bins),
                            dtype=int)
        # allocate space for first good tuple per 3d bin
        first_pass = np.zeros((bins_on_rank.size, num_bins, num_bins, 3),
                              dtype=int)

        if pmod is None:
            pmod = 2 # i.e. both even and odd sum(l1, l2, l3).
        tools.init_bins_jit(bins, idxs_on_rank, num_pass, first_pass, pmod)

        # Combine num_pass and first_pass on root
        if self.mpi:
            self.barrier()

            if self.mpi_rank == 0 and verbose:
                print('...done, gathering bins from all ranks...')

            # Allocate space to receive arrays on root.
            if self.mpi_rank == 0:
                num_pass_rec = np.empty((num_bins, num_bins, num_bins),
                                        dtype=int)
                first_pass_rec = np.empty((num_bins, num_bins, num_bins, 3),
                                          dtype=int)

                # Fill with arrays on root.
                _, idxs_on_root = self.get_bins_on_rank(
                    return_idx=True)
                num_pass_rec[idxs_on_root] = num_pass
                first_pass_rec[idxs_on_root] = first_pass

            # loop over all non-root ranks
            for rank in xrange(1, self.mpi_size):

                # Send arrays to root.
                if self.mpi_rank == rank:
                    self._comm.Send(num_pass, dest=0, tag=rank)
                    self._comm.Send(first_pass, dest=0, tag=rank + self.mpi_size)

                    if verbose == 2:
                        print(self.mpi_rank, 'sent')

                # Receive the arrays on root.
                if self.mpi_rank == 0:

                    _, idxs_on_rank = self.get_bins_on_rank(return_idx=True,
                                                            rank=rank)
                    num_bins_on_rank = idxs_on_rank.size
                    # MPI needs contiguous array to receive.
                    num_pass_rec_cont = np.empty((num_bins_on_rank, num_bins,
                                                  num_bins), dtype=int)
                    first_pass_rec_cont = np.empty((num_bins_on_rank, num_bins,
                                                    num_bins, 3), dtype=int)

                    self._comm.Recv(num_pass_rec_cont,
                                    source=rank, tag=rank)
                    self._comm.Recv(first_pass_rec_cont,
                                    source=rank, tag=rank + self.mpi_size)
                    if verbose == 2:
                        print('root received {}'.format(rank))

                    # Change to fancy indexing: num_pass_rec[[bins],...] = ..
                    num_pass_rec[idxs_on_rank] = num_pass_rec_cont
                    first_pass_rec[idxs_on_rank] = first_pass_rec_cont

        else:
            # No mpi.
            num_pass_rec = num_pass
            first_pass_rec = first_pass

        self.barrier()

        if self.mpi_rank == 0:
            self.bins['num_pass_full'] = num_pass_rec
            self.bins['first_pass_full'] = first_pass_rec
            # Trim away the zeros.
            unique_ells = np.unique(first_pass_rec)[1:]
        else:
            unique_ells = None
        
        # Each rank gets full unique_ells array for wigner comp. later.
        self.bins['unique_ells'] = self.broadcast_array(unique_ells)

        # Note that these differ per rank.
        self.bins['num_pass'] = num_pass # No. of good triplets per bin.
        self.bins['first_pass'] = first_pass # Lowest good triplet per bin.

        self.bins['init_bins'] = True
        self.bins['bins'] = bins

        if self.mpi_rank == 0 and verbose:
            print('...done')

    def init_wig3j(self):
        '''
        Precompute I^000_lL1 and I^20-2_lL2 for all unique ells
        and \Delta L \in [-2, -1, 0, 1, 2]

        Notes
        -----        
        Stored internally as wig_s and wig_t keys in bins dict.
        Arrays have shape (ells.size, (\Delta L = 5)). So full 
        ells-sized array, although only values in unique_ells 
        are calculated.

        For now, no parallel computations, it is not really 
        needed.
        '''

        u_ells = self.bins['unique_ells']
        ells = self.bins['ells']

        lmin = ells[0]

        wig_s = np.zeros((ells.size, 5))
        wig_t = np.zeros((ells.size, 5))

        for ell in u_ells:
            # ell is unbinned, so we can simply infer the indices
            lidx = ell - lmin

            for Lidx, DL in enumerate([-2, -1, 0, 1, 2]):

                L = ell + DL

                if DL == -1 or DL == 1:
                    # only here we need to fill the scalar factor
                    tmp = wig.wig3jj_array([2*ell, 2*L, 2,
                                      0, 0, 0])
                    tmp *= np.sqrt((2 * L + 1) * (2 * ell + 1) * 3)
                    tmp /= (2 * np.sqrt(np.pi))
                    wig_s[lidx,Lidx] = tmp

                tmp = wig.wig3jj_array([2*ell, 2*L, 4,
                                  4, 0, -4])
                tmp *= np.sqrt((2 * L + 1) * (2 * ell + 1) * 5)
                tmp /= (2 * np.sqrt(np.pi))

                wig_t[lidx,Lidx] = tmp

        self.bispec['wig_s'] = wig_s
        self.bispec['wig_t'] = wig_t

    def init_beta(self, func=None, L_range=[-2, -1, 0, 1, 2], radii=None,
             optimize=True, interp_factor=None, sparse=True, verbose=False):
        '''
        Calculate beta_l,L(r) = 2/pi * \int k^2 dk f(k) j_L(kr) T_X,l^(Z)(k)
        for provided functions of k.

        Keyword Arguments
        -----------------
        func : array-like, str
            Factor f(k) of (primordial) factorized shape function.
            Can be of shape (n, k.size). If string, choose
            "local", "equilateral", "orthogonal". If None, uses local
            (default : None)
        radii : array-like
            Array with radii to compute. In units [Mpc], if None,
            use default_radii (default : None)
        L_range : array-like, int
            Possible deviations from ell, e.g. [-2, -1, 0, 1, 2].
            Must be monotonically increasing with steps of 1.
            (default : [0])
        optimize : bool
            Do no calculate spherical bessel for kr << L (default :
            True)
        interp_factor : int, None
            Factor of extra point in k-dimension of transfer function
            calculated by cubic interpolation.
        sparse : bool
            Calculate beta over multipoles given by bins, then
            interpolate of full mulitpole range. (default : True)
        verbose : bool, int
            Print progress (default : False)
        
        Notes
        -----
        If sparse = False, beta is produced over same multipole range
        as transfer funtions.

        Populates following keys in internal beta dictionary.

        beta_s : array-like
            Scalar beta. Shape = (ell, L, n, {I, E}, radii)
        beta_t : array-like
            Tensor beta. Shape = (ell, L, n, {I, E, B}, radii)
        b_beta_s, b_beta_t : array-like
            Binned (mean per multipole bin) versions of beta_s, beta_t.
        ells : array-like
            Multipole values used. 
        L_range : array-like
            Second multipole values used.
        bins : array-like
            Left (lower) side of bins used for binned betas.
        radii : array-like
            Comoving radii used.
        func : array-like
            Function(s) of k used. Shape = (n, k)
        init_beta : bool
            Whether or not this function has been run.
        '''

        # ~80 % of time is spent on spher. bessels.

        if not self.bins['init_bins']:
            raise ValueError('bins not initialized')

        ells_transfer = self.cosmo['transfer']['ells']

        if sparse:
            # Use bins as ells.
            # Mapping from bin to lidx to access transfer correctly.
            bins = self.bins['bins']
            # This assumes bins[-1] <= ells[-1] which should always be true.
            # Use lidx = idxmap[bidx].
            idxmap = np.digitize(bins, bins=ells_transfer, right=True)
            ells = bins
            ells_out = self.bins['ells']
        else:
            # Calculate beta over all multipoles in transfer function arrays.
            ells = ells_transfer
            ells_out = self.bins['ells']

        L_range = np.asarray(L_range)
        if np.any(~(L_range == np.unique(L_range))):
            print("L_range: ", L_range)
            raise ValueError("L_range is not monotonically increasing "+
                             "with steps of 1")

        self.beta['L_range'] = L_range
        k = self.cosmo['transfer']['k']

        if interp_factor is not None and interp_factor != 1:
            if interp_factor % 1 != 0:
                raise ValueError("interp_factor has to be integer.")
            interp_factor = int(interp_factor)

            if self.mpi_rank == 0 and verbose:
                print("Interpolating beta k integral by factor "
                      "{} ...".format(interp_factor))
            # Determine new k array.
            k_interp = np.zeros(k.size * interp_factor)
            for i in xrange(interp_factor):
                k_interp[i::interp_factor] = k
            conv_window = np.ones(interp_factor) / float(interp_factor)
            k_interp = convolve(k_interp, conv_window, mode='valid')
            k_old = k
            k_old_size = k.size
            k = k_interp

        if func is None or func == 'local':
            func = self.local(k)
            fname = 'local'
        elif func == 'equilateral':
            func = self.equilateral(k)
            fname = 'equilateral'
        elif func == 'orthogonal':
            func = self.orthogonal(k)
            fname = 'orthogonal'

        self.beta['template'] = fname
        self.beta['func'] = func

        # You want to allow f to be of shape (n, k.size). n are number
        # of k arrays, i.e. 2 for local, 4 for equilateral.
        ndim = func.ndim
        if ndim == 1:
            func = func.copy()[np.newaxis,:]
        elif ndim == 2:
            func = func.copy()
        else:
            raise ValueError('dimension {} of func not supported'.format(ndim))

        if k.size != func.shape[1]:
            raise ValueError('func and k not compatible: {}, {}'.format(
                    func.shape, k.shape))

        ks = func.shape[0] # i.e. n

        if radii is None:
            radii = self.get_updated_radii()
        self.beta['radii'] = radii

        # scale func by k^2
        k2 = k**2
        func *= k2

        # allocate arrays for integral over k
        tmp_s = np.zeros_like(k)
        tmp_t = np.zeros_like(k)

        # load both scalar and tensor transfer functions
        transfer_s = self.cosmo['transfer']['scalar']
        transfer_t = self.cosmo['transfer']['tensor']

        pols_s = ['I', 'E']
        pols_t = ['I', 'E', 'B']

        # Distribute radii among cores
        # Weird split for more even load balance, as larger r is slower.
        radii_per_rank = []

        radii_sub = radii[self.mpi_rank::self.mpi_size]
        for rank in xrange(self.mpi_size):
            radii_per_rank.append(radii[rank::self.mpi_size])

        # beta scalar and tensor
        beta_s = np.zeros((ells.size,L_range.size,ks,
                           len(pols_s), radii_sub.size))
        beta_t = np.zeros((ells.size,L_range.size,ks,
                           len(pols_t), radii_sub.size))

        # allocate space for bessel functions
        jL = np.zeros((L_range.size, k.size))

        # an array with all possible L values
        ells_ext = np.arange(ells[-1] + L_range[-1] + 1)

        for ridx, radius in enumerate(radii_sub):
            kr = k * radius
            # Array that gives L -> kr idx mapping, i.e. in which kr sits ell
            kr_idx = np.digitize(ells_ext, bins=kr, right=True)
            kr_idx[kr_idx == kr.size] = kr.size - 1 # fix last element

            if interp_factor is not None and interp_factor != 1:
                interp = True
                # Same for original k array
                kr_old = k_old * radius
                kr_old_idx = np.digitize(ells_ext, bins=kr_old, right=True)
                kr_old_idx[kr_old_idx == kr_old.size] = kr_old.size - 1
            else:
                interp = False

            if self.mpi_rank == 0 and verbose:
                print('rank: {}, ridx: {}/{}, radius: {} Mpc'.format(
                    self.mpi_rank, ridx, radii_sub.size - 1, radius))

            ell_prev = ells[0] - 1

            for lidx_b, ell in enumerate(ells):

                # ells is possible bins, so map lidx_b to multipole index
                # of transfer function.
                # Otherwise lidx_b should just be lidx.
                if sparse:
                    lidx = idxmap[lidx_b]
                else:
                    lidx = lidx_b

                if self.mpi_rank == 0 and verbose:
                    sys.stdout.write('\r'+'lidx: {}/{}, ell: {}'.format(
                        lidx_b, ells.size-1, ell))
                for Lidx, L in enumerate(L_range):
                    L = ell + L
                    if L < 0:
                        continue

                    # spherical bessel is zero for kr << L
                    # so dont waste time calculating those values
                    if optimize:
                        if L < 20:
                            kmin_idx = 0
                        elif L < 100:
                            kmin_idx = kr_idx[int(0.5 * L)]
                        elif L < 500:
                            kmin_idx = kr_idx[int(0.75 * L)]
                        elif L < 1000:
                            kmin_idx = kr_idx[int(0.8 * L)]
                        else:
                            kmin_idx = kr_idx[int(0.9 * L)]
                    else:
                        kmin_idx = 0

                    if interp and optimize:

                        if L < 20:
                            kmin_old_idx = 0
                        elif L < 100:
                            kmin_old_idx = kr_old_idx[int(0.5 * L)]
                        elif L < 500:
                            kmin_old_idx = kr_old_idx[int(0.75 * L)]
                        elif L < 1000:
                            kmin_old_idx = kr_old_idx[int(0.8 * L)]
                        else:
                            kmin_old_idx = kr_old_idx[int(0.9 * L)]

                        if kmin_old_idx == k_old_size - 1:
                            kmin_old_idx -= 1
                    else:
                        kmin_old_idx = 0

                    # If possible, reuse spherical bessels j_L from ell-1
                    # in case these ells are not full-sized, dont do it
                    if lidx == 0 or ell - 1 != ell_prev:
                        # first pass, or sparse ells: fill all
                        jL[Lidx,kmin_idx:] = spherical_jn(L, kr[kmin_idx:])
                    else:
                        # second pass only fill new row
                        if Lidx == L_range.size - 1:
                            jL[Lidx,:] *= 0.
                            jL[Lidx,kmin_idx:] = spherical_jn(L, kr[kmin_idx:])

                    # loop over T, E, B
                    for pidx, pol in enumerate(pols_t):

                        if pol != 'B':
                            # This assumes no B contribution to scalar
                            if interp:
                                cs = CubicSpline(k_old[kmin_old_idx:],
                                         transfer_s[pidx,lidx,kmin_old_idx:])
                                tmp_s[kmin_idx:] = cs(k[kmin_idx:])
                            else:
                                tmp_s[kmin_idx:] = transfer_s[pidx,lidx,kmin_idx:]

                            tmp_s[kmin_idx:] *= jL[Lidx,kmin_idx:]

                        # Tensor.
                        if interp:
                            cs = CubicSpline(k_old[kmin_old_idx:],
                                     transfer_t[pidx,lidx,kmin_old_idx:])
                            tmp_t[kmin_idx:] = cs(k[kmin_idx:])
                        else:
                            tmp_t[kmin_idx:] = transfer_t[pidx,lidx,kmin_idx:]
                        tmp_t[kmin_idx:] *= jL[Lidx,kmin_idx:]

                        for kidx in xrange(ks):

                            if pol != 'B':
                                # scalars
                                integrand_s = tmp_s[kmin_idx:] * \
                                    func[kidx,kmin_idx:]
                                b_int_s = trapz(integrand_s, k[kmin_idx:])
                                beta_s[lidx_b,Lidx,kidx,pidx,ridx] = b_int_s

                            # tensors
                            integrand_t = tmp_t[kmin_idx:] * \
                                func[kidx,kmin_idx:]
                            b_int_t = trapz(integrand_t, k[kmin_idx:])
                            beta_t[lidx_b,Lidx,kidx,pidx,ridx] = b_int_t

                # Permute rows such that oldest row can be replaced next ell
                # no harm doing this even when ells are sparse, see above.
                jL = np.roll(jL, -1, axis=0)

                ell_prev = ell

                if self.mpi_rank == 0:
                    sys.stdout.flush()

            if self.mpi_rank == 0 and verbose:
                # To have correct next print statement.
                sys.stdout.write('\n')

        beta_s *= (2 / np.pi)
        beta_t *= (2 / np.pi)

        self.barrier()

        if self.mpi_rank == 0 and verbose:
            print('Interpolating beta over multipoles')
        # Spline betas to full size in ell again.
        beta_s_full = np.zeros((ells_out.size, L_range.size,
                                ks, len(pols_s), radii_sub.size))
        beta_t_full = np.zeros((ells_out.size, L_range.size,
                                ks, len(pols_t), radii_sub.size))

        for ridx, _ in enumerate(radii_sub):

            if self.mpi_rank == 0 and verbose:
                sys.stdout.write('\r'+'ridx: {}/{}'.format(
                    ridx+1, radii_sub.size))

            for Lidx, _ in enumerate(L_range):
                for pidx, pol in enumerate(pols_t):
                    for kidx in xrange(ks):

                        # Tensor.
                        cs = CubicSpline(ells,
                             beta_t[:,Lidx,kidx,pidx,ridx])
                        beta_t_full[:,Lidx,kidx,pidx,ridx] \
                            = cs(ells_out)

                        if pol == 'B':
                            continue

                        # Scalar.
                        cs = CubicSpline(ells,
                             beta_s[:,Lidx,kidx,pidx,ridx])
                        beta_s_full[:,Lidx,kidx,pidx,ridx] \
                            = cs(ells_out)

        if self.mpi_rank == 0 and verbose:
            print('')
            sys.stdout.flush()

        beta_s = beta_s_full
        beta_t = beta_t_full

        # Combine all sub range (part of radii) betas on root if mpi.
        if self.mpi:
            self.barrier()

            if self.mpi_rank == 0 and verbose:
                print('Combining parallel computed parts of beta')

            # create full size beta on root
            if self.mpi_rank == 0:

                beta_s_full = np.zeros((ells_out.size, L_range.size,
                                        ks, len(pols_s), radii.size))
                beta_t_full = np.zeros((ells_out.size, L_range.size,
                                        ks, len(pols_t), radii.size))

                # Already place root beta sub into beta_full.
                for ridx, radius in enumerate(radii_per_rank[0]):
                    # Find radius index in total radii.
                    ridx_tot, = np.where(radii == radius)[0]
                    beta_s_full[:,:,:,:,ridx_tot] = beta_s[:,:,:,:,ridx]
                    beta_t_full[:,:,:,:,ridx_tot] = beta_t[:,:,:,:,ridx]

            else:
                beta_s_full = None
                beta_t_full = None

            # loop over all non-root ranks
            for rank in xrange(1, self.mpi_size):

                # allocate space for sub beta on root
                if self.mpi_rank == 0:
                    r_size = radii_per_rank[rank].size

                    beta_s_sub = np.ones((ells_out.size,L_range.size,
                                          ks,len(pols_s),r_size))
                    beta_t_sub = np.ones((ells_out.size,L_range.size,
                                          ks,len(pols_t),r_size))

                # send beta_sub to root
                if self.mpi_rank == rank:
                    self._comm.Send(beta_s, dest=0, tag=rank)
                    self._comm.Send(beta_t, dest=0,
                                    tag=rank + self.mpi_size + 1)
                    if verbose == 2:
                        print(self.mpi_rank, 'sent')

                if self.mpi_rank == 0:
                    self._comm.Recv(beta_s_sub,
                                    source=rank, tag=rank)
                    self._comm.Recv(beta_t_sub, source=rank,
                                    tag=rank + self.mpi_size + 1)

                    if verbose == 2:
                        print('root received {}'.format(rank))

                    # place into beta_full
                    for ridx, radius in enumerate(radii_per_rank[rank]):

                        # find radius index in total radii
                        ridx_tot, = np.where(radii == radius)[0]

                        beta_s_full[:,:,:,:,ridx_tot] = \
                            beta_s_sub[:,:,:,:,ridx]
                        beta_t_full[:,:,:,:,ridx_tot] = \
                            beta_t_sub[:,:,:,:,ridx]

            # broadcast full beta array to all ranks
            beta_s = self.broadcast_array(beta_s_full)
            beta_t = self.broadcast_array(beta_t_full)

        # Bin beta (i.e. mean per bin).
        if self.mpi_rank == 0 and verbose:
            print('Binning beta')

        b_beta_s, b_beta_t = self._bin_beta(ells_out, beta_s, beta_t)

        # Unbinned versions.
        self.beta['beta_s'] = beta_s
        self.beta['beta_t'] = beta_t

        # Binned versions.
        self.beta['b_beta_s'] = b_beta_s
        self.beta['b_beta_t'] = b_beta_t

        self.beta['ells'] = ells_out

        self.beta['init_beta'] = True

        return
        
    def _bin_beta(self, ells, beta_s, beta_t, bins=None):
        '''
        Compute mean of beta in bins.

        Arguments
        ---------
        ells : array-like
            multipoles used for beta, must match size first
            dimension beta_s and beta_t.
        beta_s : array-like
            Scalar beta array (shape = (ells, Ls, ks, pols,rs))
        beta_t : array-like
            Tensor beta array (shape = (ells, Ls, ks, pols,rs))

        Keyword arguments
        -----------------
        bins : array-like, None
            Lower/left edges of bins, if None, use internal
            bins. (default : None)

        Returns
        -----
        b_beta_s : array-like
            Binned scalar beta (shape = (bins, Ls, ks, pols,rs)).
        b_beta_t : array-like
            Binned tensor beta (shape = (bins, Ls, ks, pols,rs)).
        '''

        if bins is None:
            bins = self.bins['bins']

        self.beta['bins'] = bins

        new_shape_s = list(beta_s.shape)
        new_shape_t = list(beta_t.shape)
        new_shape_s[0] = len(bins)
        new_shape_t[0] = len(bins)
        
        beta_s_f = np.asfortranarray(beta_s)
        beta_t_f = np.asfortranarray(beta_t)

        b_beta_s_f = np.zeros(new_shape_s)
        b_beta_t_f = np.zeros(new_shape_t)

        # This is just done because inner loop is over ell.
        b_beta_s_f = np.asfortranarray(b_beta_s_f)
        b_beta_t_f = np.asfortranarray(b_beta_t_f)

        # Float array with bins + (lmax + 0.1) for binned_stat
        # binned_stat output is one less than input bins size
        bins_ext = np.empty(len(bins) + 1, dtype=float)
        bins_ext[:-1] = bins
        bins_ext[-1] = bins[-1] + 0.1
        
        # Use tensor because it has I, E, \emph{and} B.
        n_Ls, n_ks, n_pols_t, n_radii = new_shape_t[1:] 

        for pidx in xrange(n_pols_t):
            for kidx in xrange(n_ks):
                for ridx in xrange(n_radii):
                    for Lidx in xrange(n_Ls):

                        if pidx != 2: # i.e. not B-mode.
                            # Scalar.
                            tmp_beta = beta_s_f[:,Lidx,kidx,pidx,ridx]

                            b_beta_s_f[:,Lidx,kidx,pidx,ridx], _, _ = \
                                tools.binned_statistic(ells, tmp_beta, statistic='mean',
                                                 bins=bins_ext)

                        # Tensor
                        tmp_beta = beta_t_f[:,Lidx,kidx,pidx,ridx]

                        b_beta_t_f[:,Lidx,kidx,pidx,ridx], _, _ = \
                            tools.binned_statistic(ells, tmp_beta, statistic='mean',
                                             bins=bins_ext)

        # Check for nans.
        if np.any(np.isnan(beta_s)):
            raise ValueError('nan in beta_s')
        if np.any(np.isnan(beta_t)):
            raise ValueError('nan in beta_t')

        # Turn back to c-ordered.
        beta_s = np.ascontiguousarray(beta_s_f)
        beta_t = np.ascontiguousarray(beta_t_f)

        b_beta_s = np.ascontiguousarray(b_beta_s_f)
        b_beta_t = np.ascontiguousarray(b_beta_t_f)
        
        return b_beta_s, b_beta_t
        
class Template(object):
    '''
    Create n k-functions in arrays of shape (n, k.size)
    '''

    def __init__(self, **kwargs):
        '''

        Keyword Arguments
        -----------------
        kwargs : {MPIBase_opts}

        Notes
        -----
        <h zeta zeta > = (2pi)^3 f(k1, k2, k3) delta(k1 + k2 + k3)
        * e(\hat(k1)) \hat(k2) \hat(k3),
        where f = 16 pi^4 As^2 fnl * S

        S (shape) is for (see Planck XVII):
        local: S = 2(1/(k1k2)^3 + 2 perm.),
        equil: S = 6(-1/(k1k2)^3 - 2 perm.  - 2/(k1k2k3)^2 + 1/(k1k2^2k3^3 + 5 perm.) )
        ortho: S = 6(-3/(k1k2)^3 - 2 perm.  - 8/(k1k2k3)^2 + 3/(k1k2^2k3^3 + 5 perm.) )

        For local we only need alpha and beta.
        For equilateral and orthgonal we need alpha, beta, gamma, delta
        (see Creminelli et al. 2005).
        '''

        self.scalar_amp = 2.1056e-9
        self.common_amp = 16 * np.pi**4 * self.scalar_amp**2

        super(Template, self).__init__(**kwargs)

    def local(self, k):
        '''
        Get the wavenumber arrays for the local template.

        Arguments
        ---------
        k : array-like, None
            Monotonically increasing array of wavenumbers.

        Returns
        -------
        template : array-like
            shape (2, k.size) used for beta (0), alpha (1).
            For beta we simply have k^-3, for alpha
            just an array of ones.

        Notes
        -----
        '''

        if k is None:
            k = self.cosmo['scalar']['k']

        km3 = k**-3
        ones = np.ones(k.size)

        template = np.asarray([km3, ones])

        return template

    def equilateral(self, k):
        '''
        Get the wavenumber arrays for the equilateral template.

        Arguments
        ---------
        k : array-like, None
            Monotonically increasing array of wavenumbers.

        Returns
        -------
        template : array-like
            shape (2, k.size) used for beta (0), alpha (1).
            For beta we simply have k^-3, for alpha
            just an array of ones.
        '''

        if k is None:
            k = self.cosmo['scalar']['k']

        km3 = k**-3
        km2 = k**-2
        km1 = k**-1
        ones = np.ones(k.size)

        # Keep same ordering as local.
        template = np.asarray([km3, ones, km2, km1])

        return template

    def orthogonal(self, k):
        '''
        Note, same as equilateral.
        '''

        return self.equilateral(k=k)

    def get_prim_amp(self, prim_template):
        '''
        Return the overal factor of given primordial
        shape template: 2 of local, 6 for equillateral
        and orthogonal.

        Arguments
        ---------
        prim_template : str
            Either "local", "equilateral" or "orthogonal"

        Returns
        -------
        amp : float
        '''

        if prim_template == 'local':
            return 2.

        elif prim_template == 'equilateral':
            return 6.

        elif prim_template == 'orthogonal':
            return 6.


class Fisher(Template, PreCalc):

    def __init__(self, base_dir, **kwargs):
        '''

        Arguments
        ---------
        base_dir : str
            Working directory.

        Keyword Arguments
        -----------------
        kwargs : {MPIBase_opts}


        Notes
        -----
        Will create "precomputed" and "fisher" directories
        in specified working directory if not present already.

        Example run:

        >> F = Fisher('.')
        >> F.get_camb_output(camb_out_dir=camb_dir)
        >> F.get_bins(*kwargs)
        >> F.get_beta(**kwargs)
        >> F.get_binned_bispec('equilateral') # Precomputes beta
        >> fisher = F.interp_fisher()
        '''

        super(Fisher, self).__init__(**kwargs)

        if self.mpi_rank == 0:
            if not os.path.exists(base_dir):
                raise ValueError('base_dir not found')
            self.base_dir = base_dir

        # Create subdirectories.
        self.subdirs = {}
        sdirs = ['precomputed', 'fisher']

        for sdir in sdirs:
            self.subdirs[sdir] = opj(base_dir, sdir)
            if self.mpi_rank == 0:
                if not os.path.exists(self.subdirs[sdir]):
                    os.makedirs(self.subdirs[sdir])

    def init_pol_triplets(self, single_bmode=True):
        '''
        Store polarization triples internally in (..,3) shaped array.

        For now, only triplets containing a single B-mode.

        Keyword Arguments
        -----------------
        single_bmode : bool
            Only create triplets that contain a single B-mode.
            (default : True)

        Notes
        -----
        I = 0, E = 1, B = 2
        '''

        if single_bmode is False:
            raise NotImplementedError('Not possible yet')

        pol_trpl = np.zeros((12, 3), dtype=int)

        pol_trpl[0] = 0, 0, 2
        pol_trpl[1] = 0, 2, 0
        pol_trpl[2] = 2, 0, 0
        pol_trpl[3] = 1, 0, 2
        pol_trpl[4] = 1, 2, 0
        pol_trpl[5] = 2, 1, 0
        pol_trpl[6] = 0, 1, 2
        pol_trpl[7] = 0, 2, 1
        pol_trpl[8] = 2, 0, 1
        pol_trpl[9] = 1, 1, 2
        pol_trpl[10] = 1, 2, 1
        pol_trpl[11] = 2, 1, 1

        self.bispec['pol_trpl'] = pol_trpl

#    @profile
    def _binned_bispectrum(self, DL1, DL2, DL3, prim_template='local',
                           radii_sub=None):
        '''
        Return binned bispectrum for given \Delta L triplet.

        Arguments
        ---------
        DL1 : Delta L_1
        DL2 : Delta L_2
        DL3 : Delta L_3

        Keyword arguments
        -----------------
        prim_template : str
            Either "local", "equilateral" or "orthogonal". 
            (default : local)
        radii_sub : array-like, None
            If array, use these radii instead of those used for
            beta. Radii must be a subset the beta radii.
            (default : None)

        Returns
        -------
        binned_bispectrum : array-like
            Shape: (nbin, nbin, nbin, npol), where npol
            is the number of polarization triplets considered.
            Note, only on root.
        '''

        bins = self.bins['bins']
        ells = self.bins['ells']
        lmin = ells[0]

        # Note that these are different per rank.
        num_pass = self.bins['num_pass']
        first_pass = self.bins['first_pass']

        wig_t = self.bispec['wig_t']
        wig_s = self.bispec['wig_s']

        pol_trpl = self.bispec['pol_trpl']
        psize = pol_trpl.shape[0]

        # Binned betas.
        beta_s = self.beta['b_beta_s']
        beta_t = self.beta['b_beta_t']

        # Get radii used for beta
        if radii_sub is None:
            radii = self.beta['radii']
        else:
            radii_full = list(self.beta['radii'])
            # Find elements of radii_sub in radii_full.
            ridxs = np.ones(len(radii_sub), dtype=int)
            for ridx, r in enumerate(radii_sub):
                # Raises error when r is not found.
                ridxs[ridx] = radii_full.index(r)
            
            # Reshape betas to radii_sub. Makes copy
            beta_s = np.ascontiguousarray(beta_s[...,ridxs])
            beta_t = np.ascontiguousarray(beta_t[...,ridxs])
            radii = radii_sub

        r2 = radii**2

        # allocate r arrays
        integrand = np.zeros_like(radii)
        integrand_tss = integrand.copy()
        integrand_sts = integrand.copy()
        integrand_sst = integrand.copy()

        # check parity
        parity = self.bins['parity']
        if parity == 'odd' and (DL1 + DL2 + DL3) % 2 == 0:
            warnings.warn('parity is odd and DL1 + DL2 + DL3 is even, '
                          'bispectrum is zero')
            return

        elif parity == 'even' and (DL1 + DL2 + DL3) % 2 == 1:
            warnings.warn('parity is even and DL1 + DL2 + DL3 is odd, '
                          'bispectrum is zero')
            return

        # Indices to L arrays
        Lidx1 = DL1 + 2
        Lidx2 = DL2 + 2
        Lidx3 = DL3 + 2

        # define function names locally for faster lookup
        wig3jj = wig.wig3jj_array
        wig9jj = wig.wig9jj_array
        trapz_loc = trapz

        bins_outer_f = bins.copy() # f = full
        idx_outer_f = np.arange(bins_outer_f.size)

        bins_on_rank, idxs_on_rank = self.get_bins_on_rank(return_idx=True)

        # allocate bispectrum
        nbins = bins.size
        bispectrum = np.zeros((bins_on_rank.size, nbins, nbins, psize))

        # numerical factors that differentiate equilateral and orthogonal
        if prim_template == 'equilateral':
            num_a = -1.
            num_b = -2.
            num_c = 1.
        elif prim_template == 'orthogonal':
            num_a = -3.
            num_b = -8.
            num_c = 3.

        for idxb, (idx1, i1) in enumerate(zip(idxs_on_rank, bins_on_rank)):
            # note, idxb is bins_on_rank index for bispectrum per rank
            # idx1 is index to full-sized bin array, i1 is bin
            # load binned beta
            beta_s_l1 = beta_s[idx1,Lidx1,0,:,:] # (2,r.size)
            beta_t_l1 = beta_t[idx1,Lidx1,0,:,:] # (3,r.size)

            alpha_s_l1 = beta_s[idx1,Lidx1,1,:,:] # (2,r.size)
            alpha_t_l1 = beta_t[idx1,Lidx1,1,:,:] # (3,r.size)

            if prim_template != 'local':
                delta_s_l1 = beta_s[idx1,Lidx1,2,:,:]
                delta_t_l1 = beta_t[idx1,Lidx1,2,:,:]

                gamma_s_l1 = beta_s[idx1,Lidx1,3,:,:]
                gamma_t_l1 = beta_t[idx1,Lidx1,3,:,:]

            for idx2, i2 in enumerate(bins[idx1:]):
                idx2 += idx1

                # load binned beta
                beta_s_l2 = beta_s[idx2,Lidx2,0,:,:] # (2,r.size)
                beta_t_l2 = beta_t[idx2,Lidx2,0,:,:] # (3,r.size)

                alpha_s_l2 = beta_s[idx2,Lidx2,1,:,:] # (2,r.size)
                alpha_t_l2 = beta_t[idx2,Lidx2,1,:,:] # (3,r.size)

                if prim_template != 'local':
                    delta_s_l2 = beta_s[idx2,Lidx2,2,:,:]
                    delta_t_l2 = beta_t[idx2,Lidx2,2,:,:]

                    gamma_s_l2 = beta_s[idx2,Lidx2,3,:,:]
                    gamma_t_l2 = beta_t[idx2,Lidx2,3,:,:]

                for idx3, i3 in enumerate(bins[idx2:]):
                    idx3 += idx2

                    num = num_pass[idxb, idx2, idx3]
                    if num == 0.:
                        # No valid ell tuples in this bin
                        # Note: do not impose conditions on
                        # the bins here!
                        continue

                    # load binned beta
                    beta_s_l3 = beta_s[idx3,Lidx3,0,:,:] # (2,r.size)
                    beta_t_l3 = beta_t[idx3,Lidx3,0,:,:] # (3,r.size)

                    alpha_s_l3 = beta_s[idx3,Lidx3,1,:,:] # (2,r.size)
                    alpha_t_l3 = beta_t[idx3,Lidx3,1,:,:] # (3,r.size)

                    if prim_template != 'local':
                        delta_s_l3 = beta_s[idx3,Lidx3,2,:,:]
                        delta_t_l3 = beta_t[idx3,Lidx3,2,:,:]

                        gamma_s_l3 = beta_s[idx3,Lidx3,3,:,:]
                        gamma_t_l3 = beta_t[idx3,Lidx3,3,:,:]

                    # Load the ell triplets used per bin
                    ell1, ell2, ell3 = first_pass[idxb, idx2, idx3,:]

                    if ell1 < 2 or ell2 < 2 or ell3 < 2:
                        raise ValueError("ells are wrong: ell1: {}, "
                                         "ell2: {}, ell3: {}".format(
                                             ell1, ell2, ell3))

                    # indices to full-size ell arrays
                    lidx1 = ell1 - lmin
                    lidx2 = ell2 - lmin
                    lidx3 = ell3 - lmin

                    L1 = DL1 + ell1
                    L2 = DL2 + ell2
                    L3 = DL3 + ell3

                    # Calculate the angular prefactor (eq. 6.19 thesis Shiraishi)
                    # before polarization loop
                    # Overall angular part
                    ang = wig3jj([2*L1, 2*L2, 2*L3,
                                  0, 0, 0])
                    if ang == 0:
                        # Triangle constraint on L's not obeyed.
                        continue

                    # We do not need this prefactor, cancels in Fisher.
                    # ang *= np.real((-1j)**(ell1 + ell2 + ell3 - 1))
                    # ang *= np.imag((-1j)**(ell1 + ell2 + ell3)) # Matches shiraishi.

                    ang *= (-1)**((L1 + L2 + L3)/2) # Power is always int.

                    # prefactor of I^000_{L1L2L3}
                    ang *= np.sqrt( (2*L1 + 1) * (2*L2 + 1) * (2*L3 + 1))
                    ang /= (2. * np.sqrt(np.pi))

                    # calculate the angular parts for each S, S, T comb
                    # TSS
                    ang_tss = wig_t[lidx1, Lidx1]
                    ang_tss *= wig_s[lidx2, Lidx2]
                    ang_tss *= wig_s[lidx3, Lidx3]

                    ang_tss *= ang
                    if ang_tss != 0.:
                        ang_tss *= wig9jj( [(2*ell1),  (2*ell2),  (2*ell3),
                                            (2*L1),  (2*L2),  (2*L3),
                                            4,  2,  2] ) 

                    else:
                        # don't waste time on 9j if zero anyway
                        pass

                    # STS
                    ang_sts = wig_s[lidx1, Lidx1]
                    ang_sts *= wig_t[lidx2, Lidx2]
                    ang_sts *= wig_s[lidx3, Lidx3]

                    ang_sts *= ang
                    if ang_sts != 0.:
                        ang_sts *= wig9jj( [(2*ell1),  (2*ell2),  (2*ell3),
                                                (2*L1),  (2*L2),  (2*L3),
                                                2, 4,  2] )
                    else:
                        pass
                    # TSS
                    ang_sst = wig_s[lidx1, Lidx1]
                    ang_sst *= wig_s[lidx2, Lidx2]
                    ang_sst *= wig_t[lidx3, Lidx3]

                    ang_sst *= ang
                    if ang_sst != 0.:
                        ang_sst *= wig9jj( [(2*ell1),  (2*ell2),  (2*ell3),
                                                (2*L1),  (2*L2),  (2*L3),
                                                2,  2,  4] )
                    else:
                        pass

                    if ang_tss == 0. and ang_sts == 0. and ang_sst == 0. \
                       and ell1!=ell2!=ell3:
                        # wrong L,ell comb, determine what went wrong
                        print(ell1, ell2, ell3, L1, L2, L3, num)
                        print(ang_tss, ang_sts, ang_sst)
                        print(np.abs(ell1 - ell2) <= ell3 <= ell1 + ell2)
                        print(np.abs(L1 - L2) <= L3 <= L1 + L2)
                        print(np.abs(ell1 - L1) <= 1 <= ell1 + L1)
                        print(np.abs(ell1 - L1) <= 2 <= ell1 + L1)
                        print(np.abs(ell2 - L2) <= 1 <= ell2 + L2)
                        print(np.abs(ell2 - L2) <= 2 <= ell2 + L2)
                        print(np.abs(ell3 - L3) <= 1 <= ell3 + L3)
                        print(np.abs(ell3 - L3) <= 2 <= ell3 + L3)
                        raise ValueError("angular prefactor is zero")
                        #continue

                    # loop over pol combs
                    # TODO put this in function
                    for pidx in xrange(psize):

                        pidx1, pidx2, pidx3 = pol_trpl[pidx,:]

                        # integrate over bba + bab + abb for each T, S, S comb
                        integrand[:] = 0.

                        # TSS
                        if pidx2 == 2 or pidx3 == 2 or ang_tss == 0.:
                            # no B-mode for scalar or this l, L combination
                            pass
                        else:
                            if pidx1 == 2:
                                assert (1 + L1 + ell1)%2 == 0

                            integrand_tss[:] = beta_t_l1[pidx1,:] * \
                                beta_s_l2[pidx2,:] * alpha_s_l3[pidx3,:]

                            integrand_tss += beta_t_l1[pidx1,:] * \
                                alpha_s_l2[pidx2,:] * beta_s_l3[pidx3,:]

                            integrand_tss += alpha_t_l1[pidx1,:] * \
                                beta_s_l2[pidx2,:] * beta_s_l3[pidx3,:]

                            if prim_template != 'local':
                                integrand_tss *= num_a

                                integrand_tss += num_b * delta_t_l1[pidx1,:] * \
                                delta_s_l2[pidx2,:] * delta_s_l3[pidx3,:]

                                # bgd
                                integrand_tss += num_c * beta_t_l1[pidx1,:] * \
                                gamma_s_l2[pidx2,:] * delta_s_l3[pidx3,:]

                                # bdg
                                integrand_tss += num_c * beta_t_l1[pidx1,:] * \
                                delta_s_l2[pidx2,:] * gamma_s_l3[pidx3,:]

                                # gbd
                                integrand_tss += num_c * gamma_t_l1[pidx1,:] * \
                                beta_s_l2[pidx2,:] * delta_s_l3[pidx3,:]

                                # gdb
                                integrand_tss += num_c * gamma_t_l1[pidx1,:] * \
                                delta_s_l2[pidx2,:] * beta_s_l3[pidx3,:]

                                # dgb
                                integrand_tss += num_c * delta_t_l1[pidx1,:] * \
                                gamma_s_l2[pidx2,:] * beta_s_l3[pidx3,:]

                                # dbg
                                integrand_tss += num_c * delta_t_l1[pidx1,:] * \
                                beta_s_l2[pidx2,:] * gamma_s_l3[pidx3,:]

                            integrand_tss *= ang_tss

                            integrand += integrand_tss

                        # STS
                        if pidx1 == 2 or pidx3 == 2 or ang_sts == 0.:
                            # no B-mode for scalar
                            pass
                        else:
                            if pidx2 == 2:
                                assert (1 + L2 + ell2)%2 == 0

                            integrand_sts[:] = beta_s_l1[pidx1,:] * \
                                beta_t_l2[pidx2,:] * alpha_s_l3[pidx3,:]

                            integrand_sts += beta_s_l1[pidx1,:] * \
                                alpha_t_l2[pidx2,:] * beta_s_l3[pidx3,:]

                            integrand_sts += alpha_s_l1[pidx1,:] * \
                                beta_t_l2[pidx2,:] * beta_s_l3[pidx3,:]

                            if prim_template != 'local':
                                integrand_sts *= num_a

                                integrand_sts += num_b * delta_s_l1[pidx1,:] * \
                                delta_t_l2[pidx2,:] * delta_s_l3[pidx3,:]

                                # bgd
                                integrand_sts += num_c * beta_s_l1[pidx1,:] * \
                                gamma_t_l2[pidx2,:] * delta_s_l3[pidx3,:]

                                # bdg
                                integrand_sts += num_c * beta_s_l1[pidx1,:] * \
                                delta_t_l2[pidx2,:] * gamma_s_l3[pidx3,:]

                                # gbd
                                integrand_sts += num_c * gamma_s_l1[pidx1,:] * \
                                beta_t_l2[pidx2,:] * delta_s_l3[pidx3,:]

                                # gdb
                                integrand_sts += num_c * gamma_s_l1[pidx1,:] * \
                                delta_t_l2[pidx2,:] * beta_s_l3[pidx3,:]

                                # dgb
                                integrand_sts += num_c * delta_s_l1[pidx1,:] * \
                                gamma_t_l2[pidx2,:] * beta_s_l3[pidx3,:]

                                # dbg
                                integrand_tss += num_c * delta_s_l1[pidx1,:] * \
                                beta_t_l2[pidx2,:] * gamma_s_l3[pidx3,:]

                            integrand_sts *= ang_sts

                            integrand += integrand_sts

                        # SST
                        if pidx1 == 2 or pidx2 == 2 or ang_sst == 0.:
                            # no B-mode for scalar
                            pass
                        else:
                            if pidx3 == 2:
                                assert (1 + L3 + ell3)%2 == 0

                            integrand_sst[:] = beta_s_l1[pidx1,:] * \
                                beta_s_l2[pidx2,:] * alpha_t_l3[pidx3,:]

                            integrand_sst += beta_s_l1[pidx1,:] * \
                                alpha_s_l2[pidx2,:] * beta_t_l3[pidx3,:]

                            integrand_sst += alpha_s_l1[pidx1,:] * \
                                beta_s_l2[pidx2,:] * beta_t_l3[pidx3,:]

                            if prim_template != 'local':
                                integrand_sst *= num_a

                                integrand_sst += num_b * delta_s_l1[pidx1,:] * \
                                delta_s_l2[pidx2,:] * delta_t_l3[pidx3,:]

                                # bgd
                                integrand_sst += num_c * beta_s_l1[pidx1,:] * \
                                gamma_s_l2[pidx2,:] * delta_t_l3[pidx3,:]

                                # bdg
                                integrand_sst += num_c * beta_s_l1[pidx1,:] * \
                                delta_s_l2[pidx2,:] * gamma_t_l3[pidx3,:]

                                # gbd
                                integrand_sst += num_c * gamma_s_l1[pidx1,:] * \
                                beta_s_l2[pidx2,:] * delta_t_l3[pidx3,:]

                                # gdb
                                integrand_sst += num_c * gamma_s_l1[pidx1,:] * \
                                delta_s_l2[pidx2,:] * beta_t_l3[pidx3,:]

                                # dgb
                                integrand_sst += num_c * delta_s_l1[pidx1,:] * \
                                gamma_s_l2[pidx2,:] * beta_t_l3[pidx3,:]

                                # dbg
                                integrand_sst += num_c * delta_s_l1[pidx1,:] * \
                                beta_s_l2[pidx2,:] * gamma_t_l3[pidx3,:]

                            integrand_sst *= ang_sst

                            integrand += integrand_sst

                        # Integrate over r
                        integrand *= r2
                        bispec = trapz_loc(integrand, radii)

                        bispectrum[idxb,idx2,idx3,pidx] = bispec

        bispectrum *= (8 * np.pi)**(3/2.) / 3. * self.get_prim_amp(prim_template)

        # Each rank has, for a unordered set of i1, the i2, i3, pol bispectra
        # Now we add them together on the root rank
        if self.mpi:
            self.barrier()

            # create empty full-sized bispectrum on root
            if self.mpi_rank == 0:
                bispec_full = np.zeros((bins_outer_f.size,nbins,nbins,psize))

                _, idxs_on_root = self.get_bins_on_rank(return_idx=True)

                # Place sub B on root in full B for root
                for i, fidx in enumerate(idxs_on_root):
                    # i is index to sub, fidx index to full
                    bispec_full[fidx,...] = bispectrum[i,...]

            else:
                bispec_full = 0

            # loop over all non-root ranks
            for rank in xrange(1, self.mpi_size):

                if self.mpi_rank == 0:
                    # The indices of the bins on the rank (on root).
                    _, idxs_on_rank = self.get_bins_on_rank(return_idx=True,
                                                        rank=rank)
                    bin_size = idxs_on_rank.size
                    # Allocate B sub on root for receiving.
                    bispec_rec = np.zeros((bin_size,nbins,nbins,psize))

                # send bispectrum to root
                if self.mpi_rank == rank:
                    self._comm.Send(bispectrum, dest=0, tag=rank)

                if self.mpi_rank == 0:
                    self._comm.Recv(bispec_rec, source=rank, tag=rank)

                    # place into root bispectrum
                    for i, fidx in enumerate(idxs_on_rank):
                        # i is index to sub, fidx index to full
                        bispec_full[fidx,...] = bispec_rec[i,...]

            return bispec_full

        else:
            # no MPI, so process already has full bispectrum
            return bispectrum

    def _compute_binned_bispec(self, prim_template, radii_sub=None):
        '''
        Compute the combined binned bispectra from all
        allowed L values.

        Arguments
        ---------
        prim_template : str
            Primordial template. Either "local", "equilateral"
            or "orthogonal".

        Returns
        -------
        binned_bispec : array-like
            Shape: (nbin, nbin, nbin, npol), where npol
            is the number of polarization triplets considered.
        '''

        # All the tuples for single B-mode bispectra.
        L_tups = [(+1, +1, +1),
                  (+1, -1, -1),
                  (-1, +1, +1),
                  (-1, -1, -1),
                  (+1, -1, +1),
                  (-1, +1, -1),
                  (+1, +1, -1),
                  (-1, -1, +1)]

        for Lidx, L_tup in enumerate(L_tups):

            if self.mpi_rank == 0:
                print('working on DL1, DL2, DL3:', L_tup)

            if Lidx == 0:
                binned_bispec = self._binned_bispectrum(*L_tup, 
                                    radii_sub=radii_sub,
                                    prim_template=prim_template)
            else:
                binned_bispec += self._binned_bispectrum(*L_tup,
                                    radii_sub=radii_sub,
                                    prim_template=prim_template)

        return binned_bispec

    def _load_binned_bispec(self, filename):
        '''
        Try to load and expand bispectrum.

        Arguments
        ---------
        filename : str
            Filename of bispectrum pkl file.

        Returns
        -------
        success : bool
            Returns False when bispectrum could not be loaded.
        '''
        
        success = True

        if self.mpi_rank == 0:

            path = self.subdirs['precomputed']
            bispec_file = opj(path, filename)

            try:
                pkl_file = open(bispec_file, 'rb')
            except IOError:
                print('{} not found'.format(bispec_file))
                success = False
            else:
                print('loaded bispectrum from {}'.format(bispec_file))
                self.bispec = pickle.load(pkl_file)
                pkl_file.close()

                # Loaded bispectrum is sparse, so expand.
                b_sparse = self.bispec['bispec']
                npol = b_sparse.shape[-1]
                nbins = self.bins['bins'].size
                mask_full = np.zeros((nbins, nbins, nbins, npol), dtype=bool)
                mask = self.bins['num_pass_full'].astype(bool)
                mask_full += mask[:,:,:,np.newaxis]
                b_full = np.zeros((nbins, nbins, nbins, npol))
                np.place(b_full, mask_full, b_sparse)

                self.bispec['bispec'] = b_full
                
        # This lets nonzero ranks know that file is not found.
        success = self.broadcast(success)

        if success is True:
            # for now, leave bispec on root. Pol_triplets are
            # on all ranks when calculated though, so broadcast.
            if self.mpi_rank == 0:            
                pol_trpl = self.bispec['pol_trpl']
            else:
                pol_trpl = None

            self.bispec['pol_trpl'] = self.broadcast_array(pol_trpl)

        return success

    def _save_binned_bispec(self, filename):
        '''
        Save bispectrum.

        Arguments
        ---------
        filename : str
            Filename of bispectrum pkl file (without path).

        Notes
        -----
        Only stores the elements of B that are allowed
        by parity, triangle condition etc. (determined
        by the nonzero elements of num_pass)
        '''

        if self.mpi_rank == 0:

            path = self.subdirs['precomputed']
            bispec_file = opj(path, filename)
            print('Storing bispec as: {}'.format(bispec_file))

            # Select allowed values.
            mask = self.bins['num_pass_full'].astype(bool)
            b_full = self.bispec['bispec']
            b_sparse = b_full[mask] # Copy.

            # Temporarily replace.
            self.bispec['bispec'] = b_sparse

            # Store in pickle file.            
            with open(bispec_file, 'wb') as handle:
                pickle.dump(self.bispec, handle,
                            protocol=pickle.HIGHEST_PROTOCOL)

            # Place back.
            self.bispec['bispec'] = b_full

    def get_binned_bispec(self, prim_template, load=True, 
                          radii_sub=None, tag=None):
        '''
        Compute binned bispectrum or load from disk.

        Arguments
        ---------
        prim_template : str
            Primordial template. Either "local", "equilateral"
            or "orthogonal".
        
        Keyword arguments
        -----------------
        load : bool
            Try to load bispectrum + properties.
        tag : str, None
            Tag appended to stored/loaded .pkl file as
            bispec_<prim_template>_<tag>.pkl. (default : None)
        '''

        # We need bins for loading/saving and recomputing.
        if not self.bins['init_bins']:
            raise ValueError('bins not initiated')
            
        if tag is None:
            filename = 'bispec_{}.pkl'.format(prim_template)
        else:
            filename = 'bispec_{}_{}.pkl'.format(prim_template, tag)

        recompute = not load
        
        if load:
            success = self._load_binned_bispec(filename)
            if not success:
                recompute = True

        if recompute:

            # To recompute we also need beta.
            if not self.beta['init_beta']:
                raise ValueError('beta not initiated')

            if self.mpi_rank == 0:
                print('Recomputing bispec')

            self.init_wig3j()
            self.init_pol_triplets(single_bmode=True)
            bispec = self._compute_binned_bispec(prim_template,
                                                   radii_sub=radii_sub)
            self.bispec['bispec'] = bispec

            # Save for next time.
            self._save_binned_bispec(filename)

    def get_bins(self, load=True, tag=None, **kwargs):
        '''
        Init bins or load from disk.

        Keyword arguments
        -----------------
        load : bool
            Try to load bispectrum + properties.
        tag : str, None
            Tag appended to stored/loaded .pkl file as bins_<tag>.pkl.
            (default : None)
        kwargs : {init_bins_opts}
        '''

        path = self.subdirs['precomputed']
        if tag is None:
            bins_file = opj(path, 'bins.pkl')
        else:
            bins_file = opj(path, 'bins_{}.pkl'.format(tag))
        recompute = not load

        if load:
            if self.mpi_rank == 0:
                # Loading and printing on root.
                try:
                    pkl_file = open(bins_file, 'rb')
                except IOError:
                    print('{} not found'.format(bins_file))
                    recompute = True
                else:
                    print('loaded bins from {}'.format(bins_file))
                    self.bins = pickle.load(pkl_file)
                    pkl_file.close()

            # This lets nonzero ranks know that file is not found.
            recompute = self.broadcast(recompute)

            if recompute is False:
                # Succesfully loaded on root.
                
                # Do not broadcast full-sized arrays to save memory.
                if self.mpi_rank == 0:
                    num_pass_full = self.bins.pop('num_pass_full')
                    first_pass_full = self.bins.pop('first_pass_full')

                self.bins = self.broadcast(self.bins)

                if self.mpi_rank == 0:
                    self.bins['num_pass_full'] = num_pass_full
                    self.bins['first_pass_full'] = first_pass_full

                # We might have different number of ranks
                # so we have to scatter bins again.
                self._scatter_bins()

                # Scatter num_pass and first pass.
                for rank in xrange(1, self.mpi_size):
                    if self.mpi_rank == 0:
                        # Extract slices corresponding to rank.
                        _, idxs_on_rank = self.get_bins_on_rank(
                            return_idx=True, rank=rank)
                        num_pass2send = num_pass_full[idxs_on_rank,...]
                        first_pass2send = first_pass_full[idxs_on_rank,...]
                        
                        if rank != 0:
                            self._comm.Send(num_pass2send, dest=rank, tag=rank)
                            self._comm.Send(first_pass2send, dest=rank,
                                            tag=rank + self.mpi_size)

                        if kwargs.get('verbose', False) == 2:
                            print('root sent to {}'.format(rank))

                    # Receive the arrays on the rank.
                    if self.mpi_rank == rank:

                        _, idxs_on_rank = self.get_bins_on_rank(return_idx=True)
                        num_bins_on_rank = idxs_on_rank.size
                        num_bins = self.bins['bins'].size
                        # MPI needs contiguous array to receive.
                        num_pass_rec = np.empty((num_bins_on_rank, num_bins,
                                                      num_bins), dtype=int)
                        first_pass_rec = np.empty((num_bins_on_rank, num_bins,
                                                        num_bins, 3), dtype=int)
                        self._comm.Recv(num_pass_rec,
                                        source=0, tag=rank)
                        self._comm.Recv(first_pass_rec,
                                        source=0, tag=rank + self.mpi_size)
                        if kwargs.get('verbose', False) == 2:
                            print('rank {} received'.format(rank))


                if self.mpi_rank == 0:
                    _, idxs_on_rank = self.get_bins_on_rank(
                        return_idx=True)
                    num_pass_rec = num_pass_full[idxs_on_rank,...]
                    first_pass_rec = first_pass_full[idxs_on_rank,...]
                        
                self.bins['num_pass'] = num_pass_rec
                self.bins['first_pass'] = first_pass_rec
                    
                # TODO, check if lmin, lmax, bins, parity all match
                # otherwise recompute

        if recompute:
            if self.mpi_rank == 0:
                print('Recomputing bins')

            self.init_bins(**kwargs)

            # Save for next time.
            if self.mpi_rank == 0:

                print('Storing bins as: {}'.format(bins_file))

                # Store in pickle file.
                with open(bins_file, 'wb') as handle:
                    pickle.dump(self.bins, handle,
                                protocol=pickle.HIGHEST_PROTOCOL)


    def get_beta(self, load=True, tag=None, **kwargs):
        '''
        Compute beta or load from disk.

        Keyword arguments
        -----------------
        load : bool
            Try to load beta. (default : True)
        tag : str, None
            Tag appended to stored/loaded .pkl file as beta_<tag>.pkl.
            (default : None)
        kwargs : {beta_opts}
        '''

        path = self.subdirs['precomputed']
        if tag is None:
            beta_file = opj(path, 'beta.pkl')
        else:
            beta_file = opj(path, 'beta_{}.pkl'.format(tag))
        recompute = not load

        if load:
            if self.mpi_rank == 0:
                # Loading and printing on root.
                try:
                    pkl_file = open(beta_file, 'rb')
                except IOError:
                    print('{} not found'.format(beta_file))
                    recompute = True
                else:
                    print('loaded beta from {}'.format(beta_file))
                    self.beta = pickle.load(pkl_file)
                    pkl_file.close()

            # This lets nonzero ranks know that file is not found.
            recompute = self.broadcast(recompute)

            if recompute is False:
                # Succesfully loaded on root, so broadcast.
                # Broadcast each key in beta dict separately, 
                # for lmax > 5000 file too big otherwise.

                if self.mpi_rank == 0:
                    beta_keys = list(self.beta.keys())

                if self.mpi_rank != 0:
                    self.beta = {}
                    beta_keys = None

                beta_keys = self.broadcast(beta_keys)

                if self.mpi_rank != 0:                    
                    for bkey in beta_keys:
                        self.beta[bkey] = None
                
                for bkey in beta_keys:
                    self.beta[bkey] = self.broadcast(self.beta[bkey])
                                            
        if recompute:
            if self.mpi_rank == 0:
                print('Recomputing beta')

            if not self.bins['init_bins']:
                raise ValueError('bins not initiated')

            self.init_beta(**kwargs)

            # Save for next time.
            if self.mpi_rank == 0:

                print('Storing beta as: {}'.format(beta_file))

                # Store in pickle file.
                with open(beta_file, 'wb') as handle:
                    pickle.dump(self.beta, handle,
                                protocol=pickle.HIGHEST_PROTOCOL)

    def get_cosmo(self, load=True, tag=None, **kwargs):
        '''
        Compute transfer functions and cls or load from disk.

        Keyword arguments
        -----------------
        load : bool
            Try to load cosmo quantities. (default : True)
        tag : str, None
            Tag appended to stored/loaded .pkl file as cosmo_<tag>.pkl.
            (default : None)
        kwargs : {init_cosmo_opts}
        '''

        path = self.subdirs['precomputed']
        if tag is None:
            cosmo_file = opj(path, 'cosmo.pkl')
        else:
            cosmo_file = opj(path, 'cosmo_{}.pkl'.format(tag))
        recompute = not load

        if load:
            if self.mpi_rank == 0:
                # Loading and printing on root.
                try:
                    pkl_file = open(cosmo_file, 'rb')
                except IOError:
                    print('{} not found'.format(cosmo_file))
                    recompute = True
                else:
                    print('loaded cosmo from {}'.format(cosmo_file))
                    self.cosmo = pickle.load(pkl_file)
                    pkl_file.close()

            # This lets nonzero ranks know that file is not found.
            recompute = self.broadcast(recompute)

            if recompute is False:
                # Succesfully loaded on root, so broadcast.
                self.cosmo = self.broadcast(self.cosmo)

        if recompute:
            if self.mpi_rank == 0:
                print('Recomputing cosmo')

            self.init_cosmo(**kwargs)

            # Save for next time.
            if self.mpi_rank == 0:

                print('Storing cosmo as: {}'.format(cosmo_file))

                # Store in pickle file.
                with open(cosmo_file, 'wb') as handle:
                    pickle.dump(self.cosmo, handle,
                                protocol=pickle.HIGHEST_PROTOCOL)
                
    def get_invcov(self, ells, nls, return_cov=False, write=False,
                   write_tag=None):
        '''
        Combine covariance into an inverse cov matrix.

        Arguments
        ---------
        ells : array-like
            Multipoles corresponding to nls.
        nls : array-like
            Total covariance Nl, Cl or Nl + Cl. Shape: (6, ells.size)
            in order: TT, EE, BB, TE, TB, EB. If shape is (4, ells.size),
            assume TB and EB are zero.

        Returns
        -------
        bin_invcov : array-like
            Binned inverse covariance, shape = (bins, 3, 3).
        bin_cov : array-like
            Binned covariance, shape = (bins, 3, 3), if 
            return_bin_cov is set.
        write : bool
            If True, write ells, invcov, cov to fisher directory 
            in pickle file. (default : False)
        write_tag : str, None
            If not None, write invcov_<tag>.pkl if `write` is set.
            (default : None)
        '''
        
        if not nls.shape[1] == ells.size:
            raise ValueError("ells.size = {}, nls.shape[1] = {}".format(
                ells.size, nls.shape[1]))

        nls_dict = {'TT': 0, 'EE': 1, 'BB': 2, 'TE': 3,
                    'ET': 3, 'BT': 4, 'TB': 4, 'EB': 5,
                    'BE': 5}

        cov = np.ones((ells.size, 3, 3))
        cov *= np.nan
        invcov = cov.copy()

        if nls.shape[0] == 4:
            # Assume TB and EB are zero with ell.
            nls = np.vstack([nls, np.zeros((2, ells.size), dtype=float)])

        for pidx1, pol1 in enumerate(['T', 'E', 'B']):
            for pidx2, pol2 in enumerate(['T', 'E', 'B']):

                nidx = nls_dict[pol1+pol2]
                nell = nls[nidx,:]

                # Bin
                cov[:,pidx1,pidx2] = nell

        # Invert.
        for lidx in xrange(ells.size):
            invcov[lidx,:,:] = inv(cov[lidx,:,:])

        if write and self.mpi_rank == 0:

            outdir = self.subdirs['fisher']

            if write_tag is None:
                invcov_file = opj(outdir, 'invcov.pkl')
            else:
                invcov_file = opj(outdir, 'invcov_{}.pkl'.format(write_tag))

            invcov_opts = dict(ells=ells, invcov=invcov, cov=cov)
                
            # Store in pickle file.            
            with open(invcov_file, 'wb') as handle:
                pickle.dump(invcov_opts, handle,
                            protocol=pickle.HIGHEST_PROTOCOL)

        if return_cov:
            return invcov, cov
        else:
            return invcov

    def get_binned_invcov(self, ells, nls, bins=None, return_bin_cov=False):
        '''
        Combine covariance into an inverse cov matrix
        per bin. We first bin (mean per bin), then compute inverse.

        Arguments
        ---------
        ells : array-like
            Multipoles corresponding to nls.
        nls : array-like
            Total covariance Nl, Cl or Nl + Cl. Shape: (6, ells.size)
            in order: TT, EE, BB, TE, TB, EB. If shape is (4, ells.size),
            assume TB and EB are zero.

        Keyword Arguments
        -----------------
        bins : array-like, None
            Left/lower edge of bins, if None, use internal bins.
            (default : None)

        Returns
        -------
        bin_invcov : array-like
            Binned inverse covariance, shape = (bins, 3, 3).
        bin_cov : array-like
            Binned covariance, shape = (bins, 3, 3), if 
            return_bin_cov is set.
        '''

        if not nls.shape[1] == ells.size:
            raise ValueError("ells.size = {}, nls.shape[0] = {}".format(
                ells.size, nls.shape[1]))

        if bins is None:
            bins = self.bins['bins']

        if not np.array_equal(ells, np.unique(ells)):
            raise ValueError('ells not monotonically increasing with Dl=1')

        if not ells[-1] >= bins[-1]:
            raise ValueError('lmax smaller than max bin')

        bin_cov = np.ones((bins.size, 3, 3))
        bin_cov *= np.nan
        bin_invcov = bin_cov.copy()

        if nls.shape[0] == 4:
            # Assume TB and EB are zero with ell.
            nls = np.vstack([nls, np.zeros((2, ells.size), dtype=float)])

        nls_dict = {'TT': 0, 'EE': 1, 'BB': 2, 'TE': 3,
                    'ET': 3, 'BT': 4, 'TB': 4, 'EB': 5,
                    'BE': 5}

        # Float array with bins + (lmax + 0.1) for binned_stat
        # binned_stat output is one less than input bins size.
        bins_ext = np.empty(bins.size + 1, dtype=float)
        bins_ext[:-1] = bins
        bmax = bins[-1]
        bins_ext[-1] = bmax + 0.1

        for pidx1, pol1 in enumerate(['T', 'E', 'B']):
            for pidx2, pol2 in enumerate(['T', 'E', 'B']):

                nidx = nls_dict[pol1+pol2]
                nell = nls[nidx,:]

                # Bin
                bin_cov[:,pidx1,pidx2], _, _ = tools.binned_statistic(
                    ells, nell, statistic='mean', bins=bins_ext)

        # Invert.
        for bidx in xrange(bins.size):
            bin_invcov[bidx,:,:] = inv(bin_cov[bidx,:,:])

        if return_bin_cov:
            return bin_invcov, bin_cov
        else:
            return bin_invcov

    # load_invcov function? Saves dict w/ all invcov stuff with tag..
    # if it finds correct file loads, otherwise recomputes and saves
    # nice thing is that you can save pickle files with cov and fisher
    # results with same tag.
    # or both in same file, that's even nicer because no ambiguity.
    # so load invov gives you fisher dict, then naive_fisher just populates
    # the fisher attribute.

    def _init_fisher_invcov(self, invcov):
        '''
        Given 3 x 3 covariance per multipole (or bin) return
        three (C^{-1})^XY arrays.

        Arguments
        ---------
        invcov : array-like
            Inverse covariance per ell (bin). Shape = (N, 3, 3).
        
        Returns
        -------
        invcov1, invcov2, invcov3 : array-like
            (C^{-1})^XY arrays. Shape = (N, npt, npt), where npt
            is the number of polarization triplets used for the 
            bispectrum (see `init_pol_triplets`).
        '''

        nptr = self.bispec['pol_trpl'].shape[0]
        N = invcov.shape[0] # Number of multipoles (or bins).

        # Example: for single B-mode: nptr = 12 (BTT, BEE, BET, BTE etc.)
        # Use the fact that 12x12 pol invcov can be factored
        # as (Cl-1)_l1^ip (Cl-1)_l2^jq (Cl-1)_l3^kr 
        invcov1 = np.ones((N, nptr, nptr)) 
        invcov2 = np.ones((N, nptr, nptr))
        invcov3 = np.ones((N, nptr, nptr))

        for tidx_a, tr_a in enumerate(self.bispec['pol_trpl']):
            # tr_a = ijk
            for tidx_b, tr_b in enumerate(self.bispec['pol_trpl']):
                # tr_a = pqr
                # a is first bispectrum, b second one.
                # tr = pol triplet

                tr_a1 = tr_a[0]
                tr_a2 = tr_a[1]
                tr_a3 = tr_a[2]

                tr_b1 = tr_b[0]
                tr_b2 = tr_b[1]
                tr_b3 = tr_b[2]

                invcov1[:,tidx_a,tidx_b] = invcov[:,tr_a1,tr_b1]
                invcov2[:,tidx_a,tidx_b] = invcov[:,tr_a2,tr_b2]
                invcov3[:,tidx_a,tidx_b] = invcov[:,tr_a3,tr_b3]

        return invcov1, invcov2, invcov3

    def naive_fisher(self, bin_invcov, lmin=None, lmax=None, fsky=1):
        '''
        Calculate the fisher information by squaring bins.

        Arguments
        ---------
        bin_invcov : array-like
            Binned inverse convariance, shape (bins, 3, 3), 
            see `get_binned_invcov`.
        
        Keyword Arguments
        -----------------
        lmin : int, None
            Lower limit of Fisher loop. If None,
            use first bin. (default : None)
        lmax : int, None
            Upper limit of Fisher loop. If None,
            use last bin. (default : None)
        fsky : float
            Fraction of sky assumed to be used.
            (default : 1)

        Returns
        -------
        fisher : float
            Fisher information.
        '''

        bin_size = self.bins['bins'].size
        bins = self.bins['bins']
        num_pass = self.bins['num_pass_full']
        bispec = self.bispec['bispec']

        invcov1, invcov2, invcov3 = self._init_fisher_invcov(bin_invcov)

        fisher = 0

        # allocate 12 x 12 cov for use in inner loop
        nptr = self.bispec['pol_trpl'].shape[0]
        cl123 = np.zeros((nptr, nptr), dtype=float)

        # Depending on lmin, start outer loop not at first bin.
        start_bidx = np.where(bins >= lmin)[0][0]
        # Depending on lmax, possibly end loops earlier.
        end_bidx = np.where(bins >= min(lmax, bins[-1]))[0][0] + 1

        # loop same loop as in binned_bispectrum
        for idx1, i1 in enumerate(bins[start_bidx:end_bidx]):
            idx1 += start_bidx
            cl1 = invcov1[idx1,:,:] # nptr x nptr matrix.

            for idx2, i2 in enumerate(bins[idx1:end_bidx]):
                idx2 += idx1
                cl2 = invcov2[idx2,:,:] # nptr x nptr matrix.

                cl12 = cl1 * cl2

                for idx3, i3 in enumerate(bins[idx2:end_bidx]):
                    idx3 += idx2

                    num = num_pass[idx1,idx2,idx3]
                    if num == 0:
                        continue

                    cl123[:] = cl12 # Copy.
                    cl123 *= invcov3[idx3,:,:] # nptr x nptr matrix.

                    B = bispec[idx1,idx2,idx3,:]

                    f = np.einsum("i,ij,j", B, cl123, B)

                    ## both B's have num 
                    #f /= float(num)
                    f *= num

                    # Delta_l1l2l3.
                    if i1 == i2 == i3:
                        f /= 6.
                    elif i1 != i2 != i3:
                        pass
                    else:
                        f /= 2.

                    fisher += f

        fisher *= fsky
        fisher *= self.common_amp ** 2 # (16 pi^4 As^2)^2

        return fisher

    def _mpi_scatter_for_fisher(self, bidx_per_rank, bidx_max=None):
        '''
        Scatter data required for interpolated fisher
        calculation from root to MPI ranks.
        
        Arguments
        ---------
        bidx_per_rank : list of ndarrays
            Array of bin indices for each MPI rank
            (object present on all ranks).
        
        Keyword arguments
        -----------------
        bidx_max : int, None
            Index of last bin. If None, use all bins.
            (default : None)

        Returns
        -------
        first_pass_per_bidx : list of ndarrays
            List of first_pass arrays with shape: (M, 3),
            same order as bidx_per_rank.
        bispec_per_bidx : list of ndarrays
            List of bispec arrays with shape (M, nptr),
            same order as bidx_per_rank. 
        num_triplets_per_bidx : array-like
            Number of valid tuples per bidx, same order as bidx_per_rank.

        Notes
        -----
        M = sum(num_pass[outer_bins,...].astype(bool)) (i.e. num_tuples)
        '''
        
        size = self.mpi_size
        nptr = self.bispec['pol_trpl'].shape[0]

        if bidx_max is not None:
            bidx_end = bidx_max + 1
            bins = self.bins['bins'][:bidx_end]
        else:
            bidx_end = None
            bins = self.bins['bins']

        num_bins = bins.size

        # For use in loop.
        if self.mpi_rank == 0:
            n_pass_bool = self.bins['num_pass_full'].astype(bool)
            
        # Treat bins on rank 0 separately, send/recv same rank buggy.
        for rank in xrange(1, size):
                            
            bidxs = bidx_per_rank[rank]

            if self.mpi_rank == rank:
                first_pass_per_bidx = [[] for i in bidxs]
                num_triplets_per_bidx = np.zeros(bidxs.size, dtype=int)
                bispec_per_bidx = [[] for i in bidxs]

            for idx, bidx in enumerate(bidxs):
                b_start, b_stop = tools.get_slice(bidx, num_bins)
                
                if self.mpi_rank == 0:

                    f_pass = self.bins['first_pass_full']
                    num_pass = self.bins['num_pass_full']                    
                    bispec = self.bispec['bispec']

                    # Only use valid tuples.
                    mask = n_pass_bool[b_start:b_stop,:bidx_end,:bidx_end]

                    # Select data
                    f_pass_send = f_pass[b_start:b_stop,:bidx_end,:bidx_end][mask]
                    bispec_send = bispec[b_start:b_stop,:bidx_end,:bidx_end][mask]
                    num_triplets = np.sum(num_pass[bidx,:bidx_end,:bidx_end])

                    # First send meta-data.
                    array_size = np.sum(mask)
                    self._comm.send(array_size, dest=rank, tag=rank)

                    # Then actual data
                    self._comm.Send(f_pass_send, dest=rank, tag=rank + size)
                    self._comm.Send(bispec_send, dest=rank, tag=rank + 2 * size)
                    self._comm.send(num_triplets, dest=rank, tag=rank + 3 * size)

                if self.mpi_rank == rank:

                    # Receive meta-data
                    array_size = self._comm.recv(source=0, tag=rank)

                    # Allocate empty arrays and receive data.
                    first_pass_rec = np.empty((array_size,3), dtype=int)
                    bispec_rec = np.empty((array_size,nptr), dtype=float)

                    self._comm.Recv(first_pass_rec, source=0,
                                    tag=rank + size)
                    self._comm.Recv(bispec_rec, source=0,
                                    tag=rank + 2 * size)
                    num_triplets = self._comm.recv(source=0,
                                    tag=rank + 3 * size)

                    first_pass_per_bidx[idx] = first_pass_rec
                    num_triplets_per_bidx[idx] = num_triplets
                    bispec_per_bidx[idx] = bispec_rec
                    
        # Do the same without send/recv for rank 0.
        if self.mpi_rank == 0:                    

            bidxs = bidx_per_rank[0]            

            first_pass_per_bidx = [[] for i in bidxs]
            num_triplets_per_bidx = np.zeros(bidxs.size, dtype=int)
            bispec_per_bidx = [[] for i in bidxs]

            for idx, bidx in enumerate(bidxs):

                b_start, b_stop = tools.get_slice(bidx, num_bins)
                mask = n_pass_bool[b_start:b_stop,:bidx_end,:bidx_end]

                f_pass = self.bins['first_pass_full'][b_start:b_stop,:bidx_end,:bidx_end][mask]
                num_pass = self.bins['num_pass_full'][bidx,:bidx_end,:bidx_end]
                bispec = self.bispec['bispec'][b_start:b_stop,:bidx_end,:bidx_end][mask]

                num_triplets = np.sum(num_pass)

                first_pass_per_bidx[idx] = f_pass
                num_triplets_per_bidx[idx] = num_triplets
                bispec_per_bidx[idx] = bispec

        return first_pass_per_bidx, bispec_per_bidx, num_triplets_per_bidx
    
    def _init_triplets(self, bidx, num_triplets, bidx_max=None):
        '''
        Return array of all good (l1,l2,l3) triplest given
        outer bin index.
        
        Arguments
        ---------
        bidx : int
            Bin index.
        num_triplets : int
            Number of good (l1,l2,l3) triplets in this outer bin.

        Keyword arguments
        -----------------
        bidx_max : int, None
            Index of last bin. If None, use all bins.
            (default : None)

        Returns
        -------
        triplets : ndarray
            (l1,l2,l3) triplets, shape : (num_triplets, 3)
        '''

        good_triplets = np.zeros((num_triplets, 3), dtype=int)
        
        if bidx_max is not None:
            bins = self.bins['bins'][:bidx_max + 1]
            try:
                lmax = self.bins['bins'][bidx_max + 1] - 1
            except IndexError:
                lmax = self.bins['lmax']
        else:
            bins = self.bins['bins']
            lmax = self.bins['lmax']

        bmin = bins[bidx]
        try:
            bmax = bins[bidx + 1] - 1
        except IndexError:
            bmax = lmax

        parity = self.bins['parity']
        if parity == 'odd':
            pmod = 1
        elif parity == 'even':
            pmod = 0
        else:
            pmod = 2

        tools.get_good_triplets(bmin, bmax, lmax, good_triplets, pmod)
        
        return good_triplets

    def get_bispec_slice(self, ell, verbose=False):
        '''
        Return interpolated B_{ell,...} slice.

        Arguments
        ---------
        ell : int
            l1 multipole.

        Keyword arguments
        -----------------
        verbose : bool, int
            If True or integer, print stuff.

        Returns
        -------
        slice : ndarray
            Shape = (M, nptrls)
        doublets : ndarray
            (l2, l3) doublets.
        '''

        # How to do MPI? do everything on root!
        if self.mpi_rank == 0:
            rank = self.mpi_rank # For print messages.
            bins = self.bins['bins']
            nptr = self.bispec['pol_trpl'].shape[0]
            lmax = bins[-1]
        
            # Find bidx that has ell
            bidx = tools.ell2bidx(ell, bins)

            # Get slices of bispectrum etc. for this bidx.
            b_start, b_stop = tools.get_slice(bidx, len(bins))
            n_pass_bool = self.bins['num_pass_full'].astype(bool)
            mask = n_pass_bool[b_start:b_stop,...]
            f_pass = self.bins['first_pass_full'][b_start:b_stop,...][mask]
            num_pass = self.bins['num_pass_full'][bidx,...]
            bispec = self.bispec['bispec'][b_start:b_stop,...][mask]

            bispec = bispec * self.common_amp # Make copy here to be safe.
            num_triplets = np.sum(num_pass)

            if num_triplets == 0 and bidx == bins.size - 1:
                raise ValueError("No valid triplets in last bin.")

            triplets = self._init_triplets(bidx, num_triplets)

            # Select all triplets with l1 that are ell.
            m = triplets[:,0] == ell
            triplets = triplets[m] # Copy, still has right shape.

            b_i, nb_frac, qh_exit = self._interp_b(triplets, f_pass, bispec)

            if not qh_exit and verbose:
                print('[rank {:03d}]: Completely switching to nearest-neighbor'
                    ' interpolation for bidx {}, (bin = {}, lmax = {})'.format(
                    rank, bidx, bins[bidx], lmax))
            if verbose == 2:
                print('[rank {:03d}]: Used nearest-neighbor for {:.2f}% '
                  'of triplets for bidx {}, (bin = {}, lmax = {})'.format(
                  rank, nb_frac * 100, bidx, bins[bidx], lmax))
    
            doublets = triplets[:,1:]
            
            return b_i, doublets

        else:
            # Other ranks.
            return None, None

    def _interp_b(self, triplets, fp_for_bidx, b_for_bidx):
        '''
        Use linear interpolation to interpolate bispectrum
        onto array of (l1, l2, l3) triplets.

        Arguments
        ---------
        triplets : ndarray
            (l1,l2,l3) triplets, shape : (N, 3).
        fp_for_bidx : ndarray
            first_pass array, shape = (M, 3).
        b_for_bidx : ndarray
            Bispectrum, shape = (M, nptr).
        
        Returns
        -------
        b_i : ndarray
            Interpolated bispectrum, shape (N, nptr).
        nb_frac : float
            Fraction of points obtained by nearest-neighbor interpolation.
        qh_exit : bool
            Whether or not Qhull exited normally.

        Notes
        -----
        Nearest-neighbor interpolation is used for triplets
        outside convex hull of provided samples.
        '''

        num_triplets = triplets.shape[0]
        pol_triplets = self.bispec['pol_trpl']
        nptr = pol_triplets.shape[0]

        # Allocate array for interpolated bispectrum.
        b_i = np.zeros((num_triplets, nptr), dtype=float)

        # Compute weights, can be used for all pol triplets.
        points = fp_for_bidx
        xi = triplets
        qh_exit = True
        try:
            vertices, weights = tools.get_interp_weights(points, xi, 
                                                   fill_value=np.nan)
        except qhull.QhullError as e:

            if 'QH6154' in str(e):
                # Expected Qhull error for last bin in cases where
                # lmax is smaller than the first bin with width > 1.
                weights = np.ones((xi.shape[0], 4), dtype=float) 
                vertices = np.ones((xi.shape[0], 4), dtype=int) 
                weights *= np.nan # I.e. use nearest-neighbor for all.
                qh_exit = False
            elif 'QH6214' in str(e):
                # Expected Qhull error for very low lmax.
                weights = np.ones((xi.shape[0], 4), dtype=float) 
                vertices = np.ones((xi.shape[0], 4), dtype=int) 
                weights *= np.nan # I.e. use nearest-neighbor for all.
                qh_exit = False
            else:
                # Some other Qhull error that is unexpected.
                raise

        # We need to check for nans, i.e xi that lie outsize
        # convex hull of points. Use nearest neighbor for those.
        if tools.has_nan(weights):
            interp_nearest = True
            nanmask = np.isnan(weights[:,0])
        else:
            interp_nearest = False

        nb_frac = 0.
        for pidx, pol_triplet in enumerate(pol_triplets):

            # Interpolate B
            b_c = np.ascontiguousarray(b_for_bidx[:,pidx])
            b_i_p = tools.interpolate(b_c, vertices, weights)

            if interp_nearest:
                # Second pass with nearest neighbour for edge values.
                b_i_temp = griddata(points, b_c, xi, method='nearest')
                b_i_p[nanmask] = b_i_temp[nanmask]

                if pidx == 0:
                    # Add number of NB interpolated points,
                    # (it is the same for all pol_triplets).
                    nb_frac += np.sum(nanmask) / float(nanmask.size)

            b_i[:,pidx] = b_i_p

        return b_i, nb_frac, qh_exit

    def interp_fisher(self, invcov, ells, lmin=None, lmax=None, lmax_outer=None,
                      fsky=1, verbose=True):
        '''
        Calculate Fisher information by interpolating 
        bispectra before squaring.

        Arguments
        ---------
        invcov : array-like
            Inverse convariance, shape (ells, 3, 3), 
            see `get_invcov`.
        ells : array-like
            Multipoles corresponding to invcov.
        
        Keyword Arguments
        -----------------
        lmin : int, None
            Lower limit of Fisher loop. If None,
            use first bin. (default : None)
        lmax : int, None
            Upper limit of Fisher loop. If None,
            use last bin. (default : None)
        lmax_outer : int, None
            Upper limit of ell1 in fisher loop.
        fsky : float
            Fraction of sky assumed to be used.
            (default : 1)
        verbose : bool, int
            If True or integer, print stuff.

        Returns
        -------
        fisher : float
            Fisher information.        

        Notes
        -----
        Because ell1 <= ell2 <= ell3 in Fisher loop and because
        tensor transfer functions go to zero for ell > 200, the 
        lmax_outer parameter can be set to e.g. 600 to skip
        computing the Fisher information for values of ell1 that
        would not have contributed in any case.
        '''

        bins = self.bins['bins']
        bidx_max = np.where(bins <= lmax)[0][-1] 
        bins = bins[:bidx_max + 1] 
        nptr = self.bispec['pol_trpl'].shape[0]

        # Change invcov to K to avoid overflow errors.
        invcov = invcov.copy()
        invcov *= 1e-12
        
        # Check input.
        if ells.size != invcov.shape[0]:
            raise ValueError(
             'ells.size = {}, invcov.shape = {} not compatible'.format(
                 ells.size, invcov.shape))

        if lmin is None:
            lmin = bins[0]
        if lmax is None:
            lmax = bins[-1]

        # Check if we have inv. covariance for ells in requested range.
        if lmin < ells[0]:
            raise ValueError('lmin = {} < ells[0] = {}'.format(
                lmin, ells[0]))
        if lmax > ells[-1]:
            raise ValueError('lmax = {} > ells[-1] = {}'.format(
                lmax, ells[-1]))

        # Note, no call to self.bins in _init_fisher_invcov.
        invcov1, invcov2, invcov3 = self._init_fisher_invcov(invcov)
        lmin_c = ells[0]
        lmax_c = ells[-1]

        if self.mpi_rank == 0:
            # Decide how to distribute load over MPI ranks.
            ells_fisher = np.arange(2, lmax + 1)
            sizes_bispec = tools.estimate_n_tuples(ells_fisher, lmax)        

            bidx_sorted, n_per_bin = tools.rank_bins(
                bins, sizes_bispec, ells_fisher, lmax_outer=lmax_outer)

            bidx_per_rank = tools.distribute_bins_simple(bidx_sorted,
                                      n_per_bin, self.mpi_size)

        else:
            bidx_per_rank = None

        bidx_per_rank = self.broadcast(bidx_per_rank)
        bins_on_rank = bins[bidx_per_rank[self.mpi_rank]]

        if verbose:
            for rank in xrange(self.mpi_size):
                if self.mpi_rank == rank:
                    print('[rank {:03d}]: working on bins {}'.format(
                        rank, bins_on_rank))

        # Note: fp = first_pass, b = bispec. Lists of arrays per bidx.
        fp_per_bidx, b_per_bidx, n_trpl_per_bidx = self._mpi_scatter_for_fisher(
            bidx_per_rank, bidx_max=bidx_max)

        fisher_on_rank = 0
        for idx, bidx in enumerate(bidx_per_rank[self.mpi_rank]):
            num_triplets = n_trpl_per_bidx[idx]

            if num_triplets == 0 and bidx == bins.size - 1:
                # Last bin has no valid tuples. Just skip.
                # Perhaps give 0 contribution to fisher?
                if verbose == 2:
                    print('[rank {:03d}]: Outer bin without '
                     'valid triplets (bin = {}, lmax = {})'.format(
                         self.mpi_rank, bins[bidx], lmax))
                continue

            triplets = self._init_triplets(bidx, num_triplets, bidx_max=bidx_max)
            b_i, nb_frac, qh_exit = self._interp_b(triplets, 
                            fp_per_bidx[idx], b_per_bidx[idx])

            if not qh_exit and verbose:
                print('[rank {:03d}]: Completely switching to nearest-neighbor'
                      ' interpolation for bidx {}, (bin = {}, lmax = {})'.format(
                          self.mpi_rank, bidx, bins[bidx], lmax))
            if verbose == 2:
                print('[rank {:03d}]: Used nearest-neighbor for {:.2f}% '
                      'of triplets for bidx {}, (bin = {}, lmax = {})'.format(
                          self.mpi_rank, nb_frac * 100, bidx, bins[bidx], lmax))

            # Compute Fisher information.
            fisher = tools.fisher_loop(b_i, triplets, 
                                       invcov1, invcov2, invcov3,
                                       lmin_c, lmax_c)

            fisher *= 1e36 # Converting invcov back from K to uK
            fisher *= fsky
            fisher *= self.common_amp ** 2 # (16 pi^4 As^2)^2
            if verbose == 2:
                print('[rank {:03d}]: fisher = {} for bidx {}, '
                      '(bin = {}, lmax = {})'.format(self.mpi_rank, fisher,
                                                    bidx, bins[bidx], lmax))
            fisher_on_rank += fisher

        if self.mpi:
            fisher = self._comm.allreduce(fisher_on_rank)
        else:
            fisher = fisher_on_rank
        self.barrier()

        return fisher

    def save_fisher(self, fisher, r=0, tag=None):
        '''
        Write Fisher information and used options to disk.
        
        Arguments
        ---------
        fisher : float

        Keyword arguments
        -----------------
        r : float
            Tensor-to-scalar ratio. Used for fnl x sqrt(r)
        tag : str, None
            If str, write f_<tag>.pkl. (default : None)

        Notes
        -----
        Results are written to `fisher` subdir as pickle files.
        '''

        
        if self.mpi_rank == 0:

            outdir = self.subdirs['fisher']

            sigma_fnl = 1 / np.sqrt(fisher)
            sigma_fnl_sqrtr = sigma_fnl * np.sqrt(r)

            fisher_opts = dict(fisher=fisher,
                               sigma_fnl=sigma_fnl,
                               sigma_fnl_sqrtr=sigma_fnl_sqrtr)

            if tag is None:
                fisher_file = opj(outdir, 'f.pkl')
            else:
                fisher_file = opj(outdir, 'f_{}.pkl'.format(tag))

            # Store in pickle file.            
            with open(fisher_file, 'wb') as handle:
                pickle.dump(fisher_opts, handle,
                            protocol=pickle.HIGHEST_PROTOCOL)

        # Other ranks can chill for a bit here.
        self.barrier()

        return

'''
A collection of plotting scripts
'''

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import os
import numpy as np

def cls_matrix(filename, ells, cls, plot_dell=True, log=True, lmin=2,
               lmax=None, inv=False, **kwargs):
    '''
    Plot matrix of power spectra.

    Arguments
    ---------
    filename : str
    ells : array-like
        Array of multipoles
    cls : array-like
        Power spectra, shape=(ells.size, 3, 3)    

    Keyword arguments
    -----------------
    log : bool
        Use log y-axis. (efault : True)
    plot_dell : bool
        Plot D_ell instead of C_ell. (default : True)
    lmin : int
        Minimum multipole plotted. (default : 2)
    lmax : int, None
        Maximum multipole plotted. (default : None)
    kwargs : {pyplot.savefig opts}
    '''

    if log:
        raise NotImplementedError("no log yet")

    if plot_dell:
        dell = ells * (ells + 1) / 2. /np.pi
        
    if lmin is not None:
        ells = ells[ells >= lmin]
        cls = cls[ells >= lmin,...]

    if lmax is not None:
        ells = ells[ells <= lmax]
        cls = cls[ells <= lmax,...]

    if inv is False:
        cls[cls > 1e6] = np.nan

    fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(9,7))    

    for pidx1, pol1 in enumerate(['T', 'E', 'B']):
        for pidx2, pol2 in enumerate(['T', 'E', 'B']):
            if plot_dell:
                axs[pidx1,pidx2].plot(ells, dell * cls[:,pidx1,pidx2],
                                  label=pol1+pol2)
            else:
                axs[pidx1,pidx2].plot(ells, cls[:,pidx1,pidx2],
                                  label=pol1+pol2)


    for ax in axs.reshape(-1):
        ax.legend()

    fig.tight_layout()
    fig.savefig(filename, **kwargs)
    plt.close(fig)


def plot_bispec_slice(filename, doublets, bispec, pol_trpl,
                      plot_lmin=None, plot_lmax=None):
    '''
    Plot (l2, l3) bispectrum slices.

    Arguments
    ---------
    filename : str
        Output filename
    doublets : ndarray
        (l2, l3) doublets array, shape: (N, 2)
    bispec : ndarray
        Shape: (N, M)
    pol_triplets
        Shape: (M, 3)
    
    Keyword arguments
    -----------------
    plot_lmin : int, None
        Minumum multipole plotted. If None default
        use minimum value in doublets.
    plot_lmax : int, None
        Maximum multipole plotted. If None default
        use maximum value in doublets.    
    '''

    fig, axs = plt.subplots(nrows=4, ncols=3, figsize=(9.,9),
                            sharex=True, sharey=True)

    marker = matplotlib.markers.MarkerStyle(marker='s')
    cmap = plt.get_cmap(name='RdBu_r')
    Rectangle = matplotlib.patches.Rectangle
    PatchCollection = matplotlib.collections.PatchCollection
    rec_opts = dict(width=1, height=1)

    patches = []
    for i in xrange(doublets.shape[0]):
        patches.append(Rectangle(xy=(doublets[i,0], doublets[i,1]),               
                                 **rec_opts))

    pstr = ['I', 'E', 'B']

    for pidx, pol in enumerate(pol_trpl):

        axs_idx = np.divmod(pidx,3)
        ax = axs[axs_idx]

        collection = PatchCollection(patches, cmap=cmap)
        collection.set_array(bispec[:,pidx])
        collection.set_figure(fig)

        ax.add_collection(collection)
        ax.set_xlim(plot_lmin, plot_lmax)
        ax.set_ylim(plot_lmin, plot_lmax)
        
        ratio = 1
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        ax.set_aspect(abs((xmax-xmin)/(ymax-ymin))*ratio, adjustable='box-forced')

        cb = fig.colorbar(collection, ax=ax)
        cb.formatter.set_powerlimits((0, 0))
        cb.update_ticks()

        p1 = pol[0]
        p2 = pol[1]
        p3 = pol[2]
        pol_str = pstr[p1] + pstr[p2] + pstr[p3]
        pol_str = r'$B^{'+pol_str+'}$'
        
        ax.text(0.05, 0.9, pol_str, transform=ax.transAxes)

    fig.tight_layout()    
    fig.savefig(filename, dpi=150)
    
        

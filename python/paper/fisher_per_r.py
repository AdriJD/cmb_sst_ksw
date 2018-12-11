'''
Calculate and plot Fisher information over subsets of r.
'''

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.colors as colors

import cProfile, pstats

import sys
import os
import numpy as np
from scipy.special import spherical_jn
from sst import Fisher
from sst import camb_tools as ct

opj = os.path.join

def get_cls(cls_path, lmax, A_lens=1, tag=''):
    '''
    returns
    -------
    cls : array-like
        Lensed Cls (shape (4,lmax-1) with BB lensing power
        reduced depending on A_lens.
        order: TT, EE, BB, TE
    '''

    cls_nolens, _ = ct.get_spectra(cls_path, tag=tag,
                             lensed=False, prim_type='tot')
    cls_lensed, _ = ct.get_spectra(cls_path, tag=tag,
                             lensed=True, prim_type='tot')

    # truncate to lmax
    cls_nolens = cls_nolens[:,:lmax-1]
    cls_lensed = cls_lensed[:,:lmax-1]

    BB_nolens = cls_nolens[2]
    BB_lensed = cls_lensed[2]

    # difference BB (lensed - unlensed = lens_contribution)
    BB_lens_contr = BB_lensed - BB_nolens[:BB_lensed.size]

    # depending on A_lens, remove lensing contribution
    cls_lensed[2] -= (1. - A_lens) * BB_lens_contr

    return cls_lensed

def get_prim_amp(prim_template='local', scalar_amp=2.1e-9):

    common_amp =  16 * np.pi**4 * scalar_amp**2

    if prim_template == 'local':
        return 2 * common_amp

    elif prim_template == 'equilateral':
        return 6 * common_amp

    elif prim_template == 'orthogonal':
        return 6 * common_amp

def run(out_dir, num_chunks, camb_opts=None,
               bin_opts=None,
               beta_opts=None):

    F = Fisher(out_dir)

    F.get_camb_output(**camb_opts)

    cls = get_cls(camb_opts['camb_out_dir'], bin_opts['lmax'],
                  A_lens=1, tag=camb_opts['tag'])
    lmax = cls.shape[1] + 2

    # hacky stuff
#    totcov = np.zeros((6, cls.shape[1]))
    totcov = np.zeros((6, 3999))
    totcov[:3] += 1e12 # fill diagonal
    totcov[:4,:cls.shape[1]] = cls

    F.get_bins(**bin_opts)

    radii = F.get_updated_radii()

    F.get_beta(radii=radii, **beta_opts)

    chunks = np.array_split(radii, num_chunks)
    fisher_r = np.zeros(len(chunks))
    for cidx, chunk in enumerate(chunks):
        F.get_binned_bispec('equilateral', radii_sub=chunk,
                            load=True, tag=str(cidx))

        print(F.bispec['bispec'].size)
        print(F.bispec['bispec'][F.bins['num_pass'].astype(bool)].size)
        exit()

        amp = get_prim_amp('equilateral')
        F.bispec['bispec'] *= amp
        fisher, _ = F.naive_fisher(2, lmax, totcov, fsky=1)
        fisher_r[cidx] = fisher
        print(fisher)

    # save in fisher
    np.save(opj(F.subdirs['fisher'], 'fisher_r.npy'), fisher_r)
    np.save(opj(F.subdirs['fisher'], 'chunks.npy'), chunks)
    return fisher_r, chunks

def plot_fisher_r(outdir):
    
    fisher_r = np.load(opj(outdir, 'fisher', 'fisher_r.npy'))
    chunks = np.load(opj(outdir, 'fisher', 'chunks.npy'))
        
    mean_r = [np.mean(chunk) for chunk in chunks]
    width_r = [chunk.max() - chunk.min() for chunk in chunks]
    
    fig, axs = plt.subplots()
    axs.plot(mean_r, fisher_r, markersize=2)
    axs.set_xlabel(r'Comoving radius $r$ [$Mpc$]')
    axs.set_ylabel(r'$(\mathrm{S}/\mathrm{N})^2$')
#    axs.bar(mean_r, fisher_r, width_r)
#    axs.set_xlim(11000, 17000)
   
    axs.set_yscale('log')
    fig.savefig(opj(outdir, 'fisher', 'fisher_r.png'))

if __name__ == '__main__':

    base_dir = '/mn/stornext/d8/ITA/spider/adri/analysis/'

    out_dir = opj(base_dir, '20181211_sst_fisher_r')
    camb_dir = opj(base_dir, '20180911_sst/camb_output/lensed_r0_4000')

    num_chunks = 20
    camb_opts = dict(camb_out_dir = camb_dir,
                     tag='',
                     lensed=False,
                     high_ell=False,
                     interp_factor=None)

    bin_opts = dict(lmin=2, lmax=4000, load=True,
                    parity='odd', verbose=True)

    beta_opts = dict(func='equilateral', verbose=True,
                     optimize=True, interp_factor=None,
                     load=True, sparse=True)

    fisher_r, chunks = run(out_dir, num_chunks, bin_opts=bin_opts,
        beta_opts=beta_opts, camb_opts=camb_opts)
#    plot_fisher_r(out_dir)

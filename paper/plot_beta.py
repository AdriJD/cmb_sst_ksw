'''
Plot output from calc_beta.
'''

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, MaxNLocator, ScalarFormatter, FuncFormatter

import numpy as np
import os

matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

opj = os.path.join

class ScalarFormatterForceFormat(ScalarFormatter):
    def _set_format(self):
        self.format = '%1d'


def plot_gen_alpha(beta_dir, img_dir, ell,
                   rmin=12900, rmax=14600):
    '''
    Plot generalized alpha as function of r.

    Arguments
    ---------
    beta_dir : str
       Directory containing beta .pkl files.
    img_dir : str
       Output directory
    ell : int
        Central multipole.
    
    Keyword arguments
    -----------------
    rmin, rmax : scalar
        Min max r for plot.    
    '''


    linestyles = ['--', '-', '-.']
    alphas = [0.6, 1, 0.6]
    colors = ['C1', 'C0', 'C2']

    lidx = ell - 2
    xlim = [rmin, rmax]

    # Scalar.
    beta_tag_s = 'r1_i1_l5200_16_7'

    beta_file = 'beta_{}.pkl'.format(beta_tag_s)
    beta = np.load(opj(beta_dir, beta_file))
    beta_s = beta['beta_s']
    radii = beta['radii']

    ridx = radii.size - 1
    lmax = beta_s.shape[0] + 1
    ells = np.arange(2, lmax+1)
    dell = ells * (ells + 1) / 2. / np.pi

    ridxs = np.arange(0, radii.size, 10)

    L_range = [-1, 0, 1]

    fig, axs = plt.subplots(ncols=1, nrows=2, sharey=False, sharex=True, 
                            figsize=(4, 3), squeeze=False)
    for pidx, pol in enumerate(['T', 'E']):
        for Lidx, eLL in enumerate(L_range):

            ls = linestyles[Lidx]
            alpha = alphas[Lidx]

            plot_opts = dict(ls=ls, alpha=alpha)
            if eLL == 0:
                label = r'$ \ \ \: '+'{0:d}'.format(eLL)+'$'
            else:
                label = r'$'+'{0:+d}'.format(eLL)+'$'
            axs[pidx,0].plot(radii, radii ** (2) * beta_s[lidx,eLL+2,1,pidx,:],
                             label=label, 
                             color='C0', **plot_opts)
        axs[pidx,0].set_xlim(xlim)
        axs[pidx,0].text(0.83, 0.77, r'$X='+pol+'$', transform=axs[pidx,0].transAxes)

    fig.text(0.01, 0.5,
             r'$r^{2} \, \mathcal{K}[1]_{\ell, L} \ [\mu \mathrm{K}\ \mathrm{Mpc}^{-1}]$',
             ha='center', va='center', rotation='vertical')

    axs[1,0].set_xlabel(r'comoving radial distance $r$  [$\mathrm{Mpc}]$')
    axs[0,0].set_title(r'$Z = \zeta$')
    axs[0,0].legend(ncol=1, title=r'$L - \ell\ (\ell = '+str(ell)+')$', frameon=True,
                    markerscale=0.5, handletextpad=0.3, handlelength=2)
    fig.subplots_adjust(hspace=0, wspace=None)
    for ax in axs.flatten():
        ax.tick_params(axis='both', direction='in', top=True, right=True)
        ax.ticklabel_format(axis='y', style='sci', scilimits=(2,2),
                             useMathText=True)
    fig.savefig(opj(img_dir, 'alpha_scal_{}.pdf'.format(ell)),
                dpi=300, bbox_inches='tight')
    plt.close(fig)


    # Tensor I, E.
    alpha = alphas[0]
    alpha = 0.9
    alphas = [1, 1, 1]

    beta_tag_ti = 'r1_i1_l2000_16_3'
    beta_file = 'beta_{}.pkl'.format(beta_tag_ti)
    beta = np.load(opj(beta_dir, beta_file))
    radii = beta['radii']
    beta_ti = beta['beta_t']
    lmax = beta_ti.shape[0] + 1

    beta_tag_te = 'r1_i1_l2000_16_5'
    beta_file = 'beta_{}.pkl'.format(beta_tag_te)
    beta = np.load(opj(beta_dir, beta_file))
    beta_te = beta['beta_t']

    ridx = radii.size - 1
    ells = np.arange(2, lmax+1)
    dell = ells * (ells + 1) / 2. / np.pi

    ridxs = np.arange(0, radii.size, 10)

    L_range = [-2, 0, 2]

    fig, axs = plt.subplots(ncols=1, nrows=2, sharey=False, sharex=True, 
                            figsize=(4, 3), squeeze=False)
    for pidx, pol in enumerate(['T', 'E']):
        for Lidx, eLL in enumerate(L_range):

            ls = linestyles[Lidx]
            alpha = alphas[Lidx]

            plot_opts = dict(ls=ls, alpha=alpha)
            if eLL == 0:
                label = r'$ \ \ \: '+'{0:d}'.format(eLL)+'$'
            else:
                label = r'$'+'{0:+d}'.format(eLL)+'$'
            if pol == 'T':
                beta_plot = beta_ti
            if pol == 'E':
                beta_plot = beta_te

            axs[pidx,0].plot(radii, radii ** (2) * beta_plot[lidx,eLL+2,1,pidx,:],
                             label=label, 
                             color='C1', **plot_opts)
        axs[pidx,0].set_xlim(xlim)
        axs[pidx,0].text(0.83, 0.77, r'$X='+pol+'$', transform=axs[pidx,0].transAxes)

    fig.text(0.01, 0.5,
             r'$r^{2} \, \mathcal{K}[1]_{\ell, L} \ [\mu \mathrm{K}\ \mathrm{Mpc}^{-1}]$',
             ha='center', va='center', rotation='vertical')

    axs[1,0].set_xlabel(r'comoving radial distance $r$  [$\mathrm{Mpc}]$')
    axs[0,0].set_title(r'$Z = h$')
    axs[0,0].legend(ncol=1, title=r'$L - \ell\ (\ell = '+str(ell)+')$', frameon=True,
                    markerscale=0.5, handletextpad=0.3, handlelength=2)
    fig.subplots_adjust(hspace=0, wspace=None)
    for ax in axs.flatten():
        ax.tick_params(axis='both', direction='in', top=True, right=True)
        ax.ticklabel_format(axis='y', style='sci', scilimits=(2,2),
                             useMathText=True)
    fig.savefig(opj(img_dir, 'alpha_tens_ie_{}.pdf'.format(ell)),
                dpi=300, bbox_inches='tight')
    plt.close(fig)

    # Tensor B.
    alphas = [1, 1]
    linestyles = ['--', '-.']

#    beta_tag_tb = 'r1_i1_l1500_16_5'
    beta_tag_tb = 'r1_i1_l2000_16_5'
    beta_file = 'beta_{}.pkl'.format(beta_tag_tb)
    beta = np.load(opj(beta_dir, beta_file))
    radii = beta['radii']
    beta_tb = beta['beta_t']
    lmax = beta_ti.shape[0] + 1

    ridx = radii.size - 1
    ells = np.arange(2, lmax+1)
    dell = ells * (ells + 1) / 2. / np.pi

    ridxs = np.arange(0, radii.size, 10)

    L_range = [-1, 1]

    pidx = 2
    pol = 'B'
    fig, axs = plt.subplots(ncols=1, nrows=1, sharey=False, sharex=True, 
                            figsize=(4, 1.5), squeeze=False)
    for Lidx, eLL in enumerate(L_range):

        ls = linestyles[Lidx]
        alpha = alphas[Lidx]

        plot_opts = dict(ls=ls, alpha=alpha)
        if eLL == 0:
            label = r'$ \ \ \: '+'{0:d}'.format(eLL)+'$'
        else:
            label = r'$'+'{0:+d}'.format(eLL)+'$'

        axs[0,0].plot(radii, radii ** (2) * beta_tb[lidx,eLL+2,1,pidx,:],
                         label=label, 
                         color='C1', **plot_opts)
        axs[0,0].set_xlim(xlim)
        axs[0,0].text(0.83, 0.77, r'$X='+pol+'$', transform=axs[0,0].transAxes)

    fig.text(0.01, 0.5,
             r'$r^{2} \, \mathcal{K}[1]_{\ell, L} \ [\mu \mathrm{K}\ \mathrm{Mpc}^{-1}]$',
             ha='center', va='center', rotation='vertical')

    axs[0,0].set_xlabel(r'comoving radial distance $r$  [$\mathrm{Mpc}]$')
    axs[0,0].set_title(r'$Z = h$')
    axs[0,0].legend(ncol=1, title=r'$L - \ell\ (\ell = '+str(ell)+')$', frameon=True,
                    markerscale=0.5, handletextpad=0.3, handlelength=2)
    fig.subplots_adjust(hspace=0, wspace=None)
    for ax in axs.flatten():
        ax.tick_params(axis='both', direction='in', top=True, right=True)
        ax.ticklabel_format(axis='y', style='sci', scilimits=(2,2),
                             useMathText=True)
    fig.savefig(opj(img_dir, 'alpha_tens_b_{}.pdf'.format(ell)),
                dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_alpha_beta(beta_dir, img_dir, ell, beta_tag=None):
    '''
    Plot generalized alpha and beta.
    
    Arguments
    ---------
    beta_dir : str
       Directory containing beta_<beta_tag>.pkl files.
    img_dir : str
       Output directory
    ell : int
        Multipole to plot.

    Keyword arguments
    -----------------
    beta_tag : str, None
        If str, look for beta_<beta_tag>.pkl files.

    '''


    if not os.path.exists(img_dir):
        os.makedirs(img_dir)

    if beta_tag is None:
        beta_file = 'beta.pkl'
    else:
        beta_file = 'beta_{}.pkl'.format(beta_tag)

    beta = np.load(opj(beta_dir, beta_file))
    beta_s = beta['beta_s']
    beta_t = beta['beta_t']
    radii = beta['radii']

    lidx = ell - 2

    ridx = radii.size - 1
    lmax = beta_s.shape[0] + 1
    ells = np.arange(2, lmax+1)
    dell = ells * (ells + 1) / 2. / np.pi

    ridxs = np.arange(0, radii.size, 10)
    grays = np.linspace(0, 0.7, num=ridxs.size)

    linestyles = ['--', '-', '-.']
    alphas = [0.4, 1, 0.4]
    colors = ['C1', 'C0', 'C2']

#    xlim = [8000, 15500]
#    xlim = [12000, 15200]
    xlim = [12500, 15000]
#    xlim = [0, 25000]
    #xlim = None

#    L_range = [-1, 0, 1]
    L_range = [-2, 0, 2]
    for pidx in [0, 1, 2]:

        # Alpha and beta as function of radius.
        fig, axs = plt.subplots(ncols=2, sharey=False, figsize=(10, 4))
        for Lidx, L in enumerate(L_range):

            L += lidx + 2
            
            ls = linestyles[Lidx]
            alpha = alphas[Lidx]

            plot_opts = dict(ls=ls, alpha=alpha)

            if pidx != 2:
                axs[0].plot(radii, beta_s[lidx,Lidx,0,pidx,:], 
                            label=r'$\ell={'+str(lidx+2)+'}, L='+str(L)+'$',
                            color='C0',
                            **plot_opts)
            axs[1].plot(radii, beta_t[lidx,Lidx,0,pidx,:], 
                        color='C1', **plot_opts)
        axs[0].set_xlabel(r'Comoving radius $r$ [$\mathrm{Mpc}$]')
        axs[1].set_xlabel(r'Comoving radius $r$ [$\mathrm{Mpc}$]')
        axs[0].set_ylabel(r'$\beta_{\ell, L}(r)$')
        axs[0].set_xlim(xlim)
        axs[1].set_xlim(xlim)
        axs[0].set_title('Scalar')        
        axs[1].set_title('Tensor')
        axs[0].legend()
        fig.tight_layout()
        fig.savefig(opj(img_dir, 'beta_pidx{}.png'.format(pidx)), dpi=200)
        plt.close(fig)

    # Scalar
    fig, axs = plt.subplots(ncols=1, nrows=2, sharey=False, sharex=True, 
                            figsize=(4, 3), squeeze=False)
    for pidx, pol in enumerate(['T', 'E']):
        for Lidx, eLL in enumerate(L_range):

            L = eLL + lidx + 2
            
            ls = linestyles[Lidx]
            alpha = alphas[Lidx]

            plot_opts = dict(ls=ls, alpha=alpha)
            if eLL == 0:
                label = r'$ \ \ \: '+'{0:d}'.format(eLL)+'$'
            else:
                label = r'$'+'{0:+d}'.format(eLL)+'$'
            axs[pidx,0].plot(radii, radii ** (2) * beta_s[lidx,eLL+2,1,pidx,:],
                             label=label, 
                             color='C0', **plot_opts)
#        axs[pidx,0].set_ylabel(r'$r^{2} \alpha_{'+pol+',\ell, L}(r)$')
        axs[pidx,0].set_xlim(xlim)

        axs[pidx,0].text(0.85, 0.8, r'$X='+pol+'$', transform=axs[pidx,0].transAxes)

    fig.text(0.01, 0.5, r'$r^{2} \alpha_{\ell, L}(r)$', ha='center', 
             va='center', rotation='vertical')

    axs[1,0].set_xlabel(r'comoving radius $r$  [$\mathrm{Mpc}]$')
    axs[0,0].set_title(r'$Z = \mathcal{R}$')
    axs[0,0].legend(ncol=1, title=r'$\Delta L \ (\ell = '+str(lidx+2)+')$', frameon=True,
                    markerscale=0.5, handletextpad=0.3, handlelength=2)
    fig.subplots_adjust(hspace=0, wspace=None)
    #fig.tight_layout()
    for ax in axs.flatten():
        ax.tick_params(axis='both', direction='in', top=True, right=True)
        ax.ticklabel_format(axis='y', style='sci', scilimits=(2,2),
                             useMathText=True)

    #fig.subplots_adjust(right=0.8)
    fig.savefig(opj(img_dir, 'alpha_scal_{}.pdf'.format(beta_tag)),
                dpi=300, bbox_inches='tight')
    plt.close(fig)


    fig, axs = plt.subplots(ncols=1, nrows=3, sharey=False, sharex=True, 
                            figsize=(4, 5), squeeze=False)
    for pidx, pol in enumerate(['T', 'E', 'B']):
        for Lidx, eLL in enumerate(L_range):

            L = eLL + lidx + 2
            
            ls = linestyles[Lidx]
            alpha = alphas[Lidx]

            plot_opts = dict(ls=ls, alpha=alpha)

            axs[pidx,0].plot(radii, radii ** (2) * beta_t[lidx,eLL+2,1,pidx,:], 
                             color='C1', label=r'$'+'{0:+d}'.format(eLL)+'$', **plot_opts)

        axs[pidx,0].set_ylabel(r'$r^{2} \alpha_{'+pol+',\ell, L}(r)$')
        axs[pidx,0].set_xlim(xlim)

        axs[pidx,0].text(0.85, 0.8, r'$X='+pol+'$', transform=axs[pidx,0].transAxes)
    
    axs[2,0].set_xlabel(r'comoving radius $r$ [$\mathrm{Mpc}$]')
    axs[0,0].set_title(r'$Z = h$')
    axs[0,0].legend(ncol=1, title=r'$\Delta L \ (\ell = '+str(lidx+2)+')$', frameon=True)
    fig.subplots_adjust(hspace=0, wspace=None)
    # fig.tight_layout()
    for ax in axs.flatten():
        ax.tick_params(axis='both', direction='in', top=True, right=True)
        ax.ticklabel_format(axis='y', style='sci', scilimits=(-2,2))
        
        
    fig.savefig(opj(img_dir, 'alpha_tens_{}.pdf'.format(beta_tag)),
                dpi=300) #bbox_inches='tight')
    plt.close(fig)


    # Alpha and beta as function of multipole.
    for pidx in [0, 1, 2]:
        fig, axs = plt.subplots(ncols=2, sharey=False, figsize=(10, 4))
        for Lidx in [2]:
            for ii, ridx in enumerate(ridxs):
                if pidx != 2:
                    axs[0].plot(ells, dell * beta_s[:,Lidx,0,pidx,ridx],
                                color=str(grays[ii]))
                axs[1].plot(ells, dell * beta_t[:,Lidx,0,pidx,ridx],
                            color=str(grays[ii]))
        axs[0].set_xlabel(r'Multipole [$\ell$]')
        axs[1].set_xlabel(r'Multipole [$\ell$]')
        axs[0].set_ylabel(r'$\ell (\ell + 1) \beta_{\ell, L}(r) / (2 \pi)$')
        axs[0].set_title('scalar')
        axs[1].set_title('tensor')
        fig.tight_layout()
        fig.savefig(opj(img_dir, 'beta_ell_pidx{}.png'.format(pidx)), dpi=200)
        plt.close(fig)

        fig, axs = plt.subplots(ncols=2, sharey=False, figsize=(10, 4))
        for Lidx in [2]:
            for ii, ridx in enumerate(ridxs):
                if pidx != 2:                        
                    axs[0].plot(ells, dell * beta_s[:,Lidx,1,pidx,ridx],
                                color=str(grays[ii]))
                axs[1].plot(ells, dell * beta_t[:,Lidx,1,pidx,ridx],
                            color=str(grays[ii]))
        axs[0].set_xlabel(r'Multipole [$\ell$]')
        axs[1].set_xlabel(r'Multipole [$\ell$]')
        axs[0].set_ylabel(r'$ \ell (\ell + 1) \alpha_{\ell, L}(r) / (2 \pi)$')
        axs[0].set_title('scalar')
        axs[1].set_title('tensor')
        fig.tight_layout()
        fig.savefig(opj(img_dir, 'alpha_ell_pidx{}.png'.format(pidx)), dpi=200)
        plt.close(fig)

def plot_alpha_beta_matrix(base_dir, img_dir):
    '''
    Plot alpha, beta as ell x r matrices
    '''

    beta = np.load(opj(beta_dir, 'beta.pkl'))
    beta_s = beta['beta_s']
    beta_t = beta['beta_t']
    radii = beta['radii']
    
    Lidx = 2
    lmax = beta_s.shape[0] + 1

    ell = np.arange(2, lmax+1)
    dell = ell * (ell + 1) / 2. / np.pi

#    beta_s = beta_s.T
#    beta_t = beta_t.T
#    beta_s *= dell
#    beta_t *= dell
#    beta_s = beta_s.T
#    beta_t = beta_t.T

    beta_s *= radii ** 2
    beta_t *= radii ** 2

    imshow_opts = dict(aspect=0.01, interpolation='none', origin='lower', 
                       extent=[2, lmax, radii[0], radii[-1]])

    for pidx in [0, 1, 2]:
        fig, axs = plt.subplots(ncols=2, sharey=False, figsize=(8, 4))
        if pidx != 2:                        
            axs[0].imshow(beta_s[:,Lidx,0,pidx,:].T, **imshow_opts)
        axs[1].imshow(beta_t[:,Lidx,0,pidx,:].T, **imshow_opts)
        fig.tight_layout()
        fig.savefig(opj(img_dir, 'beta_matrix_pidx{}.png'.format(pidx)), dpi=200)
        plt.close(fig)

        fig, axs = plt.subplots(ncols=2, sharey=False, figsize=(8, 4))
        if pidx != 2:                        
            axs[0].imshow(beta_s[:,Lidx,1,pidx,:].T, **imshow_opts)
        axs[1].imshow(beta_t[:,Lidx,1,pidx,:].T, **imshow_opts)
        fig.tight_layout()
        fig.savefig(opj(img_dir, 'alpha_matrix_pidx{}.png'.format(pidx)), dpi=200)
        plt.close(fig)

def alpha_at_r(r, beta_tag=None):
    '''
    Plot alpha(r) as function of ell at given r.
    '''

    if beta_tag is None:
        beta_file = 'beta.pkl'
    else:
        beta_file = 'beta_{}.pkl'.format(beta_tag)

    beta = np.load(opj(beta_dir, beta_file))
    beta_s = beta['beta_s']
    beta_t = beta['beta_t']
    radii = beta['radii']
    ells = beta['ells']

    # Radii are sorted
    ridx = np.searchsorted(radii, r, side="left")
    radius = radii[ridx]
    pidx = 0
    Lidx = 2
    kidx = 1 # alpha

    fig, axs = plt.subplots(ncols=1, sharey=False)
    axs.plot(ells, beta_s[:,Lidx,kidx,pidx,ridx])
    axs.set_xlabel(r'Multipole [$\ell$]')
    axs.set_ylabel(r'$\alpha_{\ell}(r)$')
    axs.set_title('scalar')
    axs.set_xscale('log')
    fig.tight_layout()
    fig.savefig(opj(img_dir, 'alpha_scalar_p{}_L{}_r{}.png'.format(
        pidx, Lidx, radius)))
    plt.close(fig)

    
    # compare to camb
    ana_dir = '/mn/stornext/d8/ITA/spider/adri/analysis/20171217_sst/camb_output/beta/'
    alpha = np.load(opj(ana_dir, 'test__alpha.npy'))
    beta = np.load(opj(ana_dir, 'test__beta.npy'))
    err = np.load(opj(ana_dir, 'test__alpha_beta_r.npy'))

    ridx_c = np.searchsorted(err, r, side="left")
    radius = err[ridx_c]
    ells_c = np.arange(2, alpha.shape[1]+2)

    fig, axs = plt.subplots(ncols=1, sharey=False)
    axs.plot(ells_c, alpha[ridx_c,:])
    axs.set_xlabel(r'Multipole [$\ell$]')
    axs.set_ylabel(r'$\alpha_{\ell}(r)$')
    axs.set_title('scalar')
    axs.set_xscale('log')
    fig.tight_layout()
    fig.savefig(opj(img_dir, 'alpha_camb_scalar_p{}_L{}_r{}.png'.format(
        pidx, Lidx, radius)))
    plt.close(fig)

def alpha_at_ell(ell, beta_tag=None):
    '''
    Plot alpha(r) as function of r at given multipole.
    '''

    if beta_tag is None:
        beta_file = 'beta.pkl'
    else:
        beta_file = 'beta_{}.pkl'.format(beta_tag)

    beta = np.load(opj(beta_dir, beta_file))
    beta_s = beta['beta_s']
    beta_t = beta['beta_t']
    radii = beta['radii']
    ells = beta['ells']

    lidx = np.where(ells==ell)[0][0]
#    pidx = 1
    Lidx = 2
    kidx = 1 # alpha

    for pidx in [0, 1, 2]:
        if pidx < 2:
            fig, axs = plt.subplots(ncols=1, sharey=False)
            axs.plot(radii, beta_s[lidx,Lidx,kidx,pidx,:])
            axs.set_xlabel(r'Comoving radius $r$ [$\mathrm{Mpc}$]')
            axs.set_ylabel(r'$\alpha_{\ell}(r)$')
            axs.set_title('scalar')
            fig.tight_layout()
            fig.savefig(opj(img_dir, 'alpha_scalar_p{}_L{}_l{}.png'.format(
                pidx, Lidx, ell)))
            plt.close(fig)

        fig, axs = plt.subplots(ncols=1, sharey=False)
        axs.plot(radii, beta_t[lidx,Lidx,kidx,pidx,:])
        axs.set_xlabel(r'Comoving radius $r$ [$\mathrm{Mpc}$]')
        axs.set_ylabel(r'$\alpha_{\ell}(r)$')
        axs.set_title('scalar')
        fig.tight_layout()
        fig.savefig(opj(img_dir, 'alpha_tensor_p{}_L{}_l{}.png'.format(
            pidx, Lidx, ell)))
        plt.close(fig)

        # zoom in around recombination
        ridx_low = np.searchsorted(radii, 13000, side="left")
        ridx_hi = np.searchsorted(radii, 15000, side="left")

        if pidx < 2:
            fig, axs = plt.subplots(ncols=1, sharey=False)
            axs.plot(radii[ridx_low:ridx_hi], 
                     beta_s[lidx,Lidx,kidx,pidx,ridx_low:ridx_hi])
            axs.set_xlabel(r'Comoving radius $r$ [$\mathrm{Mpc}$]')
            axs.set_ylabel(r'$\alpha_{\ell}(r)$')
            axs.set_title('scalar')
            fig.tight_layout()
            fig.savefig(opj(img_dir, 'alpha_scalar_zoom_p{}_L{}_l{}.png'.format(
                pidx, Lidx, ell)))
            plt.close(fig)

        fig, axs = plt.subplots(ncols=1, sharey=False)
        axs.plot(radii[ridx_low:ridx_hi], beta_t[lidx,Lidx,kidx,pidx,ridx_low:ridx_hi])
        axs.set_xlabel(r'Comoving radius $r$ [$\mathrm{Mpc}$]')
        axs.set_ylabel(r'$\alpha_{\ell}(r)$')
        axs.set_title('scalar')
        fig.tight_layout()
        fig.savefig(opj(img_dir, 'alpha_tensor_zoom_p{}_L{}_l{}.png'.format(
            pidx, Lidx, ell)))
        plt.close(fig)

    pidx = 0 # CAMB only has scalar T
    # compare to camb
    ana_dir = '/mn/stornext/d8/ITA/spider/adri/analysis/20171217_sst/camb_output/beta/'
    alpha = np.load(opj(ana_dir, 'test__alpha.npy'))
    beta = np.load(opj(ana_dir, 'test__beta.npy'))
    err = np.load(opj(ana_dir, 'test__alpha_beta_r.npy'))

    ells_c = np.arange(2, alpha.shape[1]+2)
    lidx = np.where(ells_c==ell)[0][0]

    fig, axs = plt.subplots(ncols=1, sharey=False)
    axs.plot(err, alpha[:,lidx])
    axs.set_xlabel(r'Comoving radius $r$ [$\mathrm{Mpc}$]')
    axs.set_ylabel(r'$\alpha_{\ell}(r)$')
    axs.set_title('scalar')
    fig.tight_layout()
    fig.savefig(opj(img_dir, 'alpha_camb_scalar_p{}_L{}_l{}.png'.format(
        pidx, Lidx, ell)))
    plt.close(fig)


    # zoom in around recombination
    ridx_low = np.searchsorted(err, 13000, side="left")
    ridx_hi = np.searchsorted(err, 15000, side="left")

    
    fig, axs = plt.subplots(ncols=1, sharey=False)
    axs.plot(err[ridx_low:ridx_hi], alpha[ridx_low:ridx_hi,lidx])
    axs.set_xlabel(r'Comoving radius $r$ [$\mathrm{Mpc}$]')
    axs.set_ylabel(r'$\alpha_{\ell}(r)$')
    axs.set_title('scalar')
    fig.tight_layout()
    fig.savefig(opj(img_dir, 'alpha_camb_scalar_zoom_p{}_L{}_l{}.png'.format(
        pidx, Lidx, ell)))
    plt.close(fig)

if __name__ == '__main__':
    
    base_dir = '/mn/stornext/d8/ITA/spider/adri/analysis/'

#    beta_dir = opj(base_dir, '20180911_sst/beta/equilateral')
#    beta_dir = opj(base_dir, '20180911_sst/beta_sparse_ell/equilateral')
#    beta_dir = opj(base_dir, '20181123_sst/precomputed')
#    img_dir = opj(base_dir, '20181123_sst/img/beta/')
#    beta_dir = opj(base_dir, '20181214_sst_debug/precomputed')
    beta_dir = opj(base_dir, '20190411_beta/precomputed')
#    img_dir = opj(base_dir, '20190411_beta/img/')
    img_dir = opj(base_dir, '20190411_beta/img/img_temperature')

#    plot_alpha_beta(beta_dir, img_dir, 500, beta_tag='r1_i1_l2500_16_8')
    # plot_alpha_beta_matrix(beta_dir, img_dir)
#    alpha_at_r(13306.4, beta_tag='r1_i40_l4000')
#    alpha_at_ell(60, beta_tag='r1_i1_l2000_16_10')
    plot_gen_alpha(beta_dir, img_dir, 60)

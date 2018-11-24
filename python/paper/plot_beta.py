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

def plot_alpha_beta(beta_dir, img_dir):

    if not os.path.exists(img_dir):
        os.makedirs(img_dir)

    beta = np.load(opj(beta_dir, 'beta.pkl'))
    beta_s = beta['beta_s']
    beta_t = beta['beta_t']
    radii = beta['radii']

    lidx = 18

    ridx = radii.size - 1
    lmax = beta_s.shape[0] + 1
    ell = np.arange(2, lmax+1)
    dell = ell * (ell + 1) / 2. / np.pi

    ridxs = np.arange(0, radii.size, 10)
    grays = np.linspace(0, 0.7, num=ridxs.size)

    linestyles = ['--', '-', '-.']
    alphas = [0.4, 1, 0.4]
    colors = ['C1', 'C0', 'C2']

    #xlim = [9000, 15500]
    xlim = [12000, 15200]
    #xlim = None

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
    for pidx, pol in enumerate(['I', 'E']):
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
        #yfmt = ScalarFormatterForceFormat()
        #yfmt.set_powerlimits((0,0))
        #ax.yaxis.set_major_formatter(yfmt)
        #locator = MaxNLocator(integer=True)
        #ax.yaxis.set_major_locator(locator)

#    formatter = ScalarFormatter(useMathText=True)
#    formatter.set_scientific(True) 
#    formatter.set_powerlimits((-2,2))

#    f = ScalarFormatter(useOffset=False, useMathText=True)
#    g = lambda x,pos : "${}$".format(f._formatSciNotation('%1.10e' % x))
#    axs[0,0].yaxis.set_major_formatter(FuncFormatter(g))

    #fig.subplots_adjust(right=0.8)
    fig.savefig(opj(img_dir, 'alpha_scal.pdf'), dpi=300, bbox_inches='tight')
    plt.close(fig)


    fig, axs = plt.subplots(ncols=1, nrows=3, sharey=False, sharex=True, 
                            figsize=(4, 5), squeeze=False)
    for pidx, pol in enumerate(['I', 'E', 'B']):
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
        
        
    fig.savefig(opj(img_dir, 'alpha_tens.pdf'), dpi=300) #bbox_inches='tight')
    plt.close(fig)




    # Alpha and beta as function of multipole.
    for pidx in [0, 1, 2]:
        fig, axs = plt.subplots(ncols=2, sharey=False, figsize=(10, 4))
        for Lidx in [2]:
            for ii, ridx in enumerate(ridxs):
                if pidx != 2:
                    axs[0].plot(ell, dell * beta_s[:,Lidx,0,pidx,ridx],
                                color=str(grays[ii]))
                axs[1].plot(ell, dell * beta_t[:,Lidx,0,pidx,ridx],
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
                    axs[0].plot(ell, dell * beta_s[:,Lidx,1,pidx,ridx],
                                color=str(grays[ii]))
                axs[1].plot(ell, dell * beta_t[:,Lidx,1,pidx,ridx],
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

if __name__ == '__main__':
    
    base_dir = '/mn/stornext/d8/ITA/spider/adri/analysis/'

#    beta_dir = opj(base_dir, '20180911_sst/beta/equilateral')
#    beta_dir = opj(base_dir, '20180911_sst/beta_sparse_ell/equilateral')
    beta_dir = opj(base_dir, '20181123_sst/precomputed')
    img_dir = opj(base_dir, '20181123_sst/img/beta/')

    plot_alpha_beta(beta_dir, img_dir)
    # plot_alpha_beta_matrix(beta_dir, img_dir)

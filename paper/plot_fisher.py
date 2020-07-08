'''
Plot output from calc_fisher.
'''

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from scipy.interpolate import CubicSpline, PchipInterpolator

import numpy as np
import os

matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

opj = os.path.join

def plot_cv_scaling(fisher_dir, out_dir, prim_template='local'):
    '''

    Arguments
    ---------
    fisher_dir : str
        Directory containing fisher pkl files.
    out_dir : str
        Directory for output figures.
    '''
        
    lmax_start = 500
    lmax_end = 4900
    lmax_steps = 10
    lmax_arr =  np.logspace(np.log10(lmax_start), np.log10(lmax_end), lmax_steps)
    lmax_arr = lmax_arr.astype(int)

    lmin_b_arr = np.asarray([2, 20, 30, 50, 80])

    pol_opts_arr = [dict(no_ee=False, no_tt=False),
                    dict(no_ee=True, no_tt=False),
                    dict(no_ee=False, no_tt=True)]

    r_arr = [0, 0.001, 0.01, 0.1]
    
    noise_amp_temp = 0
    noise_amp_e = 0
    noise_amp_b = 0
    lmin_e = 2

    A_lens = 0.1
    
    # Array to fill with loaded fisher values.
    fnl_arr = np.ones((len(r_arr), lmin_b_arr.size, len(pol_opts_arr), lmax_steps)) 
    fnl_arr *= np.nan

    # Load pickle files.
    for ridx, r in enumerate(r_arr):
        for lidx, lmax in enumerate(lmax_arr): 
            for lidx_b, lmin_b in enumerate(lmin_b_arr):
                for pidx, pol_opts in enumerate(pol_opts_arr):

                    no_ee = pol_opts['no_ee']
                    no_tt = pol_opts['no_tt']

                    tag = ('{}_nt{:.4f}_ne{:.4f}_nb{:.4f}_lb{:d}_le{:d}_nee{:d}'
                           '_ntt{:d}_a{:.4f}_r{:.4f}_l{:d}'.format(prim_template,
                        noise_amp_temp, noise_amp_e, noise_amp_b, lmin_b,
                        lmin_e, int(no_ee), int(no_tt), A_lens, r, lmax))

                    try:
                        fisher_file = opj(fisher_dir, 'f_{}.pkl'.format(tag))
                        fisher_opts = np.load(fisher_file)
                    except IOError:
                        print('{} not found'.format(fisher_file))
                        continue
                    fnl_arr[ridx, lidx_b, pidx, lidx] = fisher_opts['sigma_fnl']
    
    # r, lmin_b, pol, lmax
    print fnl_arr[:,0,0,0]
        
    # Interpolate.
    i_fact = 20
    lmax_arr_i =  np.logspace(np.log10(lmax_start), np.log10(lmax_end), i_fact * lmax_steps)
    fnl_arr_i = np.ones((len(r_arr), lmin_b_arr.size, len(pol_opts_arr), i_fact * lmax_steps))

    for i in xrange(fnl_arr.shape[0]):
        for j in xrange(fnl_arr.shape[1]):
            for k in xrange(fnl_arr.shape[2]):
    
                cs = PchipInterpolator(lmax_arr, fnl_arr[i,j,k,:])
                fnl_arr_i[i,j,k,:] = cs(lmax_arr_i)

    fnl_arr = fnl_arr_i
    lmax_arr = lmax_arr_i

    # Plot.
    font = {'size' : 12}
    matplotlib.rc('font', **font)

    fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(4, 4), sharey=True, sharex=False)

    plot_opts = dict(color='black')
    lstyles = ['-', '-.', '', '--', ':']

    for lidx_b, lmin_b in enumerate(lmin_b_arr):
        for pidx, pol_opts in enumerate(pol_opts_arr):

            if pidx > 0:
                alpha = 0.5
                continue
            else:
                alpha = 1

            if lmin_b == 30:
                continue

            label = r'$\ell_{\mathrm{min}}^B = '+str(lmin_b)+'$'
            if lidx_b < 2:
                label_a = label
                label_b = None
            else:
                label_a = None
                label_b = label


            axs[0,0].plot(lmax_arr, fnl_arr[0,lidx_b,pidx,:], alpha=alpha,             
                          ls=lstyles[lidx_b], **plot_opts)
            axs[0,1].plot(lmax_arr, fnl_arr[1,lidx_b,pidx,:], alpha=alpha,
                          ls=lstyles[lidx_b], **plot_opts)
            axs[1,0].plot(lmax_arr, fnl_arr[2,lidx_b,pidx,:], alpha=alpha,
                          ls=lstyles[lidx_b], label=label_a, **plot_opts)
            axs[1,1].plot(lmax_arr, fnl_arr[3,lidx_b,pidx,:], alpha=alpha,
                          ls=lstyles[lidx_b], label=label_b, **plot_opts)

    fig.text(0.001, 0.5, r'$\sigma(\hat{f}_{\mathrm{NL}}^{\, \mathrm{tot}})$',
             ha='center', va='center', rotation='vertical')

    fig.text(0.5, 0.03, r'harmonic band-limit $\ell_{\mathrm{max}}$',
             ha='center', va='center', rotation='horizontal')

    fig.suptitle(r'Cosmic variance only, $A^{BB}_{\mathrm{lens}} = 0.1$', y=.95)

    import matplotlib.ticker as ticker
    
    for i, ax in enumerate(axs.ravel()):
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.tick_params(axis='both', direction='in', top=True, right=True, which='major')
        ax.tick_params(axis='both', direction='in', top=True, right=True, which='minor',
                       labelsize=10)
        ax.set_xlim(400, 6000)
        ax.set_ylim(4e-3, 7e0)        
        ax.text(0.9, 0.85, r'$r='+str(r_arr[i])+'$', transform=ax.transAxes, 
                horizontalalignment='right')


    axs[1,1].xaxis.set_ticklabels(['','','','','','','','',r'$5\times10^3$'], 
                                  minor=True)
    axs[1,1].set_xticks([500, 600, 700, 800, 900, 2000, 3000, 4000, 5000], minor=True)

    axs[1,0].xaxis.set_ticklabels(['','','','','','','','',r'$5\times10^3$'], 
                                  minor=True)
    axs[1,0].set_xticks([500, 600, 700, 800, 900, 2000, 3000, 4000, 5000], minor=True)


    axs[1,1].legend(ncol=1, frameon=False, loc=(0.05, 0.001), 
                    markerscale=.1, handletextpad=0.3, handlelength=1.3,
                    prop={'size': 12})

    axs[1,0].legend(ncol=1, frameon=False, loc=(0.05, 0.001),
                    markerscale=.1, handletextpad=0.3, handlelength=1.3,
                    prop={'size': 12})

    fig.subplots_adjust(hspace=0., wspace=0.)
    fig.savefig(opj(out_dir, 'cv_lim_{}.pdf'.format(prim_template)),
                dpi=300, bbox_inches='tight')
    
def plot_cv_scaling_A_lens(fisher_dir, out_dir, prim_template='local'):
    '''

    Arguments
    ---------
    fisher_dir : str
        Directory containing fisher pkl files.
    out_dir : str
        Directory for output figures.
    '''
        
    lmax_start = 500
    lmax_end = 4900
    lmax_steps = 10
    lmax_arr =  np.logspace(np.log10(lmax_start), np.log10(lmax_end), lmax_steps)
    lmax_arr = lmax_arr.astype(int)

    lmin_b_arr = np.asarray([2, 20, 30, 50, 80])

    pol_opts_arr = [dict(no_ee=False, no_tt=False),
                    dict(no_ee=True, no_tt=False),
                    dict(no_ee=False, no_tt=True)]

    A_lens_arr = [0, 0.1, 0.5, 1]
    
    noise_amp_temp = 0
    noise_amp_e = 0
    noise_amp_b = 0
    lmin_e = 2

    r = 0.001
    
    # Array to fill with loaded fisher values.
    fnl_arr = np.ones((len(A_lens_arr), lmin_b_arr.size, len(pol_opts_arr), lmax_steps)) 
    fnl_arr *= np.nan

    # Load pickle files.
    for aidx, A_lens in enumerate(A_lens_arr):
        for lidx, lmax in enumerate(lmax_arr): 
            for lidx_b, lmin_b in enumerate(lmin_b_arr):
                for pidx, pol_opts in enumerate(pol_opts_arr):

                    no_ee = pol_opts['no_ee']
                    no_tt = pol_opts['no_tt']

                    tag = ('{}_nt{:.4f}_ne{:.4f}_nb{:.4f}_lb{:d}_le{:d}_nee{:d}'
                           '_ntt{:d}_a{:.4f}_r{:.4f}_l{:d}'.format(prim_template,
                        noise_amp_temp, noise_amp_e, noise_amp_b, lmin_b,
                        lmin_e, int(no_ee), int(no_tt), A_lens, r, lmax))

                    try:
                        fisher_file = opj(fisher_dir, 'f_{}.pkl'.format(tag))
                        fisher_opts = np.load(fisher_file)
                    except IOError:
                        print('{} not found'.format(fisher_file))
                        continue
                    fnl_arr[aidx, lidx_b, pidx, lidx] = fisher_opts['sigma_fnl']
    
    # r, lmin_b, pol, lmax
    print fnl_arr[:,0,0,0]
        
    # Interpolate.
    i_fact = 20
    lmax_arr_i =  np.logspace(np.log10(lmax_start), np.log10(lmax_end), i_fact * lmax_steps)
    fnl_arr_i = np.ones((len(A_lens_arr), lmin_b_arr.size, len(pol_opts_arr), i_fact * lmax_steps))

    for i in xrange(fnl_arr.shape[0]):
        for j in xrange(fnl_arr.shape[1]):
            for k in xrange(fnl_arr.shape[2]):
    
                cs = PchipInterpolator(lmax_arr, fnl_arr[i,j,k,:])
                fnl_arr_i[i,j,k,:] = cs(lmax_arr_i)

    fnl_arr = fnl_arr_i
    lmax_arr = lmax_arr_i

    # Plot.
    font = {'size' : 12}
    matplotlib.rc('font', **font)

    fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(4, 4), sharey=True, sharex=False)

    plot_opts = dict(color='black')
    lstyles = ['-', '-.', '', '--', ':']

    for lidx_b, lmin_b in enumerate(lmin_b_arr):
        for pidx, pol_opts in enumerate(pol_opts_arr):

            if pidx > 0:
                alpha = 0.5
                continue
            else:
                alpha = 1

            if lmin_b == 30:
                continue

            label = r'$\ell_{\mathrm{min}}^B = '+str(lmin_b)+'$'
            if lidx_b < 2:
                label_a = label
                label_b = None
            else:
                label_a = None
                label_b = label


            axs[0,0].plot(lmax_arr, fnl_arr[0,lidx_b,pidx,:], alpha=alpha,             
                          ls=lstyles[lidx_b], **plot_opts)
            axs[0,1].plot(lmax_arr, fnl_arr[1,lidx_b,pidx,:], alpha=alpha,
                          ls=lstyles[lidx_b], **plot_opts)
            axs[1,0].plot(lmax_arr, fnl_arr[2,lidx_b,pidx,:], alpha=alpha,
                          ls=lstyles[lidx_b], label=label_a, **plot_opts)
            axs[1,1].plot(lmax_arr, fnl_arr[3,lidx_b,pidx,:], alpha=alpha,
                          ls=lstyles[lidx_b], label=label_b, **plot_opts)

    fig.text(0.001, 0.5, r'$\sigma(\hat{f}_{\mathrm{NL}}^{\, \mathrm{tot}})$',
             ha='center', va='center', rotation='vertical')

    fig.text(0.5, 0.03, r'harmonic band-limit $\ell_{\mathrm{max}}$',
             ha='center', va='center', rotation='horizontal')

    fig.suptitle(r'Cosmic variance only, $r= 0.001$', y=.95)

    for i, ax in enumerate(axs.ravel()):
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.tick_params(axis='both', direction='in', top=True, right=True, which='major')
        ax.tick_params(axis='both', direction='in', top=True, right=True, which='minor',
                       labelsize=10)
        ax.set_xlim(400, 6000)
        ax.set_ylim(4e-3, 7e0)
        ax.text(0.9, 0.85, r'$A^{BB}_{\mathrm{lens}}='+str(A_lens_arr[i])+'$',
                transform=ax.transAxes, horizontalalignment='right')

    axs[1,1].xaxis.set_ticklabels(['','','','','','','','',r'$5\times10^3$'], 
                                  minor=True)
    axs[1,1].set_xticks([500, 600, 700, 800, 900, 2000, 3000, 4000, 5000], minor=True)

    axs[1,0].xaxis.set_ticklabels(['','','','','','','','',r'$5\times10^3$'], 
                                  minor=True)
    axs[1,0].set_xticks([500, 600, 700, 800, 900, 2000, 3000, 4000, 5000], minor=True)


    axs[1,1].legend(ncol=1, frameon=False, loc=(0.05, 0.001),
                    markerscale=.1, handletextpad=0.3, handlelength=1.3,
                    prop={'size': 12})

    axs[1,0].legend(ncol=1, frameon=False, loc=(0.05, 0.001),
                    markerscale=.1, handletextpad=0.3, handlelength=1.3,
                    prop={'size': 12})

    fig.subplots_adjust(hspace=0., wspace=0.)
    fig.savefig(opj(out_dir, 'cv_lim_alens.pdf'), dpi=300, bbox_inches='tight')
    
def plot_pol(fisher_dir, out_dir, prim_template='local', plot_invcov=False):
    '''
    
    Arguments
    ---------
    fisher_dir : str
        Directory containing fisher pkl files.
    out_dir : str
        Directory for output figures.
    '''
        
    lmax_start = 500
    lmax_end = 4900
    lmax_steps = 10
    lmax_arr =  np.logspace(np.log10(lmax_start), np.log10(lmax_end), lmax_steps)
    lmax_arr = lmax_arr.astype(int)

    lmin_b = 2

    pol_opts_arr = [dict(no_ee=True, no_tt=False),
                    dict(no_ee=False, no_tt=True),
                    dict(no_ee=False, no_tt=False)]

    pol_opts_names = [r'$B+T$', r'$B+E$', r'$B+T + E$']

    r = 0.001
    lmin_e = 2

    # Add noise array.
    noise_opts_arr = [dict(noise_amp_temp=0, noise_amp_e=0, noise_amp_b=0),
                      dict(noise_amp_temp=4, noise_amp_e=4*np.sqrt(2),
                           noise_amp_b=4*np.sqrt(2))]
    n_ell_arr = ['0', '4']
    A_lens = 0.5
    
    # Array to fill with loaded fisher values.
    fnl_arr = np.ones((len(noise_opts_arr), len(pol_opts_arr), lmax_steps)) 
    fnl_arr *= np.nan

    # Load pickle files.
    for nidx, noise_opts in enumerate(noise_opts_arr):
        for lidx, lmax in enumerate(lmax_arr): 
            for pidx, pol_opts in enumerate(pol_opts_arr):

                no_ee = pol_opts['no_ee']
                no_tt = pol_opts['no_tt']

                noise_amp_temp = noise_opts['noise_amp_temp']
                noise_amp_e = noise_opts['noise_amp_e']
                noise_amp_b = noise_opts['noise_amp_b']

                tag = ('{}_nt{:.4f}_ne{:.4f}_nb{:.4f}_lb{:d}_le{:d}_nee{:d}'
                       '_ntt{:d}_a{:.4f}_r{:.4f}_l{:d}'.format(prim_template,
                    noise_amp_temp, noise_amp_e, noise_amp_b, lmin_b,
                    lmin_e, int(no_ee), int(no_tt), A_lens, r, lmax))
                # Load fisher.
                try:
                    fisher_file = opj(fisher_dir, 'f_{}.pkl'.format(tag))
                    fisher_opts = np.load(fisher_file)
                except IOError:
                    print('{} not found'.format(fisher_file))
                    continue
                fnl_arr[nidx, pidx, lidx] = fisher_opts['sigma_fnl']

                if plot_invcov:
                    # Load invcov and cov, plot right away.
                    try:
                        invcov_file = opj(fisher_dir, 'invcov_{}.pkl'.format(tag))
                        invcov_opts = np.load(invcov_file)
                    except IOError:
                        print('{} not found'.format(invcov_file))
                        continue


                    invcov_name = opj(out_dir, 'invcov', 'invcov_{}.png'.format(tag))
                    cov_name = opj(out_dir, 'invcov', 'cov_{}.png'.format(tag))
                    ells = invcov_opts['ells']
                    invcov = invcov_opts['invcov']
                    cov = invcov_opts['cov']
                    plot_invcov(ells, invcov, invcov_name, dell=False)
                    plot_invcov(ells, cov, cov_name)
    
    # pol, lmax
    print fnl_arr.shape
        
    # Interpolate.
    i_fact = 20
    lmax_arr_i =  np.logspace(np.log10(lmax_start), np.log10(lmax_end), i_fact * lmax_steps)
    fnl_arr_i = np.ones((len(noise_opts_arr), len(pol_opts_arr), i_fact * lmax_steps))

    for i in xrange(fnl_arr.shape[0]):
        for j in xrange(fnl_arr.shape[1]):
            cs = PchipInterpolator(lmax_arr, fnl_arr[i,j,:])
            fnl_arr_i[i,j,:] = cs(lmax_arr_i)

    fnl_arr = fnl_arr_i
    lmax_arr = lmax_arr_i


    # Plot.
    font = {'size' : 12}
    matplotlib.rc('font', **font)

    fig, axs = plt.subplots(ncols=2, nrows=1, figsize=(4, 2), sharey=True, sharex=False)

    plot_opts = dict(color='black')
    lstyles = ['--', '-.', '-', ':']

    for pidx, pol_opts in enumerate(pol_opts_arr):

        label = pol_opts_names[pidx]

        if pidx < 2:
            label_a = label
            label_b = None
        else:
            label_a = None
            label_b = label

        axs[0].plot(lmax_arr, fnl_arr[0,pidx,:],
                      ls=lstyles[pidx], label=label_a, **plot_opts)

        axs[1].plot(lmax_arr, fnl_arr[1,pidx,:],
                      ls=lstyles[pidx], label=label_b, **plot_opts)

    fig.text(0.001, 0.5, r'$\sigma(\hat{f}_{\mathrm{NL}}^{\, \mathrm{tot}})$',
             ha='center', va='center', rotation='vertical')

    fig.text(0.5, -0.05, r'harmonic band-limit $\ell_{\mathrm{max}}$',
             ha='center', va='center', rotation='horizontal')

    fig.suptitle(r'$A^{BB}_{\mathrm{lens}} = '+str(A_lens)+'$, $r='+str(r)+'$', y=1.03)
    
    for i, ax in enumerate(axs.ravel()):
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.tick_params(axis='both', direction='in', top=True, right=True, which='major')
        ax.tick_params(axis='both', direction='in', top=True, right=True, which='minor',
                       labelsize=10)
        ax.set_xlim(400, 6000)
        ax.set_ylim(4e-3, 7e0)        
        ax.text(0.9, 0.85, 
                r"$"+str(n_ell_arr[i])+"\ \mu \mathrm{K}$-$\mathrm{arcmin}$",
                transform=ax.transAxes, horizontalalignment='right')


    axs[1].xaxis.set_ticklabels(['','','','','','','','',r'$5\times10^3$'], 
                                  minor=True)
    axs[1].set_xticks([500, 600, 700, 800, 900, 2000, 3000, 4000, 5000], minor=True)

    axs[0].xaxis.set_ticklabels(['','','','','','','','',r'$5\times10^3$'], 
                                  minor=True)
    axs[0].set_xticks([500, 600, 700, 800, 900, 2000, 3000, 4000, 5000], minor=True)


    axs[0].legend(ncol=1, frameon=False, loc=(0.05, 0.001),
                    markerscale=.1, handletextpad=0.3, handlelength=1.3,
                    prop={'size': 12})

    axs[1].legend(ncol=1, frameon=False, loc=(0.05, 0.001),
                    markerscale=.1, handletextpad=0.3, handlelength=1.3,
                    prop={'size': 12})

    fig.subplots_adjust(hspace=0., wspace=0.)
    fig.savefig(opj(out_dir, 'pol.pdf'), dpi=300, bbox_inches='tight')


def plot_noise(fisher_dir, out_dir, prim_template='local'):
    '''
    Plot sigma as function of B noise for several T/E noise curves and two
    choices of A_lens.

    Arguments
    ---------
    fisher_dir : str
        Directory containing fisher pkl files.
    out_dir : str
        Directory for output figures.
    '''

    r = 0.001
    lmax = 4900
    lmin_b = 50        

    pol_opts = dict(no_ee=False, no_tt=False)
    lmin_e = 2

#    noise_i_arr = [0.3, 1, 3, 10]
    noise_i_arr = [10, 1]
    noise_b_arr = np.logspace(np.log10(0.3), np.log10(50), 10)

    A_lens_arr = [0.1, 1]

    fnl_arr = np.ones((len(A_lens_arr), len(noise_i_arr), len(noise_b_arr)))
    fnl_arr *= np.nan

    # Load pickle files.
    for aidx, A_lens in enumerate(A_lens_arr):
        for ni_idx, n_i in enumerate(noise_i_arr):
            for nb_idx, n_b in enumerate(noise_b_arr):

                no_ee = pol_opts['no_ee']
                no_tt = pol_opts['no_tt']

                noise_amp_temp = n_i
                noise_amp_e = n_i * np.sqrt(2)
                noise_amp_b = n_b

                tag = ('{}_nt{:.4f}_ne{:.4f}_nb{:.4f}_lb{:d}_le{:d}_nee{:d}'
                       '_ntt{:d}_a{:.4f}_r{:.4f}_l{:d}'.format(prim_template,
                    noise_amp_temp, noise_amp_e, noise_amp_b, lmin_b,
                    lmin_e, int(no_ee), int(no_tt), A_lens, r, lmax))

                # Load fisher.
                try:
                    fisher_file = opj(fisher_dir, 'f_{}.pkl'.format(tag))
                    fisher_opts = np.load(fisher_file)
                except IOError:
                    print('{} not found'.format(fisher_file))
                    continue
                fnl_arr[aidx, ni_idx, nb_idx] = fisher_opts['sigma_fnl']
    
    # pol, lmax
    print fnl_arr.shape
        
    # Interpolate.
    i_fact = 20
    noise_b_arr_i = np.logspace(np.log10(0.3), np.log10(50), i_fact * 10)
    fnl_arr_i = np.ones((len(A_lens_arr), len(noise_i_arr), i_fact * len(noise_b_arr)))

    for i in xrange(fnl_arr.shape[0]):
        for j in xrange(fnl_arr.shape[1]):
            cs = PchipInterpolator(noise_b_arr, fnl_arr[i,j,:])
            fnl_arr_i[i,j,:] = cs(noise_b_arr_i)

#    fnl_arr = fnl_arr_i
#    noise_b_arr = noise_b_arr_i

    # Plot.
    font = {'size' : 12}
    matplotlib.rc('font', **font)

    fig, axs = plt.subplots(ncols=2, nrows=1, figsize=(4, 2), sharey=True, sharex=False)

    plot_opts = dict(color='black')
    lstyles = ['-', ':', '--', '-.']

    for ni_idx, n_i in enumerate(noise_i_arr):

        label = r"$"+str(n_i)+"\ \mu \mathrm{K}$-$'$"

#        if ni_idx < 2:
#            label_a = label
#            label_b = None
#        else:
#            label_a = None
#            label_b = label

        axs[0].plot(noise_b_arr, fnl_arr[0,ni_idx,:],
                      ls=lstyles[ni_idx], label=None, **plot_opts)

        axs[1].plot(noise_b_arr, fnl_arr[1,ni_idx,:],
                      ls=lstyles[ni_idx], label=label, **plot_opts)

    fig.text(0.001, 0.5, r'$\sigma(\hat{f}_{\mathrm{NL}}^{\, \mathrm{tot}})$',
             ha='center', va='center', rotation='vertical')

    fig.text(0.5, -0.05, r'$B$-mode noise [$\mu \mathrm{K}$-$\mathrm{arcmin}$]',
             ha='center', va='center', rotation='horizontal')

    fig.suptitle(r'$\ell_{\mathrm{min}}^B='+str(lmin_b)+'$, $r='+str(r)+'$', y=1.03)
    
    for i, ax in enumerate(axs.ravel()):
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.tick_params(axis='both', direction='in', top=True, right=True, which='major')
        ax.tick_params(axis='both', direction='in', top=True, right=True, which='minor',
                       labelsize=10)
        ax.set_xlim(0.2, 70)
#        ax.set_ylim(4e-3, 7e0)        
        ax.text(0.1, 0.85, 
                r"$A^{BB}_{\mathrm{lens}} = "+str(A_lens_arr[i])+"$",
                transform=ax.transAxes, horizontalalignment='left')


#    axs[1].xaxis.set_ticklabels(['','','','','','','','',r'$5\times10^3$'], 
#                                  minor=True)
#    axs[1].set_xticks([500, 600, 700, 800, 900, 2000, 3000, 4000, 5000], minor=True)

#    axs[0].xaxis.set_ticklabels(['','','','','','','','',r'$5\times10^3$'], 
#                                  minor=True)
#    axs[0].set_xticks([500, 600, 700, 800, 900, 2000, 3000, 4000, 5000], minor=True)


#    axs[0].legend(ncol=1, frameon=False, loc=(0.45, 0.001),
#                    markerscale=.1, handletextpad=0.3, handlelength=1.3,
#                    prop={'size': 12})

    axs[1].legend(ncol=1, frameon=False, loc=(0.42, 0.0001),
                    markerscale=.1, handletextpad=0.3, handlelength=1.3,
                    prop={'size': 12})

    fig.subplots_adjust(hspace=0., wspace=0.)
    fig.savefig(opj(out_dir, 'noise.pdf'), dpi=300, bbox_inches='tight')


def plot_invcov(ells, invcov, filename, dell=True):
    ''' Plot invcov and cov. '''

    nls_dict = {'TT': 0, 'EE': 1, 'BB': 2, 'TE': 3,
                'ET': 3, 'BT': 4, 'TB': 4, 'EB': 5,
                'BE': 5}

    if dell:
        dells = ells * (ells + 1) / 2. / np.pi
    else:
        dells = 1.

    fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(9,7))

    for pidx1, pol1 in enumerate(['T', 'E', 'B']):
        for pidx2, pol2 in enumerate(['T', 'E', 'B']):
            axs[pidx1,pidx2].plot(ells, dells*invcov[:,pidx1,pidx2],
                                  label=pol1+pol2)

    for ax in axs.reshape(-1):
        ax.legend()
    plt.tight_layout()
    fig.savefig(filename)
    plt.close(fig)

if __name__ == '__main__':
    
    base_dir = '/mn/stornext/d8/ITA/spider/adri/analysis/'
    ana_dir = opj(base_dir, '20190411_beta')
#    out_dir = opj(ana_dir, 'img/fisher')
    out_dir = opj(ana_dir, 'img/img_temperature')
    fisher_dir = opj(ana_dir, 'fisher')

    
    #plot_cv_scaling(fisher_dir, out_dir, prim_template='local')
    #plot_cv_scaling_A_lens(fisher_dir, out_dir)
    plot_pol(fisher_dir, out_dir)
    #plot_noise(fisher_dir, out_dir)

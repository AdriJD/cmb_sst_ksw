'''
Load up txt output from camb and convert to npy
'''
import os
import numpy as np

opj = os.path.join

ana_dir = '/mn/stornext/d8/ITA/spider/adri/analysis/20171217_sst/camb_output/beta/'

alpha = np.genfromtxt(opj(ana_dir, 'test__alpha.txt'))
beta = np.genfromtxt(opj(ana_dir, 'test__beta.txt'))
err = np.genfromtxt(opj(ana_dir, 'test__alpha_beta_r.txt'))

alpha = np.save(opj(ana_dir, 'test__alpha.npy'), alpha)
beta = np.save(opj(ana_dir, 'test__beta.npy'), beta)
err = np.save(opj(ana_dir, 'test__alpha_beta_r.npy'), err)



'''
Load up txt output from camb and convert to npy
'''
import os
import numpy as np

opj = os.path.join

#ana_dir = '/mn/stornext/d8/ITA/spider/adri/analysis/20171217_sst/camb_output/beta/'
ana_dir = '/mn/stornext/d8/ITA/spider/adri/analysis/20180908_camb_alpha_beta/'
tag = ''

alpha = np.genfromtxt(opj(ana_dir, '{}__alpha.txt'.format(tag)))
beta = np.genfromtxt(opj(ana_dir, '{}__beta.txt'.format(tag)))
err = np.genfromtxt(opj(ana_dir, '{}__alpha_beta_r.txt'.format(tag)))

alpha = np.save(opj(ana_dir, '{}__alpha.npy'.format(tag)), alpha)
beta = np.save(opj(ana_dir, '{}__beta.npy'.format(tag)), beta)
err = np.save(opj(ana_dir, '{}__alpha_beta_r.npy'.format(tag)), err)



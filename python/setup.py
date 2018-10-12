from setuptools import setup

setup(name='cmb_sst',
      version='0.1',
      description='Code for analysis of tensor-scalar bispectra.',
      url='https://github.com/oskarkleincentre/cmb_sst_ksw',
      author='Adri J. Duivenvoorden',
      author_email='adri.j.duivenvoorden@gmail.com',
      license='MIT',
      packages=['sst'],
      install_requires=['scipy', 'numpy'],
#      test_suite='python/tests',
      test_suite='nose.collector',
      tests_require=['nose'],
      zip_safe=False)

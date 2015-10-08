#!/usr/bin/env python
""" Setup utility for the linearizedGP package. """

from distutils.core import setup

setup(
    name='linearizedGP',
    version='1.0',
    description='Implementation of the extended and unscented Gaussian '
                'processes.',
    author='Daniel Steinberg',
    author_email='daniel.steinberg@nicta.com.au',
    url='https://github.com/NICTA/linearizedGP',
    license='LGPLv3',
    packages=['linearizedGP'],
    install_requires=["scipy >= 0.12.0",
                      "numpy >= 1.8.0"
                      # NLopt >= 2.4.2
                      ],
    provides=['linearizedGP']
)

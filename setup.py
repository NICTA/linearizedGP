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
    url='',
    packages=['linearizedGP'],
    install_requires=[
        "scipy >= 0.12.0",  # These may be able to be lower, just not tested
        "numpy >= 1.8.0",
        # NLopt >= 2.4.2
        ]
)

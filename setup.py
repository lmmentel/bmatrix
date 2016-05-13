''' writeBmat setup script'''

import sys

from setuptools import setup

def readme():
    '''Return the contents of the README.md file.'''
    with open('README.md') as freadme:
        return freadme.read()

setup(
    include_package_data = True,
    entry_points = {
        'console_scripts' : [
            'writeBmat = writeBmat.writeBmat:main',
        ]
    },
    long_description = readme(),
    name = 'writeBmat',
    packages = ['writeBmat'],
    url = 'https://bitbucket.org/lukaszmentel/writebmat/',
    version = '0.1.0',
    classifiers = [
        'Environment :: Console',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Topic :: Scientific/Engineering :: Chemistry',
        'Topic :: Scientific/Engineering :: Physics',
    ],
)

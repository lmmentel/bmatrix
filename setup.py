
''' writeBmat setup script'''

import sys

from setuptools import setup

def readme():
    '''Return the contents of the README.md file.'''
    with open('README.rst') as freadme:
        return freadme.read()

setup(
    author='Tomas Bucko',
    include_package_data = True,
    entry_points = {
        'console_scripts' : [
            'writeBmat = writeBmat.writeBmat:main',
        ]
    },
    license='GPLv3',
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
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
    ],
)

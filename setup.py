""" bmatrix setup script"""

from setuptools import setup


def readme():
    """Return the contents of the README.md file."""
    with open("README.rst") as freadme:
        return freadme.read()


setup(
    author="Tomas Bucko, Łukasz Mentel",
    include_package_data=True,
    entry_points={
        "console_scripts": [
            "bmatrix = bmatrix.bmatrix:main",
        ]
    },
    license="GPLv3",
    long_description=readme(),
    name="bmatrix",
    packages=["bmatrix"],
    url="https://github.com/lmmentel/bmatrix",
    version="1.0.1",
    install_requires=["numpy", "scipy"],
    classifiers=[
        "Environment :: Console",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Scientific/Engineering :: Physics",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
    ],
)

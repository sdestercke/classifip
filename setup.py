from setuptools import setup, find_packages
from setuptools.command.install import install
import os

resources_dir = os.path.join('resources')
datafiles = [(d, [os.path.join(d, f) for f in files])
             for d, folders, files in os.walk(resources_dir)]

classifiers = """\
Development Status :: 2 - Pre-Alpha
License :: OSI Approved :: GNU General Public License (GPL)
Intended Audience :: Developers
Intended Audience :: End Users/Desktop
Intended Audience :: Science/Research
Topic :: Scientific/Engineering :: Artificial Intelligence
Topic :: Scientific/Engineering :: Mathematics
Programming Language :: Python
Programming Language :: Python :: 3
Operating System :: OS Independent"""

# Override standard setuptools commands.
# Enforce the order of dependency installation.
# -------------------------------------------------
PREREQS = ['numpy==1.19.5',
           'matplotlib==3.1.1',
           'scipy==1.2.3',
           'scikit-learn==0.23.2',
           'pandas==0.25.1',
           'CVXcanon==0.1.1',
           'cvxopt==1.2.3',
           'Orange3==3.24.0',
           'python-constraint==1.4.0',
           'feather-format==0.4.0',
           'cvxpy==0.4.11',
           'xxhash==1.4.3',
           'qcqp==0.8.3'
           ]


def requires(packages):
    from os import system
    from sys import executable as PYTHON_PATH
    from pkg_resources import require
    require("pip")
    CMD_TMPLT = '"' + PYTHON_PATH + '" -m pip install %s'
    for pkg in packages: system(CMD_TMPLT % (pkg,))


class OrderedInstall(install):
    def run(self):
        requires(PREREQS)
        install.run(self)


CMD_CLASSES = {
    "install": OrderedInstall
}
# -------------------------------------------------
setup(
    name='classifip',
    version='0.2.1',
    author='Sebastien Destercke',
    author_email='sebastien.destercke@hds.utc.fr',
    packages=find_packages(),
    url='http://pypi.python.org/pypi/classifip/',
    license="GNU General Public License (GPL)",
    platforms="any",
    description='Classification with Imprecise Probability methods.',
    long_description=open('README.rst').read(),
    classifiers=classifiers.split('\n'),
    cmdclass=CMD_CLASSES,
    data_files=datafiles
)

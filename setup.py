from setuptools import setup, find_packages
import os

resources_dir = os.path.join('resources')
datafiles = [(d, [os.path.join(d,f) for f in files])
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

setup(
    name='classifip',
    version='0.2.1',
    author='Sebastien Destercke',
    author_email='sebastien.destercke@hds.utc.fr',
    packages=find_packages(),
    url='http://pypi.python.org/pypi/classifip/',
    license="GNU General Public License (GPL)",
    platforms = "any",
    description='Classification with Imprecise Probability methods.',
    long_description=open('README.rst').read(),
    classifiers = classifiers.split('\n'),
    install_requires=['numpy', 'cvxopt', 'scikit-learn', 'matplotlib', 'pandas',
                      'Orange3', 'python-constraint', 'feather-format'],
    data_files= datafiles
)

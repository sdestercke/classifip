from setuptools import setup, find_packages
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
    install_requires=[
           'numpy==1.19.5',
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
           'qcqp==0.8.3'],
    data_files=datafiles
)

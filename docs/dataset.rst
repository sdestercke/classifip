.. module:: classifip.dataset

.. moduleauthor:: Sebastien Destercke <sebastien.destercke@hds.utc.fr> 
.. moduleauthor:: Gen Yang <gen.yang@hds.utc.fr> 

.. _dataset-doc:

dataset module
==============

dataset module includes the tools necessary to read/write data file (currently
only WEKA arff format is available).

.. autofunction:: discretize

.. todo:: 
    Build an extension of arff accepting various uncertainty representations
    (generic constraints matrix, possibility distributions, etc..)

.. toctree::
   :maxdepth: 1

   dataset/arff
   
References
^^^^^^^^^^

.. [#fayyad1993]

   Usama Fayyad, Keki Irani. "Multi-interval discretization of
   continuous-valued attributes for classification learning". International
   Joint Conference on Artificial Intelligence 1022-1027 (1993).

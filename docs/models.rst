models module
=============

models module includes the different classifiers. Most of them have two methods:

    * **learning()** that takes an ArffFile object (see :ref:`dataset-doc`) as argument from which the classifier is learned
    * **evaluate()** that takes a list of instances and return, for each of them, a object of module representation (see :ref:`representations-doc`) describing our knowledge about the class

.. toctree::
   :maxdepth: 1

   models/knn
   models/ncc
   models/pairpip
   models/knnbr


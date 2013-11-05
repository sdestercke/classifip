.. _representations-doc: 

representations module
======================

Representation module contains representation tools usually returned by learning tasks
to make a prediction, includng imprecise probability representations. Each representation
comes with a set of prediction rules, both precise (maximax, maximin, ...) and imprecise
(interval dominance, maximality, ...)

.. toctree::
   :maxdepth: 1

   representations/probint
   representations/voting
   representations/credalset
   representations/genpbox

Number of + signs indicate how much work implies the change

* add doctest when possible

dataset package
^^^^^^^^^^^^^^^

* make an outer function that selects, from an Arff file, specified values of specified columns (+) 
* extend ArffFile object and arff file types to accomodate with uncertain data (+++)

models package
^^^^^^^^^^^^^^

* add a general binary decomposition method (+++)
* add multilabel chaining methods (+++)
* add multilabel calibrating methods (+++)
* add decision trees (+++)
* add credal model averaging logistic regressions (+++)
* add credal model averaging of networks (+++)

evaluation package
^^^^^^^^^^^^^^^^^^

* add classifiers comparison using Demsar paper (+++)
* add a function doing fold cross validation for any method (++)
* add structured validation in fold creation function (+)

representations package
^^^^^^^^^^^^^^^^^^^^^^^

* add transformation and approximations between different formats (+++)
* add belief functions (+++)
* add possibility distributions (+++)
* add comparative probabilities on singletons (+++)
* add neighbourhood models: linear-vacuous, PMM, … (++)
* for each model, add extreme point extraction method (+++)
* for each model, implement directly combination method (conjunctive, disjunctive)

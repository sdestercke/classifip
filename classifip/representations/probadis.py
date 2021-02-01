import numpy as np
from classifip.representations.credalset import CredalSet
from math import fabs


class ProbaDis(CredalSet):
    """Class of (discrete) probability distribution: a single probability
    distribution.
    
    :param proba: a 1xn array containing probability
    :type proba: :class:`~numpy.array`
    :param nbDecision: number of elements of the space
    :type nbDecision: integer
    
    >>> from classifip.representations import probadis
    >>> from numpy import array
    >>> distrib=array([0.15,0.1,0.4,0.35])
    >>> Test=probadis.ProbaDis(distrib)
    >>> subset=array([1.,0.,0.,1.])
    >>> Test.getlowerprobability(subset)
    0.5
    """

    def __init__(self, proba):
        """Instanciate proba values
        
        :param proba: a 1xn array containing upper (1st row) and lower bounds
        :type proba: :class:`~numpy.array`
        """
        if proba.__class__.__name__ != 'ndarray':
            raise Exception('Expecting a numpy array as argument')
        self.proba = proba.copy()
        self.nbDecision = proba.size
        if fabs(proba.sum() - 1.) > 1e-7:
            raise Exception('proba weights do not sum to one')

    def isproper(self):
        """Check if probability is well-defined. 
        
        :returns: 0 (empty/incur sure loss) or 1 (non-empty/avoid sure loss).
        :rtype: integer
        
        """
        # check inequality of bounds
        if fabs(self.proba.sum() - 1.) > 1e-7:
            return 0
        return 1

    def isreachable(self):
        """Check if probability is reachable
        
        :returns: 0 (not coherent/tight) or 1 (tight/coherent).
        :rtype: integer
        
        """
        return self.isproper()

    def setreachableprobability(self):
        """Make the bounds reachable.
        
        """
        if self.isproper() == 0:
            raise Exception('ill-defined probability inducing empty set: operation not possible')

    def getlowerprobability(self, subset):
        """Compute probability of an event expressed in binary code. 
        
        :param subset: the event of interest (a 1xn vector containing 1 for elements
            in the event, 0 otherwise.)
        :param type: np.array
        :returns: probability value
        :rtype: float
        
        """
        if subset.__class__.__name__ != 'ndarray':
            raise Exception('Expecting a numpy array as argument')
        if subset.size != self.nbDecision:
            raise Exception('Subset incompatible with the frame size')
        if self.isproper() == 0:
            raise Exception('Not a well-defined probability')
        return np.dot(self.proba, subset)

    def getupperprobability(self, subset):
        """Compute probability of an event expressed in binary code. 
        
        :param subset: the event of interest (a 1xn vector containing 1 for elements
            in the event, 0 otherwise.)
        :param type: np.array
        :returns: upper probability value
        :rtype: float
        
        """
        return self.getlowerprobability(subset)

    def getlowerexpectation(self, function):
        """Compute the expectation of a given (bounded) function by using
        weighted sum
        
        :param function: the function values
        :param type: np.array
        :returns: lower expectation value
        :rtype: float
        """
        lowerexpe = 0.
        if function.__class__.__name__ != 'ndarray':
            raise Exception('Expecting a numpy array as argument')
        if function.size != self.nbDecision:
            raise Exception('number of elements incompatible with the frame size')
        if self.isproper() == 0:
            raise Exception('Not a well-defined probability')
        function = function.astype(float)
        return np.dot(self.proba, function)

    def getDelCozmulti(self, beta=1.):
        """
        Making set-valued predictions using the method based on the article 
        "Learning nondeterministic classifiers" of J. Del Coz and A. Bahamonde.
    
        The posterior probabilities computed by a base learner is required as parameter.

        The algorithm optimizes the output to a set of classes basing on these posterior 
        probabilities and the F-beta measure.
        
        :param beta: parameter for the F_beta loss used for the loss minimization procedure,
            by default, the F1 measure is used
        :type beta: int
        
        :return: the set of optimal classes (under Del Coz rule) as a 1xn vector
            where indices of optimal classes are set to one
        :rtype: :class:`~numpy.array`
        """

        # get the descending ordering of porbabilities
        proba_ordered = np.argsort(-np.array(self.proba))
        ret = np.zeros(self.nbDecision)

        # initialization of optimization
        i = 1
        loss = 1.
        loss_buffer = 1. - (1. + beta * beta) / (beta * beta + i) * \
                      sum([self.proba[ind] for ind in proba_ordered[0:i]])
        while (i < self.nbDecision) and (loss_buffer < loss):
            i += 1
            loss = loss_buffer
            loss_buffer = 1. - (1. + beta * beta) / (beta * beta + i) * \
                          sum([self.proba[ind] for ind in proba_ordered[0:i]])

        if loss_buffer >= loss:
            for ind in proba_ordered[0:(i - 1)]:
                ret[ind] = 1.
        else:
            for ind in range(0, self.nbDecision):
                ret[ind] = 1.
        return ret

    def getDelCozordinal(self, beta=1.):
        """
        Making interval-valued predictions using the method based on the article 
        "Learning to Predict One or More Ranks in Ordinal Regression Tasks" of J. Del Coz.
    
        The posterior probabilities computed by a base learner is required as a parameter.
        The algorithm optimizes the output to a set of classes basing on these posterior probabilities.
        
        :param beta: parameter for the F_beta loss used for the loss minimization procedure,
            by default, the F1 measure is used
        :type beta: int
        
        :return: the set of optimal classes (under Del Coz rule) as a 1xn vector
            where indices of optimal classes are set to one
        :rtype: :class:`~numpy.array`
        """

        ret = np.zeros(self.nbDecision)

        max_proba = np.zeros((self.nbDecision, 2))

        # we start by finding out the max probabilities when the length of the interval is fixed
        # i is the length of class intervals
        for i in range(1, self.nbDecision + 1):

            # j is the j-th class
            for j in range(0, self.nbDecision - i + 1):
                buffer_proba = sum(self.proba[j:(j + i)])

                if buffer_proba > max_proba[i - 1][1]:
                    max_proba[i - 1][0] = j  # starting class
                    max_proba[i - 1][1] = buffer_proba  # new highest probability of the intervals of length i starting at j
        # when the length of the interval is nbclass, i.e. we take the whole class, and the proba is 1        
        # max_proba[nbclass-1][0] = 0 
        # max_proba[nbclass-1][1] = 1.

        # now we minimize the F_beta loss depending on the length i of the class
        # i is the length of class intervals
        min_loss = 1.
        min_i = 1
        for i in range(1, self.nbDecision):
            buffer_min = 1. - (1. + beta * beta) / (beta * beta + i) * max_proba[i - 1][1]

            if buffer_min < min_loss:
                min_loss = buffer_min
                min_i = i

        # we predict every class situated in the interval defined previously with min_i and max_proba[0]
        start_indice = int(max_proba[min_i - 1][0])
        for indice in range(start_indice, start_indice + min_i):
            ret[indice] = 1.

        return ret

    def getmaximaldecision(self,  utilities=None):
        """
            Using the zero-one loss matrix
            ToDo: Generalize for any loss matrix
        :return:
        """
        return np.argmax(self.proba)

    def __str__(self):
        """Print the current bounds 
        """
        str1 = "Proba weights |"
        str3 = "              "
        i = 0
        for j in range(self.nbDecision):
            str3 += "   y%d " % i
            str1 += " %.3f" % self.proba[j]
            i += 1
        str3 += "\n"
        str3 += "           "
        str3 += "--------------------"
        str3 += "\n"
        str3 += str1
        str3 += "\n"
        return str3

    def __and__(self, other):
        """Compute the intersection of two probabilities
        """
        if self.proba == other.proba:
            return ProbaDis(self.proba)
        else:
            raise Exception('empty intersection, unequal probabilities')

    def __add__(self, other):
        """Compute the average of two probability intervals
        """
        fusedproba = np.zeros(self.nbDecision)
        fusedproba = np.mean([self.proba, other.proba], axis=0)
        return ProbaDis(fusedproba)

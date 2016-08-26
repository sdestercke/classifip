import numpy as np
from credalset import CredalSet
from math import fabs

class BelFun(CredalSet):
    """Class of belief functions, described by their mass assignment.
    
    :param mass: a 1 x m array containing focal sets weight
    :type mass: :class:`~numpy.array`        
    :param nbDecision: number of elements of the space
    :type nbDecision: integer
    :param focals: a m x nbDecision array, each row corresponding to a focal
    :type focals: :class:`~numpy.array`
    
    >>> from numpy import array
    >>> import classifip.representations.belfun as bel
    >>> sets=np.array([[0.,0.,1.],[1.,0.,1.],[1.,1.,0.],[1.,0.,0.],[1.,1.,1.]])
    >>> mass=np.array([0.1,0.3,0.45,0.1,0.05])
    >>> test=bel.BelFun(mass,sets)
    >>> print test
    Mass value |  Focal set 
    -----------|----------------- 
       0.100   |  [ 0.  0.  1.]
       0.300   |  [ 1.  0.  1.]
       0.450   |  [ 1.  1.  0.]
       0.100   |  [ 1.  0.  0.]
       0.050   |  [ 1.  1.  1.]
    -----------|-----------------
    >>> test.getlowerprobability(np.array([1.,0.,1.]))
    0.5
    >>> test.getlowerexpectation(np.array([-2,-1,6]))
    -1.2
    >>> test.getmaximaldecision()
    array([ 1.,  1.,  1.])
    >>> test.getmaximindecision()
    0
    """
    
    def __init__(self,mass,focals):
        """Instanciate focal sets and their mass
        
        :param mass: a 1 x m array containing focal sets weight
        :type mass: :class:`~numpy.array`
        :param focals: a m x nbDecision array, each row corresponding to a focal
        :type focals: :class:`~numpy.array`
        """
        if mass.__class__.__name__ != 'ndarray':
            raise Exception('Expecting a numpy array as argument')
        if focals.__class__.__name__ != 'ndarray':
            raise Exception('Expecting a numpy array as argument')
        self.mass=mass
        self.focals=focals
        self.nbDecision=focals.shape[1]
        if fabs(mass.sum()-1.) > 1e-7:
            raise Exception('mass weights do not sum to one')
        if (mass < 0).any():
            raise Exception('mass weights must be positive')

    def isproper(self):
        """Check if belief function is well-defined. 
        
        :returns: 0 (empty/incur sure loss) or 1 (non-empty/avoid sure loss).
        :rtype: integer
        
        """
        #check whether empty set is a focal element
        empty=np.zeros(self.nbDecision)
        if np.any(np.all(self.focals==empty,axis=1)):
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
        if self.isproper()==0:
            raise Exception('mass function inducing empty set: operation not possible')

    def getlowerprobability(self,subset):
        """Compute probability of an event expressed in binary code. 
        
        :param subset: the event of interest (a 1xn vector containing 1 for elements
            in the event, 0 otherwise.)
        :param type: np.array
        :returns: probability value
        :rtype: float
        
        """
        if subset.__class__.__name__!='ndarray':
            raise Exception('Expecting a numpy array as argument')
        if subset.size != self.nbDecision:
            raise Exception('Subset incompatible with the frame size')
        if self.isproper()==0:
            raise Exception('Not a well-defined mass function')
        inclusion=np.all(self.focals==np.minimum(self.focals,subset),axis=1)
        return np.dot(self.mass,inclusion)

    def getupperprobability(self,subset):
        """Compute probability of an event expressed in binary code. 
        
        :param subset: the event of interest (a 1xn vector containing 1 for elements
            in the event, 0 otherwise.)
        :param type: np.array
        :returns: upper probability value
        :rtype: float
        
        """    
        return 1-self.getlowerprobability(1-subset)
    
    def getlowerexpectation(self,function):
        """Compute the expectation of a given (bounded) function by using
        weighted sum
        
        :param function: the function values
        :param type: np.array
        :returns: lower expectation value
        :rtype: float
        """
        lowerexpe=0.
        if function.__class__.__name__!='ndarray':
            raise Exception('Expecting a numpy array as argument')
        if function.size != self.nbDecision:
            raise Exception('number of elements incompatible with the frame size')
        if self.isproper()==0:
            raise Exception('Not a well-defined probability')
        minima=np.apply_along_axis(BelFun.minfunc, 1, self.focals, function)
        return np.dot(self.mass,minima)
    
    @staticmethod
    def minfunc(vec,function):
        return np.min(function[vec.astype(bool)])
    
    def __str__(self):
        """Print the mass function and the focal sets 
        """
        str3="\n"
        str3+="Mass value |  Focal set \n"
        str3+="-----------|----------------- \n"
        for j in range(np.shape(self.focals)[0]):
            str3+="   %.3f   |  " %self.mass[j]
            str3+= np.array_str(self.focals[j])
            str3+="\n"
        str3+="-----------|-----------------"
        str3+="\n"
        return str3

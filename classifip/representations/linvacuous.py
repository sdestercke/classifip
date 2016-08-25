import numpy as np
from credalset import CredalSet
from math import fabs

class LinVac(CredalSet):
    """Class of linear vacuous model: a single probability
    distribution + epsilon index.
    
    :param proba: a 1xn array containing probability
    :type proba: :class:`~numpy.array`
    :param epsilon: a real value in [0,1]
    :type epsilon: real 
    :param nbDecision: number of elements of the space
    :type nbDecision: integer
    
    >>> from numpy import array
    >>> ip=array([0.2,0.3,0.5])
    >>> from classifip.representations.linvacuous import LinVac
    >>> epsilon=0.3
    >>> linvac=LinVac(ip,epsilon)
    >>> print linvac
                     y0    y1    y2 
               --------------------
    Proba weights | 0.200 0.300 0.500
    
    Epsilon value= 0.300
    
    >>> subset=array([0.,1.,1.])
    >>> linvac.getlowerprobability(subset)
    0.55999999999999994
    >>> obj=array([1.,2.,3.])
    >>> linvac.getlowerexpectation(obj)
    1.9099999999999999
    """
    
    def __init__(self,proba,epsilon):
        """Instanciate proba values and epsilon
        
        :param lproba: a 1xn array containing upper (1st row) and lower bounds
        :type lproba: :class:`~numpy.array`
        """
        if proba.__class__.__name__ != 'ndarray':
            raise Exception('Expecting a numpy array as argument')
        if epsilon >= 1 or epsilon <= 0:
            raise Exception('Unadequate value of discounting')
        self.proba=proba
        self.nbDecision=proba.size
        self.epsilon=epsilon
        if fabs(proba.sum()-1.) > 1e-7:
            raise Exception('proba weights do not sum to one')

    def isproper(self):
        """Check if probability is well-defined. 
        
        :returns: 0 (empty/incur sure loss) or 1 (non-empty/avoid sure loss).
        :rtype: integer
        
        """
        #check summing to one and epsilon value    
        if fabs(self.proba.sum()-1.) > 1e-7 or self.epsilon <= 0:
            return 0
        return 1

    def isreachable(self):
        """Check if probability is reachable
        
        :returns: 0 (not coherent/tight) or 1 (tight/coherent).
        :rtype: integer
        
        """
        if self.epsilon >= 1:
            return 0
        return 1
    
    def setreachableprobability(self):
        """Make the bounds reachable.
        
        """    
        if self.isproper()==0:
            raise Exception('ill-defined probability inducing empty set: operation not possible')
        self.epsilon=min(self.epsilon,1)

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
            raise Exception('Not a well-defined probability')
        if np.all(subset):
            return 1
        else:
            return np.dot(self.proba,subset)*(1-self.epsilon)

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
        function=function.astype(float)
        return (1-self.epsilon)*np.dot(self.proba,function)+self.epsilon*min(function)
    
    def __str__(self):
        """Print the current model 
        """  
        str1="Proba weights |"
        str3="              "
        i=0
        for j in range(self.nbDecision):
            str3+="   y%d " %i
            str1+=" %.3f" % self.proba[j]
            i+=1
        str3+="\n"
        str3+="           "
        str3+="--------------------"
        str3+="\n"
        str3+=str1
        str3+="\n"
        str3+="\n"
        str3+="Epsilon value="
        str3+=" %.3f" % self.epsilon
        str3+="\n"
        return str3


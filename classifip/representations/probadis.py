import numpy as np
from credalset import CredalSet
from math import fabs

class ProbaDis(CredalSet):
    """Class of (discrete) probability distribution: a single probability
    distribution.
    
    :param proba: a 1xn array containing probability
    :type proba: :class:`~numpy.array`
    :param nbDecision: number of elements of the space
    :type nbDecision: integer
    """
    
    def __init__(self,proba):
        """Instanciate proba values
        
        :param proba: a 1xn array containing upper (1st row) and lower bounds
        :type proba: :class:`~numpy.array`
        """
        if proba.__class__.__name__ != 'ndarray':
            raise Exception('Expecting a numpy array as argument')
        self.proba=proba
        self.nbDecision=proba.size
        if fabs(proba.sum()-1.) > 1e-7:
            raise Exception('proba weights do not sum to one')

    def isproper(self):
        """Check if probability is well-defined. 
        
        :returns: 0 (empty/incur sure loss) or 1 (non-empty/avoid sure loss).
        :rtype: integer
        
        """
        #check inequality of bounds    
        if fabs(self.proba.sum()-1.) > 1e-7:
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
            raise Exception('ill-defined probability inducing empty set: operation not possible')

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
        return np.dot(self.proba,subset)

    def getupperprobability(self,subset):
        """Compute probability of an event expressed in binary code. 
        
        :param subset: the event of interest (a 1xn vector containing 1 for elements
            in the event, 0 otherwise.)
        :param type: np.array
        :returns: upper probability value
        :rtype: float
        
        """    
        return self.getlowerprobability(subset)
    
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
        return np.dot(self.proba,function)
    
    def __str__(self):
        """Print the current bounds 
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
        return str3

    def __and__(self,other):
        """Compute the intersection of two probabilities
        """  
        if self.proba==other.proba:
            return ProbaDis(self.proba)
        else:
            raise Exception('empty intersection, unequal probabilities')
        
    def __add__(self,other):
        """Compute the average of two probability intervals
        """  
        fusedproba=np.zeros(self.nbDecision)
        fusedproba=np.mean([self.proba,other.proba],axis=0)
        return ProbaDis(fusedproba)
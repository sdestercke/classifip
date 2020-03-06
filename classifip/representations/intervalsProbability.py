import numpy as np
from classifip.representations.credalset import CredalSet

class IntervalsProbability(CredalSet):
    """Class of probability intervals: probabilistic bounds on singletons
    
    :param lproba: a 2xn array containing upper (1st row) and lower bounds
    :type lproba: :class:`~numpy.array`
    :param nbDecision: number of elements of the space
    :type nbDecision: integer
    
    >>> from numpy import array
    >>> ip=array([[0.5, 0.5, 0.5], [0.1, 0.1, 0.1]])
    >>> from classifip.representations.intervalsProbability import IntervalsProbability
    >>> intprob=IntervalsProbability(ip)
    >>> print(intprob)
                     y0    y1    y2 
               --------------------
    upper bound | 0.500 0.500 0.500`
    lower bound | 0.100 0.100 0.100
    
    >>> ip2=array([[0.4, 0.5, 0.6], [0., 0.1, 0.2]])
    >>> intprob2=IntervalsProbability(ip2)
    >>> print intprob & intprob2
                     y0    y1    y2 
               --------------------
    upper bound | 0.400 0.500 0.500
    lower bound | 0.100 0.100 0.200

    >>> print intprob | intprob2
                     y0    y1    y2 
               --------------------
    upper bound | 0.500 0.500 0.600
    lower bound | 0.000 0.100 0.100
    
    >>> print intprob + intprob2
                     y0    y1    y2 
               --------------------
    upper bound | 0.450 0.500 0.550
    lower bound | 0.050 0.100 0.150
    
    >>> ip3=array([[0.7, 0.5, 0.2], [0.4, 0.2, 0.1]])
    >>> intprob3=IntervalsProbability(ip3)
    >>> intprob3.isreachable()
    1
    >>> intprob3.getmaximindecision()
    0
    >>> intprob3.getmaximaxdecision()
    0
    >>> intprob3.getintervaldomdecision()
    array([ 1.,  1.,  0.])
    >>> intprob3.getmaximaldecision()
    array([ 1.,  1.,  0.])
    """

    def __init__(self, lproba, precision_decimal=16):
        """Instanciate probability interval bounds
        
        :param lproba: a 2xn array containing upper (1st row) and lower bounds
        :type lproba: :class:`~numpy.array`
        """
        if lproba.__class__.__name__ != 'ndarray':
            raise Exception('Expecting a numpy array as argument')
        if lproba[:, 1].size != 2:
            raise Exception('Array should contain two rows: top for upper prob, bottom for lower prob')
        if lproba.ndim != 2:
            raise Exception('Bad dimension of array: should contain 2 dimensions')
        self.lproba = lproba
        self.nbDecision = lproba[0].size
        # approximation due to precision decimal greater than 16 decimals
        lproba = np.around(lproba, decimals=precision_decimal)
        if np.all(lproba[0] >= lproba[1]) != 1:
            np.set_printoptions(precision=40, suppress=True)
            raise Exception('Some upper bounds lower than lower bounds', lproba)

    def isproper(self):
        """Check if probability intervals induce a non-empty probability set. 
        
        :returns: 0 (empty/incur sure loss) or 1 (non-empty/avoid sure loss).
        :rtype: integer
        
        """
        if self.lproba[1,:].sum()<=1 and self.lproba[0,:].sum()>=1:
            return 1
        else:
            return 0

    def getlowerprobability(self,subset):
        """Compute lower probability of an event expressed in binary code. 
        
        :param subset: the event of interest (a 1xn vector containing 1 for elements
            in the event, 0 otherwise.)
        :param type: np.array
        :returns: lower probability value
        :rtype: float
        
        """
        if subset.__class__.__name__!='ndarray':
            raise Exception('Expecting a numpy array as argument')
        if subset.size != self.nbDecision:
            raise Exception('Subset incompatible with the frame size')
        if self.isreachable()==0:
            self.setreachableprobability()
        lowerProbability=max(self.lproba[1,subset[:]==1].sum(),1-self.lproba[0,subset[:]==0].sum())
        return lowerProbability

    def getupperprobability(self,subset):
        """Compute upper probability of an event expressed in binary code. 
        
        :param subset: the event of interest (a 1xn vector containing 1 for elements
            in the event, 0 otherwise.)
        :param type: np.array
        :returns: upper probability value
        :rtype: float
        
        """    
        if subset.__class__.__name__!='ndarray':
            raise Exception('Expecting a numpy array as argument')
        if subset.size != self.nbDecision:
            raise Exception('Subset incompatible with the frame size')
        if self.isreachable()==0:
            self.setreachableprobability()
        upperProbability=min(self.lproba[0,subset[:]==1].sum(),1-self.lproba[1,subset[:]==0].sum())
        return upperProbability
    
    def getlowerexpectation(self,function):
        """Compute the lower expectation of a given (bounded) function by using
        the Choquet integral
        
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
        function=function.astype(float)
        sortedf=np.sort(function)
        indexedf=np.argsort(function)
        lowerexpe=lowerexpe+sortedf[0]
        for i in range(self.nbDecision)[1:]:
            addedval=sortedf[i]-sortedf[i-1]
            event=np.zeros(self.nbDecision)
            event[indexedf[i:]]=1
            lowerexpe=lowerexpe+addedval*self.getlowerprobability(event)
        return lowerexpe

    def isreachable(self):
        """Check if the probability intervals are reachable (are coherent)
        
        :returns: 0 (not coherent/tight) or 1 (tight/coherent).
        :rtype: integer
        
        """    
        for i in range(self.nbDecision):
            subset=np.ones(self.nbDecision)
            subset[i]=0
            if self.lproba[0,i] + self.lproba[1,subset[:]==1].sum()  > 1.0:
                return 0
            if self.lproba[1,i] + self.lproba[0,subset[:]==1].sum() < 1.0:
                return 0
        return 1

    def setreachableprobability(self):
        """Make the bounds reachable.
        
        """    
        if self.isproper()==1:
            lreachableProba=np.zeros((2,self.nbDecision))
            for i in range(self.nbDecision):
                subset=np.ones(self.nbDecision)
                subset[i]=0
                lb=max(self.lproba[1,i],1-self.lproba[0,subset[:]==1].sum())
                ub=min(self.lproba[0,i],1-self.lproba[1,subset[:]==1].sum())
                lreachableProba[1,i]=lb
                lreachableProba[0,i]=ub
            self.lproba[:]=lreachableProba[:]
        else:
            raise Exception('intervals inducing empty set: operation not possible')
            

    def __str__(self):
        """Print the current bounds 
        """  
        str1,str2="upper bound |","lower bound |"
        str3="              "
        i=0
        for interval in range(self.nbDecision):
            str3+="   y%d " %i
            str1+=" %.3f" % self.lproba[0,interval]
            str2+=" %.3f" % self.lproba[1,interval]
            i+=1
        str3+="\n"
        str3+="           "
        str3+="--------------------"
        str3+="\n"
        str3+=str1
        str3+="\n"
        str3+=str2
        str3+="\n"
        return str3
    
    def __and__(self,other):
        """Compute the intersection of two probability intervals
        """  
        mini=np.maximum(self.lproba[1,:],other.lproba[1,:])
        maxi=np.minimum(self.lproba[0,:],other.lproba[0,:])
        if mini.sum() >= 0.9999999 or maxi.sum() <= 0.9999999:
            raise Exception('empty intersection')
        for i in range(self.nbDecision):
            if mini[i] >= maxi[i] - 0.0000001:
                raise Exception('empty intersection')
        fusedproba=np.zeros((2,self.nbDecision))
        fusedproba[1,:]=mini
        fusedproba[0,:]=maxi
        result=IntervalsProbability(fusedproba)
        result.setreachableprobability()
        return result
    
    def __or__(self,other):
        """Compute the union of two probability intervals
        """  
        fusedproba=np.zeros((2,self.nbDecision))
        fusedproba[1,:]=np.minimum(self.lproba[1,:],other.lproba[1,:])
        fusedproba[0,:]=np.maximum(self.lproba[0,:],other.lproba[0,:])
        result=IntervalsProbability(fusedproba)
        return result
    
    def __add__(self,other):
        """Compute the average of two probability intervals
        """  
        fusedproba=np.zeros((2,self.nbDecision))
        fusedproba[1,:]=np.mean([self.lproba[1,:],other.lproba[1,:]],axis=0)
        fusedproba[0,:]=np.mean([self.lproba[0,:],other.lproba[0,:]],axis=0)
        result=IntervalsProbability(fusedproba)
        return result

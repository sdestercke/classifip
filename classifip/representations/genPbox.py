import numpy as np
from credalset import CredalSet
from math import fabs

class GenPbox(CredalSet):
    """Class of (discrete) generalized p-box: probabilistic bounds on nested
    events of a pre-ordered space. Bounds should be monotically increasing with
    a (reached) maximum of one
    
    :param lproba: a 2xn array containing upper (1st row) and lower bounds
    :type lproba: :class:`~numpy.array`
    :param nbDecision: number of elements of the space
    :type nbDecision: integer
    
    >>> from numpy import array
    >>> ip=array([[0.5, 0.7, 1.], [0.3, 0.5, 1.]])
    >>> from classifip.representations.genPbox import GenPbox
    >>> pbox=GenPbox(ip)
    >>> print(pbox)
                     y0    y1    y2 
               --------------------
    upper bound | 0.500 0.700 1.000`
    lower bound | 0.300 0.500 1.000
    
    >>> pbox.getlowerprobability(subset)
    0.5
    >>> pbox.getupperprobability(subset)
    0.69999999999999996
    >>> pbox.isreachable()
    1
    >>> obj=array([1.,2.,3.])
    >>> pbox.getupperexpectation(obj)
    2.2000000000000002
    
    .. todo::
    
        * allow user to give list of list as well as np.array
    """
    
    def __init__(self,lproba):
        """Instanciate pbox bounds
        
        :param lproba: a 2xn array containing upper (1st row) and lower bounds
        :type lproba: :class:`~numpy.array`
        """
        if lproba.__class__.__name__ != 'ndarray':
            raise Exception('Expecting a numpy array as argument')
        if lproba[:,1].size != 2:
            raise Exception('Array should contain two rows: top for upper prob, bottom for lower prob')
        if lproba.ndim != 2:
            raise Exception('Bad dimension of array: should contain 2 dimensions')
        self.lproba=lproba
        self.nbDecision=lproba[0].size
        if np.all(lproba[0] >=lproba[1]) != 1:
            raise Exception('Some upper bounds lower than lower bounds')
        if (lproba[0,self.nbDecision-1] != 1) or (lproba[1,self.nbDecision-1] != 1):
            raise Exception('last element bounds should be one')

    def isproper(self):
        """Check if generalized p-box induce a non-empty probability set. 
        
        :returns: 0 (empty/incur sure loss) or 1 (non-empty/avoid sure loss).
        :rtype: integer
        
        """
        #check inequality of bounds    
        if all(x<=y for x, y in zip(self.lproba[1,:], self.lproba[0,:]))==False:
            return 0
        return 1

    def isreachable(self):
        """Check if generalized p-box is reachable (is coherent)
        
        :returns: 0 (not coherent/tight) or 1 (tight/coherent).
        :rtype: integer
        
        """
        #check that bounds are monotically increasing        
        if all(x<=y for x, y in zip(self.lproba[1,:], self.lproba[1,1:]))==False:
            return 0
        if all(x<=y for x, y in zip(self.lproba[0,:], self.lproba[0,1:]))==False:
            return 0
        return 1
    
    def setreachableprobability(self):
        """Make the bounds reachable.
        
        """    
        if self.isproper()==1:
            for i in range(self.nbDecision-1):
                if self.lproba[0,i] > self.lproba[0,i+1]:
                    self.lproba[0,i+1]=self.lproba[0,i]
                if self.lproba[1,i] > self.lproba[1,i+1]:
                    self.lproba[1,i+1]=self.lproba[1,i]
        else:
            raise Exception('p-box inducing empty set: operation not possible')

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
        lowerprobability=0.
        start=0
        for i in range(self.nbDecision):
            if start==0 and subset[i]==1 and i==0:
                up=0
                start=1
            if start==0 and subset[i]==1 and i!=0:
                up=self.lproba[0,i-1]
                start=1
            if start==1 and subset[i]==0:
                lowerprobability=lowerprobability+max(0,self.lproba[1,i-1]-up)
                start=0
            if start==1 and i==self.nbDecision-1:
                lowerprobability=lowerprobability+max(0,self.lproba[1,i]-up)
        return lowerprobability

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
        subset=1-subset
        compsub=np.array([fabs(i) for i in subset]).astype(int)
        upperProbability=1-self.getlowerprobability(compsub)
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


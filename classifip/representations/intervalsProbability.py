import numpy as np

class IntervalsProbability(object):
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
    
    .. todo::
    
        * allow user to give list of list as well as np.array
    """
    
    def __init__(self,lproba):
        """Instanciate probability interval bounds
        
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
            
    def nc_maximin_decision(self):
        """Return the maximin classification decision (nc: no costs)

        :returns: the index of the maximin class
        :rtype: integer
        
        """
        if self.isreachable()==0:
            self.setreachableprobability()
        return self.lproba[1,:].argmax()
        
    def nc_maximax_decision(self):
        """Return the maximax classification decision (nc: no costs)
        
        :returns: the index of the maximax class
        :rtype: integer
        
        """
        if self.isreachable()==0:
            self.setreachableprobability()
        return self.lproba[0,:].argmax()
        
    def nc_hurwicz_decision(self,alpha):
        """Return the maximax classification decision (nc: no costs)
        
        :param alpha: the optimism index :math:`\\alpha` between 1 (optimistic)
            and 0 (pessimistic)
        :param type: float
        :return: the index of the hurwicz class
        :rtype: integer
        
        """
        if self.isreachable()==0:
            self.setreachableprobability()
        hurwicz=alpha*self.lproba[0,:]+(1-alpha)*self.lproba[1,:]
        return hurwicz.argmax()
        
    def nc_maximal_decision(self):
        """Return the classification decisions using maximality (nc: no costs)
        
        :return: the set of optimal classes (under maximality) as a 1xn vector
            where indices of optimal classes are set to one
        :rtype: np.array
        
        """
        if self.isreachable()==0:
            self.setreachableprobability()
        maximality_classe=np.ones(self.nbDecision)
        for i in range(self.nbDecision):
            for j in range(self.nbDecision):
                if i != j and maximality_classe[i] == 1 and maximality_classe[j] == 1:
                    if -self.lproba[0,j]+self.lproba[1,i] > 0:
                        maximality_classe[j]=0
        return maximality_classe
    
    def nc_intervaldom_decision(self):
        """Return the classification decisions using interval dominance (nc: no costs)
        
        :return: the set of optimal classes (under int. dom.) as a 1xn vector
            where indices of optimal classes are set to one
        :rtype: :class:`~numpy.array`
        
        """
        if self.isreachable()==0:
            self.setreachableprobability()
        intervaldom_classe=np.ones(self.nbDecision)
        maxlower=self.lproba[1,:].max()
        for i in range(self.nbDecision):
                if self.lproba[0,i] < maxlower:
                        intervaldom_classe[i]=0
        return intervaldom_classe

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
        
class SetIntProba(object):
    """Class to handle sets of Int Proba

    Argument:
    intlist -- a mx2xn array containing upper (1st row) and lower (2nd row) bounds
               of the m probability intervals
               
               dim 1: index of probability set
               dim 2: lower or upper prob bounds
               dim 3: values of bounds on each element
               
    >>> from numpy import array
    >>> setproba=array([[[0.6,0.5,0.2],[0.4,0.3,0.]],[[0.55,0.55,0.2],[0.35,0.35,0.]],
    ... [[0.5,0.2,0.6],[0.3,0.,0.4]],[[0.35,0.6,0.35],[0.15,0.4,0.15]]])
    >>> from classifip.representations.intervalsProbability import SetIntProba
    >>> setip=SetIntProba(setproba)
    >>> setip.intlist
    array([[[ 0.6 ,  0.5 ,  0.2 ],
            [ 0.4 ,  0.3 ,  0.  ]],
    
           [[ 0.55,  0.55,  0.2 ],
            [ 0.35,  0.35,  0.  ]],
    
           [[ 0.5 ,  0.2 ,  0.6 ],
            [ 0.3 ,  0.  ,  0.4 ]],
    
           [[ 0.35,  0.6 ,  0.35],
            [ 0.15,  0.4 ,  0.15]]])
            
    """
    
    def __init__(self,intlist):
        if intlist.__class__.__name__ != 'ndarray':
            raise Exception('Expecting a numpy array as argument')
        if intlist.ndim != 3:
            raise Exception('Expecting a 3-dimensional array')
        self.intlist=intlist
        self.nbProbInt=intlist[:,0,0].size
        self.nbDecision=intlist[0,0].size
        
    def arecompatible(self):
        """Check whether the set of probability intervals are compatible, i.e., if the conjunction is non-empty.
        
        Return 1 if non-empty, 0 if empty
        
        >>> from numpy import array
        >>> from classifip.representations.intervalsProbability import SetIntProba
        >>> setproba=array([[[0.6,0.5,0.2],[0.4,0.3,0.]],[[0.55,0.55,0.2],[0.35,0.35,0.]]])
        >>> setip.arecompatible()
        1
        >>> setproba=array([[[0.6,0.5,0.2],[0.4,0.3,0.]],[[0.55,0.55,0.2],[0.35,0.35,0.]],
        ... [[0.5,0.2,0.6],[0.3,0.,0.4]],[[0.35,0.6,0.35],[0.15,0.4,0.15]]])
        >>> setip=SetIntProba(setproba)
        >>> setip.arecompatible()
        0
        """
        comp=1
        min=self.intlist[:,1,:].max(axis=0)
        max=self.intlist[:,0,:].min(axis=0)
        if min.sum() >= 0.9999999 or max.sum() <= 0.9999999:
            comp=0
        for i in range(self.nbDecision):
            if min[i] >= max[i] - 0.0000001:
                comp=0
        return comp
            
    def conjunction(self):
        """Perform a conjunctive merging of the set of probability intervals
        
        Return a possibly non-proper IntervalsProbability class object.
        
        >>> from numpy import array
        >>> from classifip.representations.intervalsProbability import SetIntProba
        >>> setproba=array([[[0.6,0.5,0.2],[0.4,0.3,0.]],[[0.55,0.55,0.2],[0.35,0.35,0.]]])
        >>> setip=SetIntProba(setproba)
        >>> print(setip.conjunction())
                         y0    y1    y2 
                   --------------------
        upper bound | 0.550 0.500 0.200
        lower bound | 0.400 0.350 0.000

        """
        if self.arecompatible() == 0:
            raise Exception('Probability intervals not compatible, conjunction empty') 
        fusedproba=np.zeros((2,self.nbDecision))
        for i in range(self.nbDecision):
            subset=np.ones(self.nbDecision)
            subset[i]=0
            lb=max(self.intlist[:,1,i].max(),1-self.intlist[:,0,subset[:]==1].min(axis=0).sum())
            ub=min(self.intlist[:,0,i].min(),1-self.intlist[:,1,subset[:]==1].max(axis=0).sum())
            fusedproba[1,i]=lb
            fusedproba[0,i]=ub
        result=IntervalsProbability(fusedproba)
        return result
        
    def disjunction(self):
        """Perform a disjunctive merging of the set of probability intervals
        
        Return an IntervalsProbability class object.
        
        >>> from numpy import array
        >>> from classifip.representations.intervalsProbability import SetIntProba
        >>> setproba=array([[[0.6,0.5,0.2],[0.4,0.3,0.]],[[0.55,0.55,0.2],[0.35,0.35,0.]],
        ... [[0.5,0.2,0.6],[0.3,0.,0.4]],[[0.35,0.6,0.35],[0.15,0.4,0.15]]])
        >>> setip=SetIntProba(setproba)
        >>> print(setip.disjunction())
                         y0    y1    y2 
                   --------------------
        upper bound | 0.600 0.600 0.600
        lower bound | 0.150 0.000 0.000

        """
        fusedproba=np.zeros((2,self.nbDecision))
        for i in range(self.nbDecision):
            subset=np.ones(self.nbDecision)
            subset[i]=0
            lb=self.intlist[:,1,i].min()
            ub=self.intlist[:,0,i].max()
            fusedproba[1,i]=lb
            fusedproba[0,i]=ub
        result=IntervalsProbability(fusedproba)
        return result
    
    def average(self):
        """Perform an average (equal weights) of the set of probability intervals
        
        Return an IntervalsProbability class object.
        
        >>> from numpy import array
        >>> from classifip.representations.intervalsProbability import SetIntProba
        >>> setproba=array([[[0.6,0.5,0.2],[0.4,0.3,0.]],[[0.55,0.55,0.2],[0.35,0.35,0.]],
        ... [[0.5,0.2,0.6],[0.3,0.,0.4]],[[0.35,0.6,0.35],[0.15,0.4,0.15]]])
        >>> setip=SetIntProba(setproba)
        >>> print(setip.average())
                         y0    y1    y2 
                   --------------------
        upper bound | 0.500 0.463 0.338
        lower bound | 0.300 0.262 0.138


        """
        fusedproba=np.zeros((2,self.nbDecision))
        for i in range(self.nbDecision):
            subset=np.ones(self.nbDecision)
            subset[i]=0
            lb=self.intlist[:,1,i].sum()/self.nbProbInt
            ub=self.intlist[:,0,i].sum()/self.nbProbInt
            fusedproba[1,i]=lb
            fusedproba[0,i]=ub
        result=IntervalsProbability(fusedproba)
        return result



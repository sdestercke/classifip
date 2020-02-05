import numpy as np
from scipy.sparse import dok_matrix

class Scores(object):
    """Class containing (imprecise) scores, usually obtained by a voting procedure

    :param scores: array storing lower and upper score values
    :type scores: :class:`~numpy.array`
    :param nbDecision: number of elements of the space
    :type nbDecision: integers
    
    >>> from numpy import array
    >>> score_values=array([[0.5, 1], [0.3, 0.7], [0.8, 0.8]])
    >>> from classifip.representations.voting import Scores
    >>> score_class=Scores(score_values)
    >>> print(score_class)
                 y0    y1    y2 
               --------------------
    upper bound | 1.000 0.700 0.800
    lower bound | 0.500 0.300 0.800
    
    """
    
    def __init__(self,sc_values):
        """instanciate a vote structure
        
        :param sc_values: array providing lower and upper score values
        :type sc_values: :class:`~numpy.array`
        
        """
        if sc_values.__class__.__name__ != 'ndarray':
            raise Exception('Expecting a numpy array as argument')
        if sc_values[0,:].size != 2:
            raise Exception('Array should contain two columns: minimum and maximal values')
        if sc_values.ndim != 2:
            raise Exception('Bad dimension of array: should contain 2 dimensions')
        self.scores=sc_values
        self.nbDecision=sc_values[:,0].size
        if np.all(sc_values[:,1] >=sc_values[:,0]) != 1:
            raise Exception('Some minimal values higher than maximal ones')

    def isproper(self):
        """coherence with generic representation class
        """
        print("Not an IP representation, no proper properties")
        return 1

    def getlowerprobability(self,subset):
        """coherence with generic representation class
        """
        print("Not an IP representaiton, cannot compute lower prob.")
        return 0

    def getupperprobability(self,subset):
        """coherence with generic representation class
        """
        print("Not an IP representaiton, cannot compute upper prob.")
        return 1
        
    def isreachable(self):
        """coherence with generic representation class
        """        
        print("Not an IP representaiton, no reachability property")
        return 1

    def setreachableprobability(self):
        """coherence with generic representation class
        """   
        print("Not an IP representaiton, cannot make reachable")        
            
    def nc_maximin_decision(self):
        """Return the maximin classification decision (nc: no costs)
        
        :returns: the index of the maximin class
        :rtype: integer
        
        """
        return self.scores[:,0].argmax()
        
    def nc_maximax_decision(self):
        """Return the maximax classification decision (nc: no costs)
        
        :returns: the index of the maximax class
        :rtype: integer
        
        """
        return self.scores[:,1].argmax()
        
    def nc_hurwicz_decision(self,alpha):
        """Return the hurwicz classification decision (nc: no costs)
        
        :param alpha: the optimism index :math:`\\alpha` between 1 (optimistic)
            and 0 (pessimistic)
        :param type: float
        :return: the index of the hurwicz class
        :rtype: integer
        """
        hurwicz=alpha*self.scores[:,1]+(1-alpha)*self.scores[:,0]
        return hurwicz.argmax()
        
    def nc_maximal_decision(self):
        """coherence with generic representation class
        """  
        print("Not an IP representaiton, cannot compute maximality")
        return 1
    
    def nc_intervaldom_decision(self):
        """Return the classification decisions that are optimal under interval dominance (nc: no costs)
        
        :return: the set of optimal classes (under int. dom.) as a 1xn vector
            where indices of optimal classes are set to one
        :rtype: :class:`~numpy.array`
        
        """
        intervaldom_classe=np.ones(self.nbDecision)
        maxlower=self.scores[:,0].max()
        for i in range(self.nbDecision):
                if self.scores[i,1] < maxlower:
                        intervaldom_classe[i]=0
        return intervaldom_classe
    
    def rank_intervaldom(self):
        """Return a sparse matrix encoding partial ordering obtained by Int. dominance

        :return: a preference matrix of labels based on interval dominance.
            usually provides a partial order
        :rtype: :class:`~scipy.sparse.dok_matrix`
                
        """
        rank=dok_matrix((self.nbDecision,self.nbDecision))
        for i in range(self.nbDecision):
            for j in range(self.nbDecision):
                if self.scores[j,1] < self.scores[i,0]:
                    rank[i,j] = 1
                elif self.scores[i,1] < self.scores[j,0]:
                    rank[j,i] = 1
        return rank
    
    def rank_maximin(self):
        """Return a sparse matrix encoding ordering obtained by maximin
        
        :return: a preference matrix of labels based on maximin, returning a
            complete order
        :rtype: :class:`~scipy.sparse.dok_matrix`
               
        """
        rank=dok_matrix((self.nbDecision,self.nbDecision))
        for i in range(self.nbDecision):
            for j in range(self.nbDecision):
                if self.scores[j,0] < self.scores[i,0]:
                    rank[i,j] = 1
                elif self.scores[i,0] < self.scores[j,0]:
                    rank[j,i] = 1
        return rank
    
    def rank_maximax(self):
        """Return a sparse matrix encoding ordering obtained by maximax
        
        :return: a preference matrix of labels based on maximax, returning a
            complete order
        :rtype: :class:`~scipy.sparse.dok_matrix`
        
        """
        rank=dok_matrix((self.nbDecision,self.nbDecision))
        for i in range(self.nbDecision):
            for j in range(self.nbDecision):
                if self.scores[j,1] < self.scores[i,1]:
                    rank[i,j] = 1
                elif self.scores[i,1] < self.scores[j,1]:
                    rank[j,i] = 1
        return rank
    
    def rank_hurwicz(self,alpha):
        """Return a sparse matrix encoding ordering obtained by hurwicz crit.

        :param alpha: the optimism index :math:`\\alpha` between 1 (optimistic)
            and 0 (pessimistic)
        :param type: float
        :return: a preference matrix of labels based on maximin, returning a
            complete order
        :rtype: :class:`~scipy.sparse.dok_matrix`
        
        """
        rank=dok_matrix((self.nbDecision,self.nbDecision))
        hurwicz=alpha*self.scores[:,1]+(1-alpha)*self.scores[:,0]
        for i in range(self.nbDecision):
            for j in range(self.nbDecision):
                if hurwicz[j] < hurwicz[i]:
                    rank[i,j] = 1
                elif hurwicz[i] < hurwicz[j]:
                    rank[j,i] = 1
        return rank
    
    def multilab_dom(self):
        """Return an array stating if a label is in the set of labels, is not or
        if it is not known whether it belongs to the set of labels.
        
        :return: the set of potential labels as a 1xn vector
            where 1=in the set of labels, 0=not in the set of labels, -1: not
            known. 
        :rtype: :class:`~numpy.array`
        """
        multilab_classe = np.ones(self.nbDecision, dtype=int)
        for i in range(self.nbDecision):
            if self.scores[i, 0] > 0.5:
                multilab_classe[i] = 1
            elif self.scores[i, 1] < 0.5:
                multilab_classe[i] = 0
            else:
                multilab_classe[i] = -1
        return multilab_classe
        
    
    def multilab_hurwicz(self,alpha):
        """Return an array stating if a label is in the set of labels according
        to hurwicz-like criterion with optimistic index :math:`\\alpha`
        
        :param alpha: the optimism index :math:`\\alpha` between 1 (optimistic)
            and 0 (pessimistic)
        :param type: float
        :return: the set of potential labels as a 1xn vector
            where 1=in the set of labels, 0=not in the set of labels
        :rtype: :class:`~numpy.array`
        """

        multilab_classe=np.ones(self.nbDecision)
        hurwicz=alpha*self.scores[:,1]+(1-alpha)*self.scores[:,0]
        for i in range(self.nbDecision):
                if hurwicz[i]  > 0.5:
                    multilab_classe[i]=1
                else:
                    multilab_classe[i]=0
        return multilab_classe
                    
        

    def __str__(self):
        """Print the current bounds 
        """  
        str1,str2="upper bound |","lower bound |"
        str3="              "
        i=0
        for interval in range(self.nbDecision):
            str3+="   y%d " %i
            str1+=" %.3f" % self.scores[interval,1]
            str2+=" %.3f" % self.scores[interval,0]
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


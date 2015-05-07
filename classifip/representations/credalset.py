import numpy as np
from cvxopt import matrix,solvers
# set accuracy tolerance to default values
solvers.options['show_progress']=False
solvers.options['abstol']=1e-7
solvers.options['reltol'] = 1e-6
solvers.options['feastol'] = 1e-7
from cvxopt.blas import dot
from math import fabs

class CredalSet(object):
    """Class of credal set
    
    :param const: an array where each row is a linear constraint on proba masses
        specifying the upper bound constraint (upper expectation)
    :type const: :class:`~numpy.array`
    :param nbDecision: number of elements of the space
    :type nbDecision: integer
    
    .. warning::
    
        As cvxopt uses interior-point methods, there may be some problems with
        small credal sets. This may be solved in two ways: modifying the code
        to use glpk (if installed) and the simplex method rather than native
        cvxopt method or modifying the solver option to reduce tolerance
        thresholds.
    
    >>> from classifip.representations import credalset
    >>> from numpy import array
    >>> Test=credalset.CredalSet(4)
    >>> Constraints=array([[-1.,1.,0.,0.,0.],[0.,-1.,1.,0.,0.],[0.,0.,-1.,1.,0]])
    >>> Test.addconstraints(Constraints)
    >>> Test.isproper()
    1
    >>> Test.isreachable()
    1
    >>> Test.addconstraints(array([[0.,1.,0.,0.,0.75]]))
    >>> Test.isreachable()
    0
    >>> Test.setreachableprobability()
    0
    >>> Test.isreachable()
    1
    >>> subset=array([1.,0.,0.,1.])
    >>> subset2=array([0.,1.,1.,0.])
    >>> Test.getlowerprobability(subset)
    0.33333333201119253
    >>> Test.getupperprobability(subset2)
    0.6666666679888072
    >>> Test.getmaximaxdecision()
    0
    >>> Test.getintervaldomdecision()
    array([ 1.,  1.,  1.,  1.])
    >>> Test.getmaximaldecision()
    array([ 1.,  1.,  1.,  1.])
    >>> Test.gethurwiczdecision(0.2)
    0
    >>> Test.getmaximindecision()
    0
    >>> Costs=array([[2.,3.3,0.,6.],[1.,5.,3.,2.],[4.,1.,4.,3.],[0.9,0.9,0.9,0.9]])
    >>> Test.getmaximaxdecision(Costs)
    2
    >>> Test.getintervaldomdecision(Costs)
    array([ 1.,  1.,  1.,  0.])
    >>> Test.getmaximaldecision(Costs)
    array([ 1.,  1.,  1.,  0.])
    >>> Test.gethurwiczdecision(0.2,Costs)
    2
    >>> Test.getmaximindecision(Costs)
    2

    """
    
    def __init__(self,spacesize):
        """Instanciate an empty credal set
        
        :param spacesize: the size of the space on which is defined the credal set
        :type spacesize: integer 
        """
        
        self.const=np.zeros((spacesize,spacesize+1))
        #fixing basic probability conditions
        for i in range(spacesize):
            self.const[i,i]=-1.
        self.nbDecision=spacesize

    def addconstraints(self,constmat):
        """Add constraints to the current ones, each constraint being a row of
        coefficients in the matrix finishing by the constraint upper bound.
        
        :param constmat: the matrix
        :type constmat: :class:`~numpy.array` 
        """
        self.const=np.vstack((self.const,constmat))
        
    def delconstraints(self,constdel):
        """Delete a constraint or a set of constraints specified by their indices
        in the constraint matrix (first "spacesize" constraints are positivity ones).
        
        :param constdel: index or set of indices of constraint to remove
        :type constdel: :class:`~numpy.array` 
        """
        self.const=np.delete(self.const,constdel,0)
    
    def isproper(self):
        """Check if credal set is proper. 
        
        :returns: 0 (empty/incur sure loss) or 1 (non-empty/avoid sure loss).
        :rtype: integer
        
        """       
        objective=np.zeros(self.nbDecision)
        objective[0]=1.
        solution = self.solvelowerexpectation(objective)
        
        if solution['status']!='optimal':
            return 0
        else:
            return 1
    
    @staticmethod
    def issubsetmask(array):
        for v in array:
            if v!=0. and v!=1.:
                return False
        return True
        

    def getlowerprobability(self,subset):
        """Compute lower probability of an event expressed in binary code. 
        
        :param subset: the event of interest (a 1xn vector containing 1 for elements
            in the event, 0 otherwise.)
        :param type: :class:`~numpy.array`
        :returns: lower probability value
        :rtype: float
        
        """
        if subset.__class__.__name__ != 'ndarray':
            raise Exception('Expecting a numpy array as argument')
        if subset.size != self.nbDecision:
            raise Exception('Subset incompatible with the frame size')
        if not CredalSet.issubsetmask(subset):
            raise Exception('Array is not 1/0 elements')
        solution = self.solvelowerexpectation(subset)
        
        if solution['status']!='optimal':
            return "NA"
        else:
            return dot(solution['x'],matrix(subset))
        

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
        if not CredalSet.issubsetmask(subset):
            raise Exception('Array is not 1/0 elements')
        solution = self.solvelowerexpectation(-subset)
        if solution['status']!='optimal':
            return "NA"
        else:
            return dot(solution['x'],matrix(subset))
        
    def isreachable(self):
        """Check if the probability intervals are reachable (are coherent)
        
        :returns: 0 (not coherent/tight) or 1 (tight/coherent).
        :rtype: integer
        
        """
        if self.isproper()==0:
            raise Exception('intervals inducing empty set: operation not possible')
        for i in range(self.nbDecision,self.const.shape[0]):
            obj=self.const[i,0:self.nbDecision]
            sol = self.getupperexpectation(obj)
            soldiff = fabs(sol-self.const[i,self.nbDecision])
            if  soldiff > solvers.options['abstol'] :
                return 0
        return 1

    def setreachableprobability(self):
        """Make the bounds reachable (compute the natural extension of given
        constraints).
        
        """    
        if self.isproper()==1:
            for i in range(self.const.shape[0]):
                obj=self.const[i,0:self.nbDecision]
                sol = self.getupperexpectation(obj)
            if  fabs(sol-self.const[i,self.nbDecision]) > solvers.options['abstol'] :
                self.const[i,self.nbDecision] = sol
                return 0
        else:
            raise Exception('intervals inducing empty set: operation not possible')
           
 
    def getmaximindecision(self,costs=None):
        """Return the maximin classification decision

        :param costs: the cost matrix entered as an np array
        :param type: np.array
        :returns: the index of the maximin class
        :rtype: integer
        
        """
        if costs is None:
            costs=np.identity(self.nbDecision)
        
        if costs.shape[1]!=self.nbDecision:
            raise Exception('bad numbers of columns in costs')

        if self.isreachable()==0:
            self.setreachableprobability()
        
        optdec=0
        maxlprob=0.
        for i in range(len(costs)):
            objective=costs[i]
            sol=self.getlowerexpectation(objective)
            if sol>maxlprob :
                maxlprob=sol
                optdec=i
        return optdec
        
    def getmaximaxdecision(self, costs=None):
        """Return the maximax classification decision

        :param costs: the cost matrix entered as an np array
        :param type: np.array
        :returns: the index of the maximax class
        :rtype: integer
        
        """
        if costs is None:
            costs=np.identity(self.nbDecision)
        
        if costs.shape[1]!=self.nbDecision:
            raise Exception('bad numbers of columns in costs')

        if self.isreachable()==0:
            self.setreachableprobability()
        
        optdec=0
        maxuprob=0.
        for i in range(len(costs)):
            objective=costs[i]
            sol=self.getupperexpectation(objective)
            if sol>maxuprob :
                maxuprob=sol
                optdec=i
        return optdec
        
    def gethurwiczdecision(self,alpha,costs=None):
        """Return the maximax classification decision

        :param costs: the cost matrix entered as an np array
        :param type: np.array
        :param alpha: the optimism index :math:`\\alpha` between 1 (optimistic)
            and 0 (pessimistic)
        :param type: float
        :return: the index of the hurwicz class
        :rtype: integer
        
        """
        if costs is None:
            costs=np.identity(self.nbDecision)
        
        if costs.shape[1]!=self.nbDecision:
            raise Exception('bad numbers of columns in costs')

        if self.isreachable()==0:
            self.setreachableprobability()
        
        optdec=0
        maxhurw=0.
        for i in range(len(costs)):
            objective=costs[i]
            usol=self.getupperexpectation(objective)
            lsol=self.getlowerexpectation(objective)
            hursol=(1-alpha)*lsol + alpha*usol 
            if hursol>maxhurw :
                maxhurw=hursol
                optdec=i
        return optdec
    
    def getmaximaldecision(self, costs=None):
        """Return the classification decisions using maximality
        
        :param costs: the cost matrix entered as an np array
        :param type: np.array
        :return: the set of optimal classes (under maximality) as a 1xn vector
            where indices of optimal classes are set to one
        :rtype: np.array
        
        """
        if costs is None:
            costs=np.identity(self.nbDecision)
        
        if costs.shape[1]!=self.nbDecision:
            raise Exception('bad numbers of columns in costs')
            
        if self.isreachable()==0:
            self.setreachableprobability()

        maximality_classe=np.ones(len(costs))
        for i in range(len(costs)):
            for j in range(i)+range(i+1,len(costs)):
                if maximality_classe[i] == 1 and maximality_classe[j] == 1:
                    objective=costs[i]-costs[j]
                    sol=self.getlowerexpectation(objective)
                    if sol > 0:
                        maximality_classe[j]=0
        return maximality_classe
    
    def getintervaldomdecision(self, costs=None):
        """Return the classification decisions using interval dominance
        
        :param costs: the cost matrix entered as an np array
        :param type: np.array
        :return: the set of optimal classes (under int. dom.) as a 1xn vector
            where indices of optimal classes are set to one
        :rtype: :class:`~numpy.array`
        
        """
        if costs is None:
            costs=np.identity(self.nbDecision)
        
        if costs.shape[1]!=self.nbDecision:
            raise Exception('bad numbers of columns in costs')
        
        if self.isreachable()==0:
            self.setreachableprobability()
        
        intervaldom_classe=np.ones(len(costs))
        maxlower=0.
        for i in range(len(costs)):
            objective=costs[i]
            sol=self.getlowerexpectation(objective)
            if sol>maxlower :
                maxlower=sol
        for i in range(len(costs)):
            objective=costs[i]
            sol=self.getupperexpectation(objective)
            if sol < maxlower:
                intervaldom_classe[i]=0
        return intervaldom_classe



    def solvelowerexpectation(self,obj):
        """Compute the solution of the linear program corresponding to the lower expectation of the given function
        
        :param obj: values of the function whose lower expectation is to be computed
        :param type: :class:`~numpy.array`
        :return: the solution of the lp
        :rtype: dictionary
        """

        if obj.__class__.__name__!='ndarray':
            raise Exception('Expecting a numpy array as argument')
        if obj.size != self.nbDecision:
            raise Exception('Number of values in obj incompatible with the frame size')

        solution = solvers.lp(matrix(obj),
                              matrix(self.const[:,0:self.nbDecision].copy()),
                              matrix(self.const[:,self.nbDecision].copy()),
                              A=matrix(1.,(1,self.nbDecision)),b=matrix(1.))
        return solution
    
    def getlowerexpectation(self,obj):
        """Compute the lower expectation of the given function
        
        :param obj: values of the function whose lower expectation is to be computed
        :param type: :class:`~numpy.array`
        :return: the lower expectation value
        :rtype: float
        """

        if(CredalSet.issubsetmask(obj)):
            return self.getlowerprobability(obj)
        else:
            solution=self.solvelowerexpectation(obj)
            return dot(solution['x'],matrix(obj))
    
    def getupperexpectation(self,obj):
        """Compute the upper expectation of the given function
        
        :param obj: values of the function whose upper expectation is to be computed
        :param type: :class:`~numpy.array`
        :return: the upper expectation value
        :rtype: float
        """
        
        lowexp=self.getlowerexpectation(-obj)
        return -lowexp

    def __str__(self):
        """Print the current contraints
        """
        str3=""
        i=0
        for interval in range(self.nbDecision):
            str3+="    y%d  " %i
            i+=1
        str3+=" |  Upper bound "
        str3+="\n"
        str3+="---------------------------------------------"
        for j in range(self.nbDecision,self.const.shape[0]):
            strconst=""
            for k in range(self.nbDecision):
                if self.const[j,k] < 0.:
                    strconst+=" %.3f " % self.const[j,k]
                if self.const[j,k] >= 0.:
                    strconst+="  %.3f " % self.const[j,k]
            strconst+=" |  %.3f " % self.const[j,self.nbDecision]
            str3+="\n"
            str3+=strconst
        return str3

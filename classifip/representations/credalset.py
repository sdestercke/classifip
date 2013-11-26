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
    >>> Test=credalset.CredalSet(4)
    >>> from numpy import array
    >>> Constraints=array([[-1.,1.,0.,0.,0.],[0.,-1.,1.,0.,0.],[0.,0.,-1.,1.,0]])
    >>> Test.addconstraints(Constraints)
    >>> Test.isproper()
    1
    >>> subset=array([1.,0.,0.,1.])
    >>> Test.getlowerprobability(subset)
    0.33333333616905164
    >>> Test.nc_maximax_decision()
    0
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
        solution = solvers.lp(matrix(objective),
                              matrix(self.const[:,0:self.nbDecision].copy()),
                              matrix(self.const[:,self.nbDecision].copy()),
                              A=matrix(1.,(1,self.nbDecision)),b=matrix(1.))
        
        if solution['status']!='optimal':
            return 0
        else:
            return 1    
        

    def getlowerprobability(self,subset):
        """Compute lower probability of an event expressed in binary code. 
        
        :param subset: the event of interest (a 1xn vector containing 1 for elements
            in the event, 0 otherwise.)
        :param type: :class:`~numpy.array`
        :returns: lower probability value
        :rtype: float
        
        """
        if subset.__class__.__name__!='ndarray':
            raise Exception('Expecting a numpy array as argument')
        if subset.size != self.nbDecision:
            raise Exception('Subset incompatible with the frame size')
        solution = solvers.lp(matrix(subset),
                              matrix(self.const[:,0:self.nbDecision].copy()),
                              matrix(self.const[:,self.nbDecision].copy()),
                              A=matrix(1.,(1,self.nbDecision)),b=matrix(1.))
        
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
        solution = solvers.lp(matrix(-subset),
                              matrix(self.const[:,0:self.nbDecision].copy()),
                              matrix(self.const[:,self.nbDecision].copy()),
                              A=matrix(1.,(1,self.nbDecision)),b=matrix(1.))
        print solution['x']
        if solution['status']!='optimal':
            return "NA"
        else:
            return dot(solution['x'],matrix(subset))
        
    def isreachable(self):
        """Check if the probability intervals are reachable (are coherent)
        
        :returns: 0 (not coherent/tight) or 1 (tight/coherent).
        :rtype: integer
        
        """
        if self.isproper()==1:
            raise Exception('intervals inducing empty set: operation not possible')
        for i in range(self.const.shape[0]):
            obj=self.const[i,0:self.nbDecision]
            solution = solvers.lp(matrix(-obj),
                                  matrix(self.const[:,0:self.nbDecision].copy()),
                                  matrix(self.const[:,self.nbDecision].copy()),
                                  A=matrix(1.,(1,self.nbDecision)),b=matrix(1.))
            soldiff = fabs(dot(solution['x'],matrix(obj))-const[i,self.nbDecision])
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
                solution = solvers.lp(matrix(-obj),
                                      matrix(self.const[:,0:self.nbDecision].copy()),
                                      matrix(self.const[:,self.nbDecision].copy()),
                                      A=matrix(1.,(1,self.nbDecision)),b=matrix(1.))
                sol=-dot(solution['x'],matrix(obj))
            if  fabs(sol-const[i,self.nbDecision]) > solvers.options['abstol'] :
                const[:,self.nbDecision] = sol
                return 0
        else:
            raise Exception('intervals inducing empty set: operation not possible')
            
    def nc_maximin_decision(self):
        """Return the maximin classification decision (nc: no costs)

        :returns: the index of the maximin class
        :rtype: integer
        
        """
        optdec=0
        maxlprob=0.
        for i in range(self.nbDecision):
            objective=np.zeros(self.nbDecision)
            objective[i]=1.
            solution = solvers.lp(matrix(objective),
                                  matrix(self.const[:,0:self.nbDecision].copy()),
                                  matrix(self.const[:,self.nbDecision].copy()),
                                  A=matrix(1.,(1,self.nbDecision)),b=matrix(1.))
            sol=dot(solution['x'],matrix(objective))
            if sol>maxlprob :
                maxlprob=sol
                optdec=i
        return optdec
        
    def nc_maximax_decision(self):
        """Return the maximax classification decision (nc: no costs)
        
        :returns: the index of the maximax class
        :rtype: integer
        
        """
        optdec=0
        maxuprob=0.
        for i in range(self.nbDecision):
            objective=np.zeros(self.nbDecision)
            objective[i]=1.
            solution = solvers.lp(matrix(-objective),
                                  matrix(self.const[:,0:self.nbDecision].copy()),
                                  matrix(self.const[:,self.nbDecision].copy()),
                                  A=matrix(1.,(1,self.nbDecision)),b=matrix(1.))
            sol=dot(solution['x'],matrix(objective))
            if sol>maxuprob :
                maxuprob=sol
                optdec=i
        return optdec
        
    def nc_hurwicz_decision(self,alpha):
        """Return the maximax classification decision (nc: no costs)
        
        :param alpha: the optimism index :math:`\\alpha` between 1 (optimistic)
            and 0 (pessimistic)
        :param type: float
        :return: the index of the hurwicz class
        :rtype: integer
        
        """
        optdec=0
        maxhurw=0.
        for i in range(self.nbDecision):
            objective=np.zeros(self.nbDecision)
            objective[i]=1.
            solution = solvers.lp(matrix(-objective),
                                  matrix(self.const[:,0:self.nbDecision].copy()),
                                  matrix(self.const[:,self.nbDecision].copy()),
                                  A=matrix(1.,(1,self.nbDecision)),b=matrix(1.))
            usol=dot(solution['x'],matrix(objective))
            solution = solvers.lp(matrix(objective),
                                  matrix(self.const[:,0:self.nbDecision].copy()),
                                  matrix(self.const[:,self.nbDecision].copy()),
                                  A=matrix(1.,(1,self.nbDecision)),b=matrix(1.))
            lsol=dot(solution['x'],matrix(objective))
            hursol=(1-alpha)*lsol + alpha*usol 
            if hursol>maxhurw :
                maxhurw=hursol
                optdec=i
        return optdec
    
    def nc_maximal_decision(self):
        """Return the classification decisions using maximality (nc: no costs)
        
        :return: the set of optimal classes (under maximality) as a 1xn vector
            where indices of optimal classes are set to one
        :rtype: np.array
        
        """
        maximality_classe=np.ones(self.nbDecision)
        for i in range(self.nbDecision):
            for j in range(i)+range(i+1,self.nbDecision):
                if maximality_classe[i] == 1 and maximality_classe[j] == 1:
                    objective=np.zeros(self.nbDecision)
                    objective[i]=1.
                    objective[j]=-1.
                    solution = solvers.lp(matrix(objective),
                                          matrix(self.const[:,0:self.nbDecision].copy()),
                                          matrix(self.const[:,self.nbDecision].copy()),
                                          A=matrix(1.,(1,self.nbDecision)),b=matrix(1.))
                    sol=dot(solution['x'],matrix(objective))
                    if sol > 0:
                        maximality_classe[j]=0
        return maximality_classe
    
    def nc_intervaldom_decision(self):
        """Return the classification decisions using interval dominance (nc: no costs)
        
        :return: the set of optimal classes (under int. dom.) as a 1xn vector
            where indices of optimal classes are set to one
        :rtype: :class:`~numpy.array`
        
        """
        intervaldom_classe=np.ones(self.nbDecision)
        maxlower=0.
        for i in range(self.nbDecision):
            objective=np.zeros(self.nbDecision)
            objective[i]=1.
            solution = solvers.lp(matrix(objective),
                                  matrix(self.const[:,0:self.nbDecision].copy()),
                                  matrix(self.const[:,self.nbDecision].copy()),
                                  A=matrix(1.,(1,self.nbDecision)),b=matrix(1.))
            sol=dot(solution['x'],matrix(objective))
            if sol>maxlower :
                maxlower=sol
        for i in range(self.nbDecision):
            objective=np.zeros(self.nbDecision)
            objective[i]=1.
            solution = solvers.lp(matrix(-objective),
                                  matrix(self.const[:,0:self.nbDecision].copy()),
                                  matrix(self.const[:,self.nbDecision].copy()),
                                  A=matrix(1.,(1,self.nbDecision)),b=matrix(1.))
            sol=dot(solution['x'],matrix(objective))
            if sol < maxlower:
                intervaldom_classe[i]=0
        return intervaldom_classe
    
    def getlowerexp(self,obj):
        """Compute the lower expectation of the given function
        
        :param obj: values of the function whose lower expectation is to be computed
        :param type: :class:`~numpy.array`
        :return: the lower expectation value
        :rtype: float
        """

        if obj.__class__.__name__!='ndarray':
            raise Exception('Expecting a numpy array as argument')
        if obj.size != self.nbDecision:
            raise Exception('Number of values in obj incompatible with the frame size')

        solution = solvers.lp(matrix(-objective),
                              matrix(self.const[:,0:self.nbDecision].copy()),
                              matrix(self.const[:,self.nbDecision].copy()),
                              A=matrix(1.,(1,self.nbDecision)),b=matrix(1.))
        sol=dot(solution['x'],matrix(objective))
        
        return sol
    
    def getupperexp(self,obj):
        """Compute the upper expectation of the given function
        
        :param obj: values of the function whose upper expectation is to be computed
        :param type: :class:`~numpy.array`
        :return: the upper expectation value
        :rtype: float
        """
        
        lowexp=self.getlowerexp(-obj)
        return -lowexp

    def __str__(self):
        """Print the current contraints
        """
        print self.const
        return 0

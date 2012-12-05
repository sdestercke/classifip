from .pairpip import get_binomial_int 
from ..dataset.arff import ArffFile
from scipy.spatial import kdtree, distance
from ..representations.credalset import CredalSet
import numpy as np
from scipy.sparse import dok_matrix
from scipy.misc import comb
from math import fabs

def class_to_prefmat(observed,labels):
    """return a dok_matrix transforming a class into its preference matrix
     
    :param observed: the observed class value
    :type observed: string
    :param labels: a list of possible class labels 
    :type labels: list of strings
    :returns: the preference matrix with row of class filled with 1
    :rtype: :class:`~scipy.sparse.dok_matrix`
    """
    
    rank=dok_matrix((len(labels),len(labels)))
    ref_index=labels.index(observed)
    for j in range(len(labels)):
        if j!=ref_index:
            rank[ref_index,j]=1.
    return rank

class LikOvo(object):
    """algorithm using binomial likelihood to build a one-vs-one binary classifier
    
    :param tree: a kd-tree structure storing learning set of features
    :type tree: :class:`~scipy.spatial.kdtree`
    :param trueclasses: store the observed classes
    :type trueclasses: list of classes
    :param classes: a list of possible class labels 
    :type classes: list of strings
    :param radius: radius of sphere including learning data for an instance
    :type radius: float
    :param normal: store information for data normalisation
        (if normalisation=True)
    :type normal: list
    
    .. note::
    
        Assumes that the class attribute is the last one in samples
        in the learning method

    .. todo::
    
        * Make it possible for the class to be in any column (retrieve index)
        * Approximate average distance for large data sets by sampling
    """
    
    
    def __init__(self):
        """Build an empty LikOvo structure
        """
        self.tree=None
        self.trueclasses=[]
        self.classes=[]
        self.radius=0.0
        self.normal=[]
        
    def learn(self,learndataset,likovo_normalise=True):
        """learn the tree structure required to perform evaluation
        
        :param learndataset: learning instances
        :type learndataset: :class:`~classifip.dataset.arff.ArffFile`
        :param likovo_normalise: normalise the input features or not
        :type likovo_normalise: boolean

        """
        self.classes=learndataset.attribute_data['class'][:]
        learndata=[row[0:len(row)-1] for row in learndataset.data]
        data_array=np.array(learndata).astype(float)
        if likovo_normalise == True:
            self.normal.append(True)
            span=data_array.max(axis=0)-data_array.min(axis=0)
            self.normal.append(span)
            self.normal.append(data_array.min(axis=0))
            data_array=(data_array-data_array.min(axis=0))/span
        else:
            self.normal.append(False)
            
        #Initalise radius as average distance between all learning instances
        if len(data_array) > 1000:
                valred=np.random.permutation(data_array)[0:1000]
                distances=distance.cdist(valred,valred)
        else:
                distances=distance.cdist(data_array,data_array)
        self.radius=distances.sum()/(2*(len(distances)**2-len(distances)))
        self.tree=kdtree.KDTree(data_array)
        self.trueclasses=[row[-1] for row in learndataset.data]
            
            
        
    def evaluate(self,testdataset,likovo_radius=None,
                 likovo_start=0.1,likovo_prec=0.01):
        """evaluate the instances provided and return a list of credal sets with
        confidence value required to obtain a proper solution. Final confidence value is
        obtained by dichotomic.
        
        :param likovo_radius: overcome default radius built during learning
        :type likovo_radius: float
        :param likovo_start: starting value for confidence intervals. If proper
            answer foun with this value, no dichotomic search is done.
            Should be not too far from zero to avoid too imprecise results
        :type likovo_start: float
        :type likovo_prec: precision required in the dichotomic search. Implicitly
            specify the number of required operation :math:`n` iteration equal a
            precision of :math:`0.5^n`
        :returns: for each value of pipp_confid, retuning voting scores
        :rtype: lists of :class:`~classifip.representations.voting.Scores`
        """
        
        if likovo_radius != None:
            self.radius=likovo_radius
        dataset=np.array(testdataset).astype(float)
        final=[]

        if self.normal[0] == True:
            dataset=(dataset-self.normal[2])/self.normal[1]
        
        for i in dataset:
            answers=[]
            #add every neighbours in the given radius
            result=dok_matrix((len(self.classes),len(self.classes)))
            if self.tree.query_ball_point(i,self.radius) !=[]:
                for ind in self.tree.query_ball_point(i,self.radius):
                    add_mat=class_to_prefmat(self.trueclasses[ind],self.classes)
                    result=result+add_mat
            cur_conf=likovo_start
            credal_res=CredalSet(len(self.classes))
            #construct first credal set with likovo_start
            for k in range(len(self.classes)):
                for l in range(k+1,len(self.classes)):
                    const=np.zeros((2,len(self.classes)+1))
                    interval=get_binomial_int(result[k,l]
                                        +result[l,k], result[k,l],cur_conf)
                    const[0,k]=interval[0]-1.
                    const[0,l]=interval[0]
                    const[1,k]=1.-interval[1]
                    const[1,l]=-interval[1]
                    credal_res.addconstraints(const)
            if credal_res.isproper()==1:
                answers.append(credal_res)
                answers.append(cur_conf)
            #start dichotomic search
            else:
                low_conf=0.
                up_conf=1.
                while fabs(up_conf - low_conf) > likovo_prec:
                    credal_res=CredalSet(len(self.classes))
                    cur_conf=(up_conf+low_conf)/2.
                    #building corresponding credal set
                    for k in range(len(self.classes)):
                        for l in range(k+1,len(self.classes)):
                            const=np.zeros((2,len(self.classes)+1))
                            interval=get_binomial_int(result[k,l]
                                            +result[l,k], result[k,l],cur_conf)
                            const[0,k]=interval[0]-1.
                            const[0,l]=interval[0]
                            const[1,k]=1.-interval[1]
                            const[1,l]=-interval[1]
                            credal_res.addconstraints(const)
                    if credal_res.isproper()==0:
                        low_conf=cur_conf
                    if credal_res.isproper()==1:
                        up_conf=cur_conf
                #check final result, if non proper, return empty set
                if credal_res.isproper()==0:
                    credal_res=CredalSet(len(self.classes))
                    cur_conf=1.
                answers.append(credal_res)
                answers.append(cur_conf)
            final.append(answers)
        
        return final
        

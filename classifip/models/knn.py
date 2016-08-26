from ..dataset.arff import ArffFile
from scipy.spatial import kdtree, distance
from ..representations.intervalsProbability import IntervalsProbability
import numpy as np
from math import exp

class IPKNN(object):
    """IPKNN implements a K-nearest neighbour method using lower previsions.
    
    If data are all precise, it returns
    :class:`~classifip.representations.intervalsProbability.IntervalsProbability`
    equivalent to a linear vacuous model. The method is based on [#destercke2012]_
 
    :param tree: kdtree structure storing learning data set instances
    :type tree: scipy.spatial.kdtree
    :param truelabels: store the true labels of learning instances
    :type truelabels: list of labels
    :param beta: exponent parameter used in discounting rate
    :type beta: positive float
    :param epsilon: base discounting rate
    :type epsilon: float between 0 and 1
    :param av_dist: average distances of members of a given class
    :type av_dist: float
    :param classes: list of class names
    
    .. note::
    
        * Assumes that the class attribute is the last one in samples in the learning method
        * If too many data, average distance approximated by sampling
    
    .. todo::
    
        * Make it possible for the class to be in any column (retrieve index)
    
    """
    
    
    def __init__(self):
        """Build an empty IPKNN structure
        """
        self.tree=None
        self.truelabels=[]
        self.beta=1.5
        self.epsilon=0.99
        self.classes=[]
        self.av_dist=[]
        
    def learn(self,learndataset):
        """learn the KNN structure required to evaluate new instances
        
        :param learndataset: learning instances
        :type learndataset: :class:`~classifip.dataset.arff.ArffFile`
        """
        self.__init__()
        self.classes=learndataset.attribute_data['class'][:]
        #Initialize average distance for every possible class
        for i in learndataset.attribute_data['class']:
            class_set=learndataset.select_class([i])
            values=[row[0:len(row)-1] for row in class_set.data]
            if len(values) > 1000:
                valred=np.random.permutation(values)[0:1000]
                class_distances=distance.cdist(valred,valred)
            else:
                class_distances=distance.cdist(values,values)
            average=class_distances.sum()/(len(class_distances)**2
                                           -len(class_distances))
            self.av_dist.append(average)
           
        # training the whole thing
        learndata=[row[0:len(row)-1] for row in learndataset.data]
        self.truelabels=[row[-1] for row in learndataset.data]
        self.tree=kdtree.KDTree(learndata)
            
            
        
    def evaluate(self,testdataset,knn_beta=1.5,knn_epsilon=0.99,knn_nb_neigh=3):
        """evaluate the instances and return a list of probability intervals
        
        :param testdataset: list of input features of instances to evaluate
        :type dataset: list
        :param knn_beta: value of beta parameter used in evaluation
        :type knn_beta: float
        :param knn_epsilon: value of base discounting rate to use
        :type knn_epsilon: float
        :param knn_nb_neigh: values of number f neighbours to use
        :type knn_nb_neigh: list of int
        :returns: for each value of knn_nb_neigh, a set of probability intervals
        :rtype: lists of :class:`~classifip.representations.intervalsProbability.IntervalsProbability`
        
        """
        final=[]
        self.beta=knn_beta
        self.epsilon=knn_epsilon
        answers=[]        
        for i in testdataset:

            resulting_int=np.zeros((2,len(self.classes)))
            query=self.tree.query(i,knn_nb_neigh)
            #ensure query returns list of array
            if query[0].__class__.__name__!='ndarray':
                query=list(query)
                query[0]=[query[0]]
                query[1]=[query[1]]
            for k in range(len(query[0])):
                #retrieve class index of kth neighbour
                neigh_class=self.classes.index(self.truelabels[query[1][k]])
                #compute the linear vacuous model of this neighbour
                #the higher discount, the most original info is kept
                #discount~reliability of the information between [0,1]
                expon=-((query[0][k])**(self.beta))/self.av_dist[neigh_class]
                discount=(self.epsilon)*(exp(expon))
                up=np.zeros(len(self.classes))
                up.fill(1-discount)
                up[neigh_class]=1
                down=np.zeros(len(self.classes))
                down[neigh_class]=discount
                resulting_int[0]+=up
                resulting_int[1]+=down
            # make the average of all k obtained models
            resulting_int[0]=resulting_int[0]/knn_nb_neigh
            resulting_int[1]=resulting_int[1]/knn_nb_neigh
            result=IntervalsProbability(resulting_int)
            answers.append(result)
        
        return answers
        

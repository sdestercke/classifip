from ..dataset.arff import ArffFile
from scipy.spatial import kdtree, distance
from ..representations.voting import Scores
import numpy as np
from math import exp

class IPKNNBR(object):
    """IPKNNBR implements a K-nearest neighbour method using lower previsions for
    multilabel classification
    
    If data are all precise, it returns
    :class:`~classifip.representations.voting.Scores` 
 
    :param tree: kdtree structure storing learning data set instances
    :type tree: scipy.spatial.kdtree
    :param truelabels: store the true labels of learning instances
    :type truelabels: list of labels
    :param beta: exponent parameter used in discounting rate
    :type beta: positive float
    :param epsilon: base discounting rate
    :type epsilon: float between 0 and 1
    :param av_dist: average distances of members of a given label
    :type av_dist: list of floats
    :param classes: list of label names
    :types classes: list of strings
    :param nblabels: number of labels
    :type nblabels: integer
    
    .. note::
    
        * Assumes that the labels are the last ones in samples in the learning method
        * If too many data, average distance approximated by sampling
    
    .. todo::
    
    """
    
    
    def __init__(self):
        """Build an empty IPKNNBR structure
        """
        self.tree=None
        self.truelabels=[]
        self.beta=1.5
        self.epsilon=0.99
        self.classes=[]
        self.av_dist=[]
        self.nblabels=0
        self.normal=[]
        
    def learn(self,learndata,nblabels,knnbr_normalise=True):
        """learn the KNN structure required to evaluate new instances
        
        :param learndataset: learning instances
        :type learndataset: :class:`~classifip.dataset.arff.ArffFile`
        :param nblabels: number of labels
        :type nblabels: integer
        :param knnbr_normalise: normalise input features
        :type knnbr_normalise: boolean
        """
        self.__init__()
        learndataset=learndata.make_clone()
        self.nblabels=nblabels
        self.classes=learndataset.attributes[-nblabels:]
        
        #normalisation if requested
        learndata=[row[0:len(row)-nblabels] for row in learndataset.data]
        self.truelabels=[row[-nblabels:] for row in learndataset.data]
        data_array=np.array(learndata).astype(float)
        if knnbr_normalise == True:
            self.normal.append(True)
            span=data_array.max(axis=0)-data_array.min(axis=0)
            self.normal.append(span)
            self.normal.append(data_array.min(axis=0))
            data_array=(data_array-data_array.min(axis=0))/span
            size=int(np.shape(data_array)[0])
            learndataset.data=[list(data_array[row,:])+self.truelabels[row]
                               for row in range(size)]
        else:
            self.normal.append(False)
        
        #Initialize average distance for every possible class
        for i in self.classes:
            class_set=learndataset.select_col_vals(i,['0'])
            values=[row[0:len(row)-nblabels] for row in class_set.data]
            if len(values) < 2:
                class_distances=np.array([0.1,0.1])
            elif len(values) > 1000:
                valred=np.random.permutation(values)[0:1000]
                class_distances=distance.cdist(valred,valred)
            else:
                class_distances=distance.cdist(values,values)
            averagein=class_distances.sum()/(len(class_distances)**2
                                           -len(class_distances))
            
            class_set=learndataset.select_col_vals(i,['1'])
            values=[row[0:len(row)-nblabels] for row in class_set.data]
            if len(values) < 2:
                class_distances=np.array([0.1,0.1])
            elif len(values) > 1000:
                valred=np.random.permutation(values)[0:1000]
                class_distances=distance.cdist(valred,valred)
            else:
                class_distances=distance.cdist(values,values)
            averageout=class_distances.sum()/(len(class_distances)**2
                                           -len(class_distances))
            self.av_dist.append([averageout,averagein])
           
        # training the whole thing
        learndata=[row[0:len(row)-nblabels] for row in learndataset.data]
        self.tree=kdtree.KDTree(learndata)
            
            
        
    def evaluate(self,testdataset,knnbr_beta=1.5,knnbr_epsilon=0.99,knnbr_nb_neigh=3,missing=None,MAR=True):
        """evaluate the instances and return a list of probability intervals
        
        :param testdataset: list of input features of instances to evaluate
        :type dataset: list
        :param knnbr_beta: value of beta parameter used in evaluation
        :type knnbr_beta: float
        :param knnbr_epsilon: value of base discounting rate to use
        :type knnbr_epsilon: float
        :param knnbr_nbneigh: values of number of neighbours to use
        :type knnbr_nbneigh: list of int
        :returns: for each value of knnbr_nbneigh, a set of scores for each label
        :rtype: lists of :class:`~classifip.representations.voting.Scores`
        
        """
        final=[]
        self.beta=knnbr_beta
        self.epsilon=knnbr_epsilon
        dataset=np.array(testdataset).astype(float)
        
        if self.normal[0] == True:
            dataset=(dataset-self.normal[2])/self.normal[1]

        answers=[]       
        for i in dataset:

            #scan all specified values of neighbours
            query=self.tree.query(i,k=knnbr_nb_neigh)
            #ensure query returns list of array
            if query[0].__class__.__name__!='ndarray':
                query=list(query)
                query[0]=[query[0]]
                query[1]=[query[1]]
            #scan all labels and affect a score
            resulting_score=np.zeros((self.nblabels,2))
            for j in range(self.nblabels):
                up=0.
                down=0.
                for k in range(len(query[0])):                 
                    label_in=int(self.truelabels[query[1][k]][j])
                    expon=-((query[0][k])**(self.beta))/self.av_dist[j][label_in]
                    discount=(self.epsilon)*(exp(expon))
                    randmiss=np.random.random()
                    if missing==None or randmiss>=missing:    
                        if label_in==1:
                            up+=1
                            down+=discount
                        else:
                            up+=1-discount
                    elif MAR==False: up+=1
                resulting_score[j,0]=down
                resulting_score[j,1]=up
            resulting_score=resulting_score/knnbr_nb_neigh
            result=Scores(resulting_score)
            answers.append(result)
        
        return answers
        

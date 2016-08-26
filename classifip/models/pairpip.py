from ..dataset.arff import ArffFile
from scipy.spatial import kdtree, distance
from ..representations.voting import Scores
import numpy as np
from scipy.sparse import dok_matrix
from scipy.misc import comb
from scipy.stats import norm
from math import fabs
import random

def get_binomial_int(n,y,conf):
    """return a confidence interval about the probability of binomial sample
    by performing a dichotomic search
    
    :param n: the number of samples
    :type n: integer
    :param y: the number of successes
    :type y: integer
    :param conf: the confidence degree (interval size > with conf. degree)
    :type conf: float
    :returns: an interval of the contour likelihood
    :rtype: list of two values 
    
    .. note::
    
        precision is set to 0.0001 in the dichotomic search
    """
    if y>n:
        raise Exception('Success number higher than sample number.')
    if conf>1. or conf<0.:
        raise Exception('Confidence value out of bound (unit interval)')

    #test for special cases (no samples, extreme confidence values)
    if n == 0:
        low_low=0.
        up_up=1.
        return [low_low,up_up]
    
    if conf == 1.:
        low_low=0.
        up_up=1.
        return [low_low,up_up]
    
    precision=0.0001
    prop=float(y)/float(n)

    if conf == 0.:
        low_low=prop
        up_up=prop
        return [low_low,up_up]

    low_up=prop
    low_low=0.
    up_up=1.
    up_low=prop

    max_lik=(prop**y)*((1-prop)**(n-y))
    #dichotomic search of upper and lower bounds
    while fabs(low_up - low_low) > precision:
        middle=(low_up+low_low)/2.
        if (((middle)**y)*((1-middle)**(n-y)))/max_lik <= 1-conf:
            low_low=middle
        else:
            low_up=middle
        
    while fabs(up_up - up_low) > precision:
        middle=(up_up+up_low)/2.
        if (((middle)**y)*((1-middle)**(n-y)))/max_lik <= 1-conf:
            up_up=middle
        else:
            up_low=middle
    
    return [low_low,up_up]
    

def ranking_matrices(ranking,labels):
    """return a dok_matrix of the order given by the ranking (class label)
     
    :param ranking: the ranking value (label ranking).
    :type ranking: string
    :param labels: a list of possible labels 
    :type labels: list of strings
    :returns: the preference matrix where M_ij=1 means label i preferred to j
    :rtype: :class:`~scipy.sparse.dok_matrix`
    """
    
    listed_data=ranking.split('>')
    rank=dok_matrix((len(labels),len(labels)))
    for i in range(len(listed_data)):
        ref_index=labels.index(listed_data[i])
        for j in range(i+1,len(listed_data)):
            rank[ref_index,labels.index(listed_data[j])]=1.
    return rank

class PairPIP(object):
    """algorithm using binomial likelihood to derive pairwise preference scores
    and predict label ranking
    
    :param tree: a kd-tree structure storing learning set of features
    :type tree: :class:`~scipy.spatial.kdtree`
    :param truerankings: store the observed rankings as matrices
    :type truerankings: list of dok matrices
    :param labels: a list of possible labels 
    :type labels: list of strings
    :param radius: radius of sphere including learning data for an instance
    :type radius: float
    :param normal: store information for data normalisation
        (if normalisation=True)
    :type normal: list
    
    .. note::
    
        * If too many data, average distance approximated by sampling
    """
    
    
    def __init__(self):
        """Build an empty PairPIP structure
        """
        self.tree=None
        self.truerankings=[]
        self.labels=[]
        self.radius=0.0
        self.normal=[]
        
    def learn(self,learndataset,pipp_normalise=True):
        """learn the tree structure required to perform evaluation
        
        :param learndataset: learning instances
        :type learndataset: :class:`~classifip.dataset.arff.ArffFile`
        :param pipp_normalise: normalise the input features or not
        :type pipp_normalise: boolean
        
        .. note::
    
            learndataset should come from a xarff file tailored for lable ranking
        """
        self.labels=learndataset.attribute_data['L'][:]
        learndata=[row[0:len(row)-1] for row in learndataset.data]
        data_array=np.array(learndata).astype(float)
        if pipp_normalise == True:
            span=data_array.max(axis=0)-data_array.min(axis=0)
            self.normal.append(True)
            self.normal.append(span)
            self.normal.append(data_array.min(axis=0))
            data_array=(data_array-data_array.min(axis=0))/span
        else:
            self.normal.append(False)
            
        #Initalise radius as average distance between all learning instances
        if len(data_array) > 1000:
            data_red=np.random.permutation(data_array)[0:1000]
            distances=distance.cdist(data_red,data_red)
        else:
            distances=distance.cdist(data_array,data_array)
        self.radius=distances.sum()/(2*(len(distances)**2-len(distances)))
        self.tree=kdtree.KDTree(data_array)
        self.truerankings=[ranking_matrices(row[-1],self.labels) for row
                         in learndataset.data]
            
            
        
    def evaluate(self,testdataset,pipp_radius=None,pipp_confid=[0.95]):
        """evaluate the instances and return a list of probability intervals
        with the given parameters
        
        :param pipp_radius: overcome default radius built during learning
        :type pipp_radius: float
        :param pipp_confid: set of confidence values used to predict rankings
        :type pipp_confid: list of floats
        :returns: for each value of pipp_confid, retuning voting scores
        :rtype: lists of :class:`~classifip.representations.voting.Scores`
        """
        if pipp_radius != None:
            self.radius=pipp_radius
        dataset=np.array(testdataset).astype(float)
        final=[]

        if self.normal[0] == True:
            dataset=(dataset-self.normal[2])/self.normal[1]
        
        #build matrix of majority opinions
        majority=dok_matrix((len(self.labels),len(self.labels)))
        for i in self.truerankings:
            majority=majority+i
        for k in range(len(self.labels)):
            for l in range(k)+range(k+1,len(self.labels)):
                if majority[k,l] > majority[l,k]:
                    majority[k,l]=1.
                    majority[l,k]=0.
                elif majority[k,l] < majority[l,k]:
                    majority[l,k]=1.
                    majority[k,l]=0.
                else:
                    majority[l,k]=1.
                    majority[k,l]=1.
        
        for i in dataset:
            #add every neighbours in the given radius
            result=dok_matrix((len(self.labels),len(self.labels)))
            if self.tree.query_ball_point(i,self.radius) !=[]:
                for ind in self.tree.query_ball_point(i,self.radius):
                    result=result+self.truerankings[ind]
            #if no neighbour in radius, take the closest one
            else:
                result=result+self.truerankings[self.tree.query(i)[1]]
            #compute the final scores from the sample matrix for each conf values
            answers=[]
            score_val=np.zeros((len(self.labels),2))
            for k in range(len(self.labels)):
                for l in range(k)+range(k+1,len(self.labels)):
        #if no samples for a given comparison, simply use majority
                    if result[k,l]+result[l,k] > 0.:
                        score_val[k,:]+=get_binomial_int(result[k,l]
                                    +result[l,k], result[k,l],pipp_confid)
                    else:
                        score_val[k,:]+=get_binomial_int(majority[k,l]
                                    +majority[l,k], majority[k,l],pipp_confid)                            
            answers.append(Scores(score_val))
        
        return answers
    
    def remove_pref(self,percentage,seed=None,remove_type=2):
        """remove a given percentage of (pairwise) preferences from
        the data set used in the learning phase
        
        :param percentage: percentage of missing data 
        :type percentage: float
        :param seed: random seed (to remove same preferences in repeated exp.)
        :type seed: integer
        :param remove_type: removal process (1:pairwise pref, 2:labelwise)
        :type param: integer
        """
        if percentage<0.0:
            raise Exception('Negative percentage.')
        if percentage>1.0:
            raise Exception('Percentage higher than one.')
        if seed != None:
            np.random.seed(seed)
        for i in range(len(self.truerankings)):
            if remove_type == 1:
                randmiss=np.random.rand(len(self.labels),len(self.labels))
                boolmiss=randmiss>percentage
                self.truerankings[i]=dok_matrix(self.truerankings[i].multiply(boolmiss))
            if remove_type == 2:
                for j in range(len(self.labels)):
                    if np.random.random() <= percentage:
                        self.truerankings[i][j,:]=0.0
                        self.truerankings[i][:,j]=0.0
                
        
        
        

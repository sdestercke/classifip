from ..dataset.arff import ArffFile
from ..representations.intervalsProbability import IntervalsProbability
import numpy as np
from math import exp

class NCCBR(object):
    """NCC implements the naive credal classification method using the IDM for
    multilabel classification with binary relevance.
    
    If data are all precise, it returns
    :class:`~classifip.representations.voting.Scores`. The base classifier method
    is based on [#zaffalon2002]_ and on the improvement proposed by [#corani2010]_
    
    :param feature_count: store counts of couples label/feature
    :type feature_count: dictionnary with keys label/feature
    :param label_counts: store counts of class labels (to instanciate prior)
    :type label_counts: list
    :param feature_names: store the names of features
    :type feature_names: list
    :param feature_values: store modalities of features
    :type feature_values: dictionnary associating each feature name to a list
    
    """
    
    
    def __init__(self):
        """Build an empty NCCBR structure
        """
        
        # both feature names and feature values contains the class (assumed to be the last)
        self.feature_names=[]
        self.feature_values=dict()
        self.feature_count=dict()
        self.label_count=[]
        
    def learn(self,learndataset):
        """learn the NCC, mainly storing counts of feature/class pairs
        
        :param learndataset: learning instances
        :type learndataset: :class:`~classifip.dataset.arff.ArffFile`
        """
        self.__init__()
        #Initializing the counts
        self.feature_names=learndataset.attributes[:]
        self.feature_values=learndataset.attribute_data.copy()
        for class_value in learndataset.attribute_data['class']:
            subset=[row for row in learndataset.data if row[-1]==class_value]
            self.label_count.append(len(subset))
            for feature in learndataset.attributes[:-1]:
                count_vector=[]
                feature_index=learndataset.attributes.index(feature)
                for feature_value in learndataset.attribute_data[feature]:                   
                    nb_items=[row[feature_index] for row in subset].count(feature_value)
                    count_vector.append(nb_items)
                self.feature_count[class_value+'|'+feature]=count_vector
                
            
            
        
    def evaluate(self,testdataset,ncc_epsilon=0.001,ncc_s_param=[2]):
        """evaluate the instances and return a list of probability intervals.
        
        :param testdataset: list of input features of instances to evaluate
        :type dataset: list
        :param ncc_epsilon: espilon issued from [#corani2010]_ (should be > 0)
            to avoid zero count issues
        :type ncc_espilon: float
        :param ncc_s_param: s parameters used in the IDM learning (settle
        imprecision level)
        :type ncc_s_param: list
        :returns: for each value of ncc_s_param, a set of probability intervals
        :rtype: lists of :class:`~classifip.representations.intervalsProbability.IntervalsProbability`
 
        .. note::
    
            * Probability intervals output
                as the function returns a probability interval, interval dominance
                is exact but maximality is not.
            
            * Precise prior
                prior class probabilities are assumed to be precise to speed up
                computations. The impact of the result is small, unless
                the number of class example in the training set is close to s or lower.
            
        """
        final=[]
        
        #computing class proportions
        class_ct=[float(n) for n in self.label_count]
        class_prop=[n/sum(class_ct) for n in class_ct]
        
        for item in testdataset:
            answers=[]
            for s_val in ncc_s_param:
                #initializing probability interval argument
                resulting_int=np.zeros((2,len(self.feature_values['class'])))
                #computes product of lower/upper prob for each class
                cl_index=0
                for class_val in self.feature_values['class']:
                    u_numerator=0
                    l_numerator=0
                    u_denom=class_prop[cl_index]
                    l_denom=class_prop[cl_index]
                    for feature in self.feature_names:
                        if feature != 'class':
                            f_index=self.feature_names.index(feature)
                            f_val_index=self.feature_values[feature].index(item[f_index])
                            count_string=class_val+'|'+feature
                            num_items=float(sum(self.feature_count[count_string]))
                            lower=(self.feature_count[count_string][f_val_index]/
                                   (num_items+s_val))
                            l_denom=l_denom*((1-ncc_epsilon)*lower+
                                ncc_epsilon/len(self.feature_count[count_string]))
                            upper=((self.feature_count[count_string]
                                    [f_val_index]+s_val)/(num_items+s_val))
                            u_denom=u_denom*((1-ncc_epsilon)*upper+
                                ncc_epsilon/len(self.feature_count[count_string]))
                            
                    for other_cl in set(self.feature_values['class'])-set([class_val]):
                        u_num_mult=class_prop[self.feature_values['class'].index(other_cl)]
                        l_num_mult=class_prop[self.feature_values['class'].index(other_cl)]
                        for feature in self.feature_names:
                            if feature != 'class':
                                f_index=self.feature_names.index(feature)
                                f_val_index=self.feature_values[feature].index(item[f_index])
                                count_string=other_cl+'|'+feature
                                num_items=float(sum(self.feature_count[count_string]))
                                lower=((self.feature_count[count_string]
                                        [f_val_index]+s_val)/(num_items+s_val))
                                l_num_mult=l_num_mult*((1-ncc_epsilon)*lower+
                                    ncc_epsilon/len(self.feature_count[count_string]))
                                upper=(self.feature_count[count_string]
                                       [f_val_index])/(num_items+s_val)
                                u_num_mult=u_num_mult*((1-ncc_epsilon)*upper+
                                    ncc_epsilon/len(self.feature_count[count_string]))
                        u_numerator+=u_num_mult
                        l_numerator+=l_num_mult
                        resulting_int[0,cl_index]=u_denom/(u_denom+u_numerator)
                        resulting_int[1,cl_index]=l_denom/(l_denom+l_numerator)
                    cl_index+=1
                result=IntervalsProbability(resulting_int)
                answers.append(result)
            final.append(answers)
        
        return final
        

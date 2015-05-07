from ..dataset.arff import ArffFile
from . import ncc
from ..representations.genPbox import GenPbox
from ..representations.probadis import ProbaDis
import numpy as np
from math import exp
import copy

class NCCOF(object):
    """NCCOF implements an ordinal regression method using binary
    decomposition [#frank2001]_ ideas with NCC as a base classifier. It returns a 
    :class:`~classifip.representations.genPbox.GenPbox` as a result
    
    :param setncc: store the various features of the ncc 
    :type setncc: list of :class:`~classifip.models.ncc.NCC`
    
    """
    
    
    def __init__(self):
        """Build an empty NCCBR structure
        """
        
        # both feature names and feature values contains the class (assumed to be the last)
        self.setncc=[]
        self.nblabels=0
        
    def learn(self,learndataset):
        """learn the NCC for each sets of labels (first against the rest,
        first and second against the rest, ...). Assumes the attribute to
        to predict has name class
        
        :param learndataset: learning instances
        :type learndataset: :class:`~classifip.dataset.arff.ArffFile`
        """
        self.__init__()
        classes=learndataset.attribute_data['class']
        self.nblabels=len(classes)
        for class_value in classes[0:-1]:
            #Initializing the model
            model=ncc.NCC()
            datarep=ArffFile()
	    datarep.attribute_data=learndataset.attribute_data.copy()
	    datarep.attribute_types=learndataset.attribute_types.copy()
            datarep.data = copy.deepcopy(learndataset.data)
	    datarep.relation=learndataset.relation
	    datarep.attributes=copy.copy(learndataset.attributes)
	    datarep.comment=copy.copy(learndataset.comment)
            positiveclasses=classes[0:classes.index(class_value)+1]
            negativeclasses=classes[classes.index(class_value)+1:len(classes)]
            for number,instance in enumerate(datarep.data):
                if instance[-1] in set(positiveclasses):
                    instance[-1]='plus'
                elif instance[-1] in set(negativeclasses):
                    instance[-1]='minus'
                else:
                    raise NameError("Warning: classes to replace neither in negative nor positive")
            datarep.attribute_data['class']=['plus','minus']
            model.learn(datarep)
            self.setncc.append(model)

                
            
            
        
    def evaluate(self,testdataset,ncc_epsilon=0.001,ncc_s_param=[2]):
        """evaluate the instances and return a list of generalized p-boxes.
        
        :param testdataset: list of input features of instances to evaluate
        :type testdataset: list
        :param ncc_epsilon: espilon issued from [#corani2010]_ (should be > 0)
            to avoid zero count issues
        :type ncc_espilon: float
        :param ncc_s_param: s parameters used in the IDM learning (settle
        imprecision level)
        :type ncc_s_param: list
        :returns: for each value of ncc_s_param, a generalized p-box over the labels
        :rtype: lists of :class:`~classifip.representations.genPbox.GenPbox`
 
        .. note::
    
            * Precise prior
                prior class probabilities are assumed to be precise to speed up
                computations. The impact on the result is small, unless
                the number of class example in the training set is close to s or lower.
        
        .. warning::
    
            * zero float division can happen if too many input features
            
        .. todo::
        
            * solve the zero division problem
            
        """
        final=[]
        
        for item in testdataset:
            answers=[]
            for s_val in ncc_s_param:
                if s_val!=0:
                    #initializing scores
                    resulting_pbox=np.zeros((2,self.nblabels))
                    #computes product of lower/upper prob for each class
                    for j in range(self.nblabels-1):
                        resulting_pbox[0,j]=self.setncc[j].evaluate([item],
                            ncc_s_param=[s_val])[0][0].lproba[0,0]
                        resulting_pbox[1,j]=self.setncc[j].evaluate([item],
                            ncc_s_param=[s_val])[0][0].lproba[1,0]
                    resulting_pbox[0,self.nblabels-1]=1.
                    resulting_pbox[1,self.nblabels-1]=1.
                    #repair p-boxes if they are inconsistent
                    for j in range(self.nblabels-1):
                        if resulting_pbox[0,j]>resulting_pbox[0,j+1]:
                            resulting_pbox[0,j+1]=resulting_pbox[0,j]
                        if resulting_pbox[1,(self.nblabels-1)-j]<resulting_pbox[1,(self.nblabels-1)-j-1]:
                            resulting_pbox[1,(self.nblabels-1)-j-1]=resulting_pbox[1,(self.nblabels-1)-j]
                    result=GenPbox(resulting_pbox)
                if s_val==0:
                    #initializing scores
                    resulting_cum=np.zeros(self.nblabels)
                    resulting_prob=np.zeros(self.nblabels)
                    #computes prob of each label
                    for j in range(self.nblabels-1):
                        resulting_cum[j]=self.setncc[j].evaluate([item],
                            ncc_s_param=[s_val])[0][0].proba[0]
                    resulting_cum[self.nblabels-1]=1.
                    #repair proba if they are inconsistent
                    for j in range(self.nblabels-1):
                        if resulting_cum[j]>resulting_cum[j+1]:
                            resulting_cum[j+1]=resulting_cum[j]
                    for j in range(1,self.nblabels):
                        resulting_prob[j]=resulting_cum[j]-resulting_cum[j-1]
                    resulting_prob[0]=resulting_cum[0]
                    result=ProbaDis(resulting_prob)
                answers.append(result)
            final.append(answers)
        
        return final
        

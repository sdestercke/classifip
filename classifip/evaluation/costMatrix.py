'''
Created on 7 April 2014

@author: Gen Yang
'''
import numpy as np
import math


class costMatrix:
    """
    Compute cost matrix for arff data sets: rows = predictions, columns = truths
    """


    def __init__(self,arff=None,size=None):
        """
        Constructor
        :param arff: source data file in arff format
        """
        if size is not None:
            self.size = size
            self.matrix = np.zeros([self.size,self.size])
            
        if arff is not None:
            class_values = arff.attribute_data['class']
            self.size = len(class_values)
            self.matrix = np.zeros([self.size,self.size])
            
        
    def ordinalCost(self,norm = 1):
        """
        Calculate cost matrix of an ordinal dataset based on the distance between
        class values.
        
        :param norm: norm used for the distance computation
        :return: cost matrix
        :rtype: numpy.ndarray
        """
        #=======================================================================
        # if norm == 1 :
        #     # use norm 1 to calculate cost matrix for ordinal data
        #     for i in range(0,self.size):
        #         for j in range(0,self.size):
        #             self.matrix[i,j] = abs(i-j)
        # else :
        #=======================================================================
        for i in range(0,self.size):
            for j in range(0,self.size):
                self.matrix[i,j] = math.pow(abs(i-j),norm) 
                
        return self.matrix
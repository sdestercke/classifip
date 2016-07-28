'''
Created on mai 2016

@author: Gen Yang
'''
from classifip.representations import binaryTree as bt
import pickle, copy
from scipy.stats import rv_discrete
import random

    
class NestedDichotomies(bt.BinaryTree):
    """ NestedDichotomies is a classifier that implements the nested dichotomy 
    binary decomposition technique [#Fox1997,#frank2004] allowing for the 
    transformation of a multi-classification problem into a set of binary ones. 
    The specific feature of this implementation is that, we allow for the 
    treatment of interval-valued probabilities [#yang2014]. 
     
    :param learner: the base (binary) classifier 
    
    """         
    
    def __init__(self,classifier, label=None,load=None, node=None):
        """
        Initialization of the Nested Dichotomies classifier
        :param classifier: one instance of the base binary classifier to be used.
        The classifier should have a "learn" and a "evaluate" methods available.
        The final output of the classifier should be a list of 
        :class:`~classifip.representations.intervalProbabilities.IntervalProbabilities`
        """
        if classifier is not None:
            self.classifier = classifier
        else :
            raise Exception('Must specify a binary classifier to be used')
        
        if load is not None:
            with open(load + '.pkl', 'rb') as f:
                tree = pickle.load(f)
                self.node = tree.node
                self.left = tree.left
                self.right = tree.right
                self.nbDecision = tree.nbDecision
                self.classifier = tree.classifier
        else :
            if node is None:
                if label is None:
                    raise Exception('No initialization information provided')
                self.node = NestedDichotomies.Node(label=label)
                self.nbDecision = self.node.count()
            else :    
                self.node = node
                self.nbDecision = node.count()
                 
            self.left = None
            self.right = None
            
    
    
    def learnCurrent(self,dataset,**kwargs):
        """
        Learn the underlying binary classification problem associated with the 
        current node of the dichotomy tree, and store the models in the Nested 
        Dichotomy tree.

        (See :func:`~classifip.models.nestedDichotomies.learn` for the detail of parameters)
        """
        
        if self.left.node.isEmpty() or self.right.node.isEmpty() :
            raise Exception("Current node has no left or/and right child node.")
        
        data = dataset.select_class_binary(positive=self.left.node.label, 
                                       negative=self.right.node.label)
        
        # Apply the base binary classifier for the current node of the tree     
        self.classifier.learn(data,**kwargs)

    
    def learn(self,dataset,**kwargs):
        """
        Recursive learning process (see :func:`~classifip.models.nestedDichotomies.learnCurrent`) 
        for the entire dichotomy tree structure and the entire dataset.
        
        :param dataset: learning data
        :type dataset: :class:`~classifip.dataset.ArffFile`
        
        :param **kwargs: the parameters available to the base binary classifier.
        
        .. warning:: no check is performed on the validity of the arguments.
        """
        if (self.left is not None) and (self.right is not None) :
            self.learnCurrent(dataset,**kwargs)
            '''
            we only try to learn the children nodes when there are more than one
            class value / label associated with them.
            '''
            if self.left.node.count() > 1:
                self.left.learn(dataset,**kwargs)
            if self.right.node.count() > 1:    
                self.right.learn(dataset,**kwargs)      
        
    
    def _evalCurrent(self,testdataset,out,**kwargs): 
        """
        Evaluation of the learnt local binary model on the test dataset for the 
        current node of the dichotomy tree.
        
        (See :func:`~classifip.models.nestedDichotomies.evaluate` for the detail of parameters)
        """            
        if self is not None :
            # for a single node
            #self.left.node.proba = None #we reset results of previous evaluations
            #self.right.node.proba = None

            self.node.proba = self.classifier.evaluate(testdataset, **kwargs)
            out.nbDecision = self.nbDecision
            out.node.label = self.node.label 
            out.node.proba = self.node.proba[0][0]
            
            out.left = bt.BinaryTree(self.left.node)
            out.right = bt.BinaryTree(self.right.node)
            #===================================================================
            # if type(result[0][0]) is probadis.ProbaDis:
            #     for ip in result:
            #         self.left.node.proba.append([ip[0].proba[0],ip[0].proba[0]])
            #         self.right.node.proba.append([ip[0].proba[1],ip[0].proba[1]])
            # else:
            #     for ip in result:
            #         self.left.node.proba.append([ip[0].lproba[1,0],ip[0].lproba[0,0]])
            #         self.right.node.proba.append([ip[0].lproba[1,1],ip[0].lproba[0,1]])
            #===================================================================
        
        # recursion
        if self.left.node.count() > 1:
            self.left._evalCurrent(testdataset,out.left,**kwargs)
        if self.right.node.count() > 1:
            self.right._evalCurrent(testdataset,out.right,**kwargs)
            
        #return out
    
    def evaluate(self,testdataset,**kwargs):
        """
        Recursively evaluate all local models (see :func:`~classifip.models.nestedDichotomies.evalCurrent`)
        of the entire dichotomy tree for the test dataset.
        
        :param testdataset: list of input features of instances to evaluate
        :type testdataset: list
        
        :param **kwargs: should contain any parameter available to the base 
        classifier.
        
        :returns: a set of probabilistic binary tree
        :rtype: lists of :class:`~classifip.representations.binaryTree.BinaryTree`
        
        .. warning:: no check is performed on the validity of the arguments.
        
        """
        
        if self is not None :
            results = []
            for item in testdataset:
                tree = bt.BinaryTree(label=['initialization'])
                self._evalCurrent([item],tree,**kwargs) 
                results.append(tree)
        return results
    
    
    def build(self, method="random",codes=None,shuffle=False):
        """
        Build the structure of the entire binary tree by splitting the initial
        root node (the ensemble of class values) into children nodes.
        
        :param method: Two methods are implemented:
        - "random" : Generate a single random tree structure uniformly among all 
        potential tree structures.
        - "codes" : Given a Lukasiewicz encoding, transform the codes into a 
        binary tree.
        
        :param shuffle: when using the "random method, set this paramter to True
        will suffle the ordering of the classes (useful for multiclass problem, 
        use with caution if the ordering has signification, e.g. ordinal classes).
        :type shuffle: boolean
        
        :param codes: a chain of Lukasiewicz bit codes
        :type codes: string
        """
        
        if self.node.isEmpty() :
            raise Exception("Cannot split an empty root node")
        
        if (self.left is not None) or (self.right is not None) :
            raise Exception("The given root node already has child")
        
        #Find where ends the first (and minimal) regular prefix bit codes
        def regular(codes):
            psum=0
            length=0
            while psum <> -1: #a regular prefix code is weighted to -1
                if length >= len(codes) : raise Exception('Bad encoding',codes)
                if codes[length] =='0':
                    psum += 1
                else :
                    psum -= 1
                length+=1
            
            # return the length of the first regular prefix code 
            return length
        
        #this internal function transforms recursively a Lukasiewicz code into tree
        def genTree(tree,codes,labels_node): 
            if codes <> '1':
                length_left = regular(codes[1:]) #the length of the bitcodes of the left child
                if codes[0] <> '0':
                    raise Exception('Bad tree-coding bit codes', codes)
                else:
                    # build the left child-node
                    codes_left = codes[1:1+length_left] #bitcodes of the left child
                    tree.left = NestedDichotomies(copy.copy(tree.classifier),label=labels_node[0:codes_left.count('1')])
                    
                    # build the right child-node
                    codes_right = codes[1+length_left:]
                    tree.right = NestedDichotomies(copy.copy(tree.classifier),label=labels_node[codes_left.count('1'):])
                    
                    genTree(tree.left,codes_left, tree.left.node.label)
                    genTree(tree.right,codes_right, tree.right.node.label)
                
        if method == 'codes':
            genTree(self,codes, self.node.label)
        elif method == 'random':
            n = len(self.node.label) - 1
            bitcodes = ''
            #Generate first a chain of bits of size 2n+1
            i = 0 
            j = 0
            while i+j < 2*n + 1:
                if i == n : #if n '0' are already generated, we complete by adding '1'
                    bitcodes = bitcodes + '1'
                    j+=1
                elif j == n+1: #if n '1' are already generated, we complete by adding '0'
                    bitcodes = bitcodes + '0'
                    i+=1 
                else : #generate the next bit
                    proba = float(n-i)/(2*n+1-i-j) #proba of having '0' generated
                    distrib = rv_discrete(values=((0,1),(proba,1-proba)))
                    
                    #generate a bit according to the defined proba distribution
                    newbit = rv_discrete.rvs(distrib) 
                    bitcodes = bitcodes + str(newbit)
                    
                    #increment i or j according to the generated bit
                    if newbit == 0 :
                        i+=1
                    else:
                        j+=1
                    
            
            # find the minimum partial sum knowing '0' weights 1 and '1' weights -1
            psum = 0
            mini = 1
            index_min = 0
            for k in range(0,2*n+1):
                psum += 1 if bitcodes[k] == '0' else -1
                if psum < mini :
                    mini = psum
                    index_min = k
            
            if index_min < 2*n+1:
                #form a new bitcodes starting by the element n_(index_min+1) of the old one
                new_bitcodes = bitcodes[index_min+1:] + bitcodes[0:index_min+1]
                #we do the same permutation with the labels
                #----------------- index_labels = bitcodes[0:index_min+1].count('1')
                #------- new_labels = labels[index_labels:] + labels[0:index_labels]
            else:
                #if the minimum is reached with the whole set, there is no permutation
                new_bitcodes = bitcodes
    
            #Randomize/shuffle the vector of labels if we don't want to preserve the structure of the class
            if shuffle is True:
                random.shuffle(self.node.label)
            
            genTree(self,new_bitcodes, self.node.label)
        else:
            raise Exception('Unrecognized method:',method)
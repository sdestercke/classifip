'''
Created on 11 mai 2016

@author: Gen Yang
'''
import random
import pickle
import numpy as np
from classifip.representations import credalset,intervalsProbability
from scipy.stats import rv_discrete
     


class BinaryTree(credalset.CredalSet):
    """
    Class of BinaryTree is a recursively defined (interval-valued) probabilistic 
    binary tree, following the structure :
        :param node: the root/current node 
        :type node: :class:`~classifip.BinaryTree.Node`
        
        :param left: the left child subtree
        :type left: :class:`~classifip.BinaryTree`
              
        :param right: the right child subtree
        :type right: :class:`~classifip.BinaryTree`
    
    >>> from classifip.representations import binaryTree as bt
    >>> labels = ['a', 'b', 'c']
    >>> tree = bt.BinaryTree(label=labels)
    >>> tree.printTree()
    ['a', 'b', 'c']
    
    >>> print tree.node
    ['a', 'b', 'c']
    
    >>> left,right = tree.node.splitNode(label=['a'])
    >>> print left,right
    ['a']
     ['b', 'c']
    
    >>> left_subtree = bt.BinaryTree(node=left)
    >>> right_subtree = bt.BinaryTree(node=right)
    
    >>> tree.left = left_subtree
    >>> tree.right = right_subtree
    
    >>> tree.printTree()
    ['a', 'b', 'c']
        ['a']
        ['b', 'c']
    
    >>> tree.right.build(method='random')
    >>> tree.right.printTree()
    ['b', 'c']
        ['c']
        ['b']
    
    >>> tree.printTree()
    ['a', 'b', 'c']
        ['a']
        ['b', 'c']
            ['c']
            ['b']
    
    >>> tree_bis = bt.BinaryTree(label=labels)
    >>> tree_bis.build(method='codes', codes='00111')
    >>> tree_bis.printTree()
    ['a', 'b', 'c']
        ['a', 'b']
            ['a']
            ['b']
        ['c']
    
    >>> from numpy import array
    >>> ip=array([[0.5, 0.6], [0.1, 0.1]])
    >>> from classifip.representations.intervalsProbability import IntervalsProbability
    >>> intprob=IntervalsProbability(ip)
    >>> print(intprob)
                     y0    y1 
           --------------------
    upper bound | 0.500 0.600
    lower bound | 0.100 0.100
    
    >>> tree.node.proba = intprob
    
    >>> ip2=array([[0.5, 0.5], [0.2, 0.1]])
    >>> intprob2=IntervalsProbability(ip2)
                     y0    y1 
           --------------------
    upper bound | 0.500 0.500
    lower bound | 0.200 0.100
    >>> tree.right.node.proba = intprob2
    >>> tree.printProba()
    ['a'] ['b', 'c']
    [0.100, 0.500] [0.100, 0.600]
        ['c']['b']
        [0.200, 0.500] [0.100, 0.500]
        
    >>> print tree.isreachable()
    0
    
    >>> tree.setreachableprobability()
    >>> print tree.isreachable()
    1
    
    >>> tree.printProba()
    ['a'] ['b', 'c']
    [0.400, 0.500] [0.500, 0.600]
        ['b']['c']
        [0.500, 0.500] [0.500, 0.500]
    
    >>> tree = bt.BinaryTree(label=labels)
    >>>  tree.build(method='codes', codes='01011', probas=[intprob,intprob2])
    >>> tree.printProba()
    ['a'] ['b', 'c']
    [0.400, 0.500] [0.500, 0.600]
        ['b']['c']
        [0.500, 0.500] [0.500, 0.500]
    
    >>> print tree.toIntervalsProbability()
                     y0    y1    y2 
           --------------------
    upper bound | 0.500 0.600 -0.000
    lower bound | 0.400 0.500 0.000
    
    >>> print tree.getlowerprobability(array([0,1,0]))
    0.5
    
    >>> print tree.getupperprobability(array([0,1,0]))
    0.6
    
    >>> print tree.getlowerexpectation(array([1,-1,0]))
    -0.2
    """

    
    class Node:
        """
        BinaryTree.Node represents a node of an BinaryTree. 
        
        Each Node contains an ensemble of class values (in the attribute "label") 
        which are then divided into two subsets contained in the nodes of the 
        children subtree "left" and "right". This division infers a binary 
        classification problem where we estimate the conditional probabilities
        "p(left | label)" and "p(right | label)".
        
        Attributes are :
            
        :param label: list of class values associated with the node
        :type label: list of string
        
        :param proba: conditional probabilities such as describe earlier
        :type proba: :class:`~classifip.representations.IntervalProbabilities`
        """
   
        def __init__(self, label=None,proba=None):
            """
            Constructor : create a node and associate class values to it.
            """
            self.label = label
            self.proba = proba
            #self.learner = ncc.NCC()
            
            
        def __str__(self):
            """
            Method for printing a node by its label
            """
            s = str(self.label) + "\n"
                    
            return s
            
            
        def isEmpty(self):
            """
            Testing if the node is "empty"
            :rtype: boolean
            """
            return (self.label is None)
        
        
        def count(self):
            """
            Count the number of labels/class values. 
            :rtype: integer
            """
            return len(self.label)    
                
                
        def splitNode(self, method = "random", label = None):
            """
            Split a node into two sub-nodes using the label
            :param method: different method for splitting a set of class values
            :type method: string
            
            :param label: if this argument is not None, then its content are used to
                        form the 'left child' of the current node, and the rest of 
                        current node's class values form the 'right child node'.
            :type label: list of string (must be class values)
            
            :returns: two children nodes resulted from the split
            :rtype: :class:`~classifip.BinaryTree.Node`, :class:`~classifip.BinaryTree.Node`
            """
            if self.isEmpty() :
                raise Exception("Cannot split an empty node")
    
            if label is not None :
                '''
                For manual splitting only 
                '''
                left = BinaryTree.Node(label)
                right = BinaryTree.Node([l for l in self.label if l not in label]) 
            
            elif method == "random" :
                '''
                Split randomly class values in two balanced subsets
                '''
                left = BinaryTree.Node(random.sample(self.label, len(self.label)/2))
                right = BinaryTree.Node([l for l in self.label if l not in left.label])  
            else :
                raise Exception("Unrecognized splitting method")    
                
            return left,right
        

    def __init__(self, node=None, label=None,load=None):
        """
        Constructor : initialize the root node and its children tree. Three 
        initialization methods are available:
        
        :param node: initialization by adding a new node as current node
        :type node: :class:`~classifip.BinaryTree.Node`
        
        :param label: initialization by specifying the label of the new node
        :type label: list of string
        
        :param load:  initialization by specifying the path of a saved BinaryTree
        structure file (".pkl")
        :type load: string
            
        """
        if load is not None:
            with open(load + '.pkl', 'rb') as f:
                tree = pickle.load(f)
                self.node = tree.node
                self.left = tree.left
                self.right = tree.right
                self.nbDecision = tree.nbDecision
        else :
            if node is None:
                if label is None:
                    raise Exception('No label information provided')
                self.node = BinaryTree.Node(label=label)
                self.nbDecision = self.node.count()
            else :    
                self.node = node
                self.nbDecision = node.count()
                 
            self.left = None
            self.right = None
        
    

         
    def build(self, method="random",codes=None,shuffle=False,probas=None):
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
        
        :param probas: a list of probabilities to be attributed to each node
        :type probas: list of :class:`~classifip.representations.IntervalProbabilities`
        """
        
        if self.node.isEmpty() :
            raise Exception("Cannot split an empty root node")
        
        if (self.left is not None) or (self.right is not None) :
            raise Exception("The given root node already has child")
        
        if probas is not None:
            if len(probas) <> codes.count('0'):
                raise Exception('Wrong number of probas provided, needed ', codes.count('0'))
        
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
        def genTree(tree,codes,proba_nodes=None): 
            if codes <> '1':
                length_left = regular(codes[1:]) #the length of the bitcodes of the left child
                if codes[0] <> '0':
                    raise Exception('Bad tree-coding bit codes', codes)
                elif proba_nodes is None:
                    # build the left child-node
                    codes_left = codes[1:1+length_left] #bitcodes of the left child
                    tree.left = BinaryTree(label=tree.node.label[0:codes_left.count('1')])               
                    
                    # build the right child-node
                    codes_right = codes[1+length_left:]
                    tree.right = BinaryTree(label=tree.node.label[codes_left.count('1'):])
                    
                    genTree(tree.left,codes_left, tree.left.node.label)
                    genTree(tree.right,codes_right, tree.right.node.label)
                else:
                    # attribute the proba to the current node
                    tree.node.proba = proba_nodes[0]
                    
                    # build the left child-node
                    codes_left = codes[1:1+length_left] #bitcodes of the left child
                    count_probas_left = codes_left.count('0')
                    tree.left = BinaryTree(label=tree.node.label[0:codes_left.count('1')]) 
                    
                    # build the right child-node
                    codes_right = codes[1+length_left:]
                    count_probas_right = codes_right.count('0')
                    tree.right = BinaryTree(label=tree.node.label[codes_left.count('1'):])
                    
                    
                    if count_probas_left == 0 :
                        genTree(tree.left,codes_left)
                    else:
                        genTree(tree.left,codes_left, 
                                proba_nodes=proba_nodes[1:count_probas_left+1])   
                    
                    if count_probas_right == 0:
                        genTree(tree.right,codes_right)
                    else:
                        genTree(tree.right,codes_right,
                            proba_nodes=proba_nodes[1+count_probas_left:])
                    
                
        if method == 'codes':
            if probas is None:
                genTree(self,codes=codes)
            else:
                genTree(self,codes=codes, proba_nodes=probas)
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
            
            genTree(self,new_bitcodes)
        else:
            raise Exception('Unrecognized method:',method)
        
            
    
    def save(self,name):
        """
        Save the tree structure to the current location on the disk
        :param name: the file name given to the saved tree structure
        :type name: string
        """
        with open('..\\datasets\\' + name + '.pkl', 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

    
    def getlowerexpectation(self,function):
        """Compute the lower expectation of a given function
        
        :param function: values of the function
        :type function: :class:`~numpy.ndarray`
        
        :return: the lower expectation value
        :rtype: float
         
        """
        if self.isreachable()==0:
            self.setreachableprobability()
        
        class_values = self.node.label            
        nb_class = len(class_values)
        
        if function.shape <> (nb_class,):
            raise Exception('Size of cost vector is not correct:',function.shape)
        cost = function
        
        
        def lowerExp(NDtree):
            '''
            Internal recursive function for expectation computation, using the 
            expected cost of the children nodes as cost for the parent binary
            classification problem.            
            '''
                
            if NDtree.left.node.count() == 1 and NDtree.left.node.count() == 1 :
                expInf = NDtree.node.proba.getlowerexpectation(function=np.array(
                            [cost[class_values.index(NDtree.left.node.label[0])], 
                            cost[class_values.index(NDtree.right.node.label[0])]])) 
                          
            elif NDtree.left.node.count() == 1:
                expInf = NDtree.node.proba.getlowerexpectation(function=np.array(
                            [cost[class_values.index(NDtree.left.node.label[0])],
                             lowerExp(NDtree.right)])) 
            elif NDtree.right.node.count() == 1:
                expInf = NDtree.node.proba.getlowerexpectation(function=np.array(
                            [lowerExp(NDtree.left), 
                             cost[class_values.index(NDtree.left.node.label[0])]])) 
                          
            else:
                expInf = NDtree.node.proba.getlowerexpectation(function=np.array(
                            [lowerExp(NDtree.left), 
                             lowerExp(NDtree.right)])) 
            return expInf
            
        return lowerExp(self)
    
    def getlowerprobability(self, subset):
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
        if not credalset.CredalSet.issubsetmask(subset):
            raise Exception('Array is not 1/0 elements')
        if self.isreachable()==0:
            self.setreachableprobability()
            
        return self.getlowerexpectation(function=subset)
    
    def getupperprobability(self,subset):
        """Compute upper probability of an event expressed in binary code. 
        
        :param subset: the event of interest (a 1xn vector containing 1 for elements
            in the event, 0 otherwise.)
        :param type: np.array
        :returns: upper probability value
        :rtype: float
        
        """    
        if subset.__class__.__name__ != 'ndarray':
            raise Exception('Expecting a numpy array as argument')
        if subset.size != self.nbDecision:
            raise Exception('Subset incompatible with the frame size')
        if not credalset.CredalSet.issubsetmask(subset):
            raise Exception('Array is not 1/0 elements')
        if self.isreachable()==0:
            self.setreachableprobability()
            
        return self.getupperexpectation(obj=subset)
    
    def toIntervalsProbability(self):
        """
        Transform the binary tree representation into the one of probability 
        intervals.
        
        :returns: posterior probabilities of each class
        :rtype: :class:`~classifip.representations.IntervalProbabilities`
        
        .. warning: the order of the classes in the output representation is the 
        one in the attribute 'label' of the root node of the binary tree.
        """
        
        probas = np.zeros((2,self.nbDecision))

        for ind, val in enumerate(self.node.label):
            subset = np.zeros((self.nbDecision,))
            subset[ind] = 1
            probas[0,ind] = self.getupperprobability(subset)
            probas[1,ind] = self.getlowerprobability(subset)
        
        return intervalsProbability.IntervalsProbability(probas)
    
    def isreachable(self):
        """Recursively check if the probability intervals are reachable (are coherent)
        for each node of the binary tree.
        
        :returns: 0 (not coherent/tight) or 1 (tight/coherent).
        :rtype: integer
        
        """    
        if self.left.node.count() == 1 and self.right.node.count() == 1: 
            return self.node.proba.isreachable()
        
        if self.left.node.count() == 1:
            return self.node.proba.isreachable() * self.right.node.proba.isreachable()
        
        if self.right.node.count() == 1:
            return self.node.proba.isreachable() * self.left.node.proba.isreachable()
        
        return self.left.node.proba.isreachable() * self.right.node.proba.isreachable()
    
    
    def isproper(self):
        """Recursively check if probability intervals induce a non-empty probability set. 
        
        :returns: 0 (empty/incur sure loss) or 1 (non-empty/avoid sure loss).
        :rtype: integer
        
        """    
        
        if self.left.node.count() == 1 and self.right.node.count() == 1: 
            return self.node.proba.isproper()
        
        if self.left.node.count() == 1:
            return self.node.proba.isproper() * self.right.node.proba.isproper()
        
        if self.right.node.count() == 1:
            return self.node.proba.isproper() * self.left.node.proba.isproper()
        
        return self.left.node.proba.isproper() * self.right.node.proba.isproper()
    
    def setreachableprobability(self):
        """Make the bounds of every node reachable.
        
        """  
        if self.isproper() == 1:
            if self.node.proba.isreachable() == 0:
                self.node.proba.setreachableprobability()
                
            if self.right.node.count() > 1: 
                self.right.setreachableprobability()
            
            if self.left.node.count() > 1:
                self.left.setreachableprobability()
        else:
            raise Exception('intervals inducing empty set: operation not possible')
        
    
    def printTree(self, _p = 0):
        '''
        Method for printing a binary tree
        '''
        
        if _p == 0 :
            print self.node.label
            
        _p += 1
        
        if self.left is not None:
            print "    " * _p + str(self.left.node.label)
            self.left.printTree(_p) 
        
        if self.right is not None:
            print "    " * _p + str(self.right.node.label)
            self.right.printTree(_p) 
    
    def printProba(self):
        '''
        Method for printing each node's probability intervals.
        '''
        
        def printP(tree, _p = 0):
            str_buf1 = "[%.3f, %.3f]" % (tree.node.proba.lproba[1,0],tree.node.proba.lproba[0,0])
            str_buf2 = "[%.3f, %.3f]" % (tree.node.proba.lproba[1,1],tree.node.proba.lproba[0,1])
            if _p == 0 :
                print tree.left.node.label, tree.right.node.label
                print str_buf1, str_buf2
            else : 
                print "    " * _p + str(tree.left.node.label) + str(tree.right.node.label)
                print "    " * _p + str_buf1, str_buf2
            _p += 1
            
            if tree.left.node.proba is not None:
                printP(tree.left,_p) 
            
            if tree.right.node.proba is not None:
                printP(tree.right,_p) 
    
        return printP(self)
            
    
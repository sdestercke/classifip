# -*- coding: utf-8 -*-
# Copyright (c) 2008, Mikio L. Braun, Cheng Soon Ong, Soeren Sonnenburg
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
#
#     * Redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above
# copyright notice, this list of conditions and the following disclaimer
# in the documentation and/or other materials provided with the
# distribution.
#     * Neither the names of the Technical University of Berlin, ETH
# Zürich, or Fraunhofer FIRST nor the names of its contributors may be
# used to endorse or promote products derived from this software without
# specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


import re

class ArffFile(object):
    """class to read and write arff data structures
    
    An ARFF File object describes a data set consisting of a number
    of data points made up of attributes. The whole data set is called
    a 'relation'. Supported attributes are:

    - 'numeric': floating point numbers
    - 'string': strings
    - 'nominal': taking one of a number of possible values
    - 'ranking': taking a ranking of labels (using the xarff format of WEKA-LR)

    Not all features of ARFF files are supported yet. The most notable
    exceptions are:

    - no sparse data
    - no support for date and relational attributes

    Also, parsing of strings might still be a bit brittle.

    You can either load or save from files, or write and parse from a
    string.

    You can also construct an empty ARFF file and then fill in your
    data by hand. To define attributes use the define_attribute method.

    Attributes are:

    :param relation: name of the relation
    :type relation: string
    :param attributes: names of attributes
    :type attributes: list of strings
    :param attribute_types: types of attributes (dictionnary keys: names of att.)
    :type attribute_types: dictionnary
    :param attribute_data: data of attributes (modalities for non-numeric)
    :type attribute_data: None or list of strings
    :param comment: the initial comment in the file. Typically contains some
        information on the data set.
    :type comment: string
    :param data: the actual data of the file
    :type data: list of items
    
    .. todo::
    
        * transform dump into __str__ method
    """
    def __init__(self):
        """Construct an empty ARFF structure."""
        self.relation = ''
        self.attributes = []
        self.attribute_types = dict()
        self.attribute_data = dict()
        self.comment = []
        self.data = []
        pass

    def load(self, filename):
        """Load an ARFF File from a file.
        
        :param filename: the name of the file containing data
        :type filename: string
        """
        
        #reinitialize the object
        self.__init__()
        #fill in the object
        o = open(filename)
        s = o.read()
        a = ArffFile.parse(s)
        self.relation = a.relation
        self.attributes = a.attributes
        self.attribute_types = a.attribute_types
        self.attribute_data = a.attribute_data
        self.comment = a.comment
        self.data = a.data
        o.close()

    def select_class(self, select):
        """return an ARFF object where only some classes are selected
        
        :param select: the names of the classes to retain
        :type select: list
        :return: a new ArffFile structure containing only selected classes
        :rtype: :class:`~classifip.dataset.arff.ArffFile`
        
        .. warning::
        
            should not be used with files containing ranks
        """
        if 'class' not in self.attribute_data.keys():
            raise NameError("Cannot find a class attribute.")
        if set(select) - set(self.attribute_data['class'])!=set([]):
            raise NameError("Specified classes not a subset of existing ones!")
        selection=ArffFile()
        #construct list with sets of indices matching class names
        #assume the class is the last provided item
        indices=[i for i,val in enumerate(self.data) if val[-1] in select]
        #data corresponding to provided class
        selected_data=[self.data[i] for i in indices]
        
        selection.attribute_data=self.attribute_data.copy()
        selection.attribute_types=self.attribute_types.copy()
        selection.attribute_data['class']=select
        selection.data = selected_data
        selection.relation=self.relation
        selection.attributes=self.attributes[:]
        selection.comment=self.comment[:]

        return selection
    
    
    def select_class_bi(self, positive, negative):
        """return an ARFF object where only some classes are selected in order
        to form a dataset for a binary classification problem.
        
        :param select: the names of the classes to retain
        :type select: list
                
        :param positive: classes values to be considered as 'positive' in the binary
        classfication problem
        :type positive:list
        
        :param positive: classes values to be considered as 'positive' in the binary
        classfication problem
        :type positive:list
        
        :returns: the dictionary ('positive'/'negative') of selected data's 
        indices and a new ArffFile structure containing only selected classes
        :rtype: list of strings, :class:`~classifip.dataset.arff.ArffFile`
        
        .. warning::
        
            should not be used with files containing ranks
        """
        if 'class' not in self.attribute_data.keys():
            raise NameError("Cannot find a class attribute.")
        if set(positive) - set(self.attribute_data['class'])!=set([]):
            raise NameError("Specified 'positive' classes not a subset of existing ones!")
        
        if set(negative) - set(self.attribute_data['class'])!=set([]):
            raise NameError("Specified 'negative' classes not a subset of existing ones!")
        
        selection=ArffFile()
        #initiate the variable for the output data
        selected_data=[] 
        
        
        #construct list with sets of indices matching class names
        #assume the class is the last provided item
        #and form data corresponding to provided class
        for i,val in enumerate(self.data) :
            if val[-1] in positive :
                buff = self.data[i][:]
                buff[-1] = 'positive'
                selected_data.append(buff)

            elif val[-1] in negative:
                buff = self.data[i][:]
                buff[-1] = 'negative'
                selected_data.append(buff)


        
        selection.attribute_data=self.attribute_data.copy()
        selection.attribute_types=self.attribute_types.copy()
        selection.attribute_data['class']=['positive','negative']
        selection.data = selected_data
        selection.relation=self.relation
        selection.attributes=self.attributes[:]
        selection.comment=self.comment[:]

        return selection
    
    
    def occurrences(self):
        """
        Computing the occurrences of each class values
        
        :return: dictionary of occurrences for class values
        :rtype: dict
        """
        occurrences = {}
        for item in self.data:
            class_value = item[-1]
            if occurrences.has_key(class_value): 
                occurrences[class_value] += 1
            else :
                occurrences[class_value] = 1
        return occurrences
    
    def select_col_vals(self,column,select):
        """return an ARFF File where only some rows are selected in data
        
        :param select: the values to retain
        :type select: list
        :param column: name of the attribute
        :type column: string
        :return: a new ArffFile structure containing only selected values in the column
        :rtype: :class:`~classifip.dataset.arff.ArffFile`
        """
        
        if column not in self.attribute_data.keys():
            raise NameError("Cannot find specified column.")
        selection=ArffFile()
        col_ind=self.attributes.index(column)
        #construct list with sets of indices matching infos
        indices=[i for i,val in enumerate(self.data) if val[col_ind] in select]
        
        selected_data=[self.data[i] for i in indices]
        
        selection.attribute_data=self.attribute_data.copy()
        selection.attribute_types=self.attribute_types.copy()
        if self.attribute_types[column]=='nominal':
            selection.attribute_data[column]=select
        selection.data = selected_data
        selection.relation=self.relation
        selection.attributes=self.attributes[:]
        selection.comment=self.comment[:]
        
        return selection 
    
    def remove_col(self,column):
        """return an ARFF File where the the column specified is removed
        
        :param column: name of the attribute
        :type column: string
        :return: a new ArffFile structure excluding the specified column
        :rtype: :class:`~classifip.dataset.arff.ArffFile`
        """
        if column not in self.attribute_data.keys():
            raise NameError("Cannot find specified column.")
        
        selection=self
        col_ind=self.attributes.index(column)
        del selection.attributes[col_ind]
        del selection.attribute_types[column]
        del selection.attribute_data[column]
        selection.data = [row[0:col_ind] + row[(col_ind+1):] for row in self.data]
        
        return selection
        

    def make_clone(self):
        """Make a copy of the current object
        
        :return: a copy
        :rtype: :class:`~classifip.dataset.arff.ArffFile`        
        """
        cloned=ArffFile()
        
        cloned.attribute_data=self.attribute_data.copy()
        cloned.attribute_types=self.attribute_types.copy()
        cloned.data = self.data[:]
        cloned.relation=self.relation
        cloned.attributes=self.attributes[:]
        cloned.comment=self.comment[:]
        
        return cloned

    @staticmethod
    def parse(s):
        """Parse an ARFF File already loaded into a string."""
        a = ArffFile()
        a.state = 'comment'
        a.lineno = 1
        for l in s.splitlines():
            a.__parseline(l)
            a.lineno += 1
        return a

    def save(self, filename):
        """Save an arff structure to a file.
        
        :param filename: the name of the file where data are saved
        :type filename: string        
        
        """
        o = open(filename, 'w')
        o.write(self.write())
        o.close()

    def write(self):
        """Write an arff structure to a string."""
        o = []
        o.append('% ' + re.sub("\n", "\n% ", self.comment))
        o.append("@relation " + self.esc(self.relation))
        for a in self.attributes:
            at = self.attribute_types[a]
            if at == 'numeric':
                o.append("@attribute " + self.esc(a) + " numeric")
            elif at == 'string':
                o.append("@attribute " + self.esc(a) + " string")
            elif at == 'nominal':
                o.append("@attribute " + self.esc(a) +
                         " {" + ','.join(self.attribute_data[a]) + "}")
            elif at == 'ranking':
                o.append("@attribute" + self.esc(a) + " ranking" +
                         " {" + ','.join(self.attribute_data[a]) + "}")
            else:
                raise NameError("Type " + at + " not supported for writing!")
        o.append("\n@data")
        for d in self.data:
            line = []
            for e, a in zip(d, self.attributes):
                at = self.attribute_types[a]
                if at == 'numeric':
                    line.append(str(e))
                elif at == 'string':
                    line.append(self.esc(e))
                elif at == 'nominal':
                    line.append(e)
                elif at == 'ranking':
                    line.append(e)
                else:
                    raise "Type " + at + " not supported for writing!"
            o.append(','.join(line))
        return "\n".join(o) + "\n"

    def esc(self, s):
        "Escape a string if it contains spaces"
        if re.match(r'\s', s):
            return "\'" + s + "\'"
        else:
            return s

    def define_attribute(self, name, atype, data=None):
        """Define a new attribute.
        
        For nominal and ranking attributes, pass the possible values as data.        
        
        :param atype: 'numeric', 'string', 'ranking' and 'nominal'.
        :type atype: string
        :param name: name of the attribute
        :type name: string
        :param data: modalities/labels of the attribute
        :type atype: list
        """
        
        self.attributes.append(name)
        self.attribute_types[name] = atype
        self.attribute_data[name] = data

    def __parseline(self, l):
        if self.state == 'comment':
            if len(l) > 0 and l[0] == '%':
                self.comment.append(l[2:])
            else:
                self.comment = '\n'.join(self.comment)
                self.state = 'in_header'
                self.__parseline(l)
        elif self.state == 'in_header':
            ll = l.lower()
            if ll.startswith('@relation '):
                self.__parse_relation(l)
            if ll.startswith('@attribute '):
                self.__parse_attribute(l)
            if ll.startswith('@data'):
                self.state = 'data'
        elif self.state == 'data':
            if len(l) > 0 and l[0] != '%':
                self.__parse_data(l)

    def __parse_relation(self, l):
        l = l.split()
        self.relation = l[1]

    def __parse_attribute(self, l):
        p = re.compile(r'[a-zA-Z_-][a-zA-Z0-9_-]*|\{[^\}]+\}|\'[^\']+\'|\"[^\"]+\"')
        l = [s.strip() for s in p.findall(l)]
        name = l[1].replace("'","") # Modification : all ' in typename are removed
        atype = l[2]
        atypel = atype.lower()
        if (atypel == 'real' or
            atypel == 'numeric' or
            atypel == 'integer'):
            self.define_attribute(name, 'numeric')
        elif atypel == 'string':
            self.define_attribute(name, 'string')
        elif atypel == 'ranking':
            labelrow=l[3]
            labels = [s.strip () for s in labelrow[1:-1].split(',')]
            self.define_attribute(name, 'ranking', labels)
        elif atype[0] == '{' and atype[-1] == '}':
            values = [s.strip () for s in atype[1:-1].split(',')]
            self.define_attribute(name, 'nominal', values)
        else:
            self.__print_warning("unsupported type " + atype + " for attribute " + name + ".")


    def __parse_data(self, l):
        l = [s.strip() for s in l.split(',')]
        if len(l) != len(self.attributes):
            self.__print_warning("contains wrong number of values")
            return 

        datum = []
        for n, v in zip(self.attributes, l):
            at = self.attribute_types[n]
            if at == 'numeric':
                if re.match(r'[+-]?[0-9]+(?:\.[0-9]*(?:[eE]-?[0-9]+)?)?', v):
                    datum.append(float(v))
                elif v == '?':
                    datum.append(float('nan'))
                else:
                    self.__print_warning('non-numeric value %s for numeric attribute %s' % (v, n))
                    return
            elif at == 'string':
                datum.append(v)
            elif at == 'ranking':
                for k in v.split('>'):
                    if k not in self.attribute_data[n]:
                        self.__print_warning('incorrect label %s for ranking attribute %s' % (v, n))
                datum.append(v)
            elif at == 'nominal':
                if v in self.attribute_data[n]:
                    datum.append(v)
                elif v == '?':
                    datum.append(None)                    
                else:
                    self.__print_warning('incorrect value %s for nominal attribute %s' % (v, n))
                    return
        self.data.append(datum)

    def __print_warning(self, msg):
        print ('Warning (line %d): ' % self.lineno) + msg

    def dump(self):
        """Print an overview of the ARFF file."""
        print "Relation " + self.relation
        print "  With attributes"
        for n in self.attributes:
            if self.attribute_types[n] != 'nominal':
                print "    %s of type %s" % (n, self.attribute_types[n])
            else:
                print ("    " + n + " of type nominal with values " +
                       ', '.join(self.attribute_data[n]))
        for d in self.data:
            print d
    


if __name__ == '__main__':
    if False:
        a = ArffFile.parse("""% yes
% this is great
@relation foobar
@attribute foo {a,b,c}
@attribute bar real
@data
a, 1
b, 2
c, d
d, 3
""")
        a.dump()

    a = ArffFile()
    a.load('../../diabetes.arff')
    
    print a.write()
    

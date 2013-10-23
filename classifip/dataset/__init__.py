from . import arff
from Orange.data import Table as OTable
from Orange.feature.discretization import Entropy as OEnt
from Orange.feature.discretization import EqualFreq as OEFreq
from Orange.feature.discretization import EqualWidth as OEWidth
from Orange.data.discretization import DiscretizeTable as DiscTable

def discretize(infilename,outfilename,discmet,numint=4,selfeat=None):
    """
    Discretize features of data sets according to specified method. Necessitate
    Orange Python module to perform the discretization. 
    
    :param infilename: name of the input file (expecting an arff file)
    :type infilename: string
    :param outfilename: name of the output file
    :type outfilename: string
    :param discmet: discretization method
    :type discmet: function
    :param numint: number of intervals
    :type numint: integer
    :param feature: specify a feature name if not all must be discretized
    :type feature: string
    """
    
    data = OTable(infilename)
    
    if selfeat==None:
        if discmet==OEnt:
            data_ent = DiscTable(data,method=discmet())
        else:
            data_ent = DiscTable(data,method=discmet(numberOfIntervals=numint))
    else:
        if discmet==OEnt:
            data_ent = DiscTable(data,method=discmet(),
                                 feature=data.domain[data.domain.index(selfeat)])
        else:
            data_ent = DiscTable(data,method=discmet(numberOfIntervals=numint),
                                 feature=data.domain[data.domain.index(selfeat)])
    # Manipulation of the discretized data
    for attr in data_ent.domain.attributes :
        #Reset renamed attributes name to original ones
        if (attr.name[0:2] == "D_"):
            attr.name = attr.name[2:]
        #Replace ',' occurring in interval-valued data instances by ';' 
        attr.values = [val.replace(',',";") for val in attr.values]
    
    # save the discretized data
    data_ent.save(outfilename)
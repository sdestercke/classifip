__author__ = 'salmuz'

import abc


class Imprecise(metaclass=abc.ABCMeta):

    @abc.abstractclassmethod
    def inference(self, query, method):
        pass

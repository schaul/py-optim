from scipy import zeros
from sgd import SGD


class AveragingSGD(SGD):
    """ Contains an arithmetic average of recently seen parameter vectors """    

    def _additionalInit(self):
        self._avg_params = zeros(self.paramdim)

    def _computeStatistics(self):
        self._avg_params *= (1-1/(1.+self._num_updates))
        self._avg_params += 1. / (1+self._num_updates) * self.parameters

    @property
    def bestParameters(self):
        return self._avg_params

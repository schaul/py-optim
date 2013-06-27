from scipy import zeros
from sgd import SGD


class AveragingSGD(SGD):
    """ Contains an arithmetic average of recently seen parameter vectors """    
    
    fixedDecay = None

    def _additionalInit(self):
        self._avg_params = zeros(self.paramdim)
        
    @property    
    def _decayProportion(self):
        if self.fixedDecay is None:
            return 1. / (1+self._num_updates)
        elif self._num_updates == 0:
            # even with a fixed decay, the first is a full update
            return 1
        else:
            return self.fixedDecay

    def _computeStatistics(self):
        d = self._decayProportion
        self._avg_params *= (1-d)
        self._avg_params += d * self.parameters

    @property
    def bestParameters(self):
        if self._num_updates > 0:
            return self._avg_params
        else:
            return self.parameters

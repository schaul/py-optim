from sgd import SGD
from benchmarks.stoch_1d import StochQuad

class OracleSGD(SGD):
    """ An algorithm that cheats, because it
    always knows the optimum learning rate (assuming a quadratic loss function) """
    
    def _additionalInit(self):
        if not isinstance(self.provider.stochfun, StochQuad):
            print 'WARNING: orcale inapplicable'
        self._noiseLevel = self.provider.stochfun.noiseLevel
        self._curvature = self.provider.stochfun.curvature
    
    @property
    def learning_rate(self):
        return self.parameters ** 2 / (self.parameters ** 2 + self._noiseLevel ** 2 
                                       / self.batch_size) / self._curvature          
    

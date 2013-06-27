from sgd import SGD
from benchmarks.stoch_1d import StochQuad
from averaging import AveragingSGD
from scipy import clip, sqrt

class OracleSGD(SGD):
    """ An algorithm that cheats, because it
    always knows the optimum learning rate (assuming a quadratic loss function) """
    
    def _additionalInit(self):
        if not isinstance(self.provider.stochfun, StochQuad):
            print 'WARNING: oracle inapplicable'
        self._noiseLevel = self.provider.stochfun.noiseLevel
        self._curvature = self.provider.stochfun.curvature
    
    @property
    def learning_rate(self):
        return self.parameters ** 2 / (self.parameters ** 2 + self._noiseLevel ** 2 
                                       / self.batch_size) / self._curvature          



class AveragingOracle(OracleSGD, AveragingSGD):
    """ The oracle learning rates are boosted in inverse proportion to the averaging decay, but with a maximum of 1/h. 
    This appears to work very well. """
    
    def _additionalInit(self):
        OracleSGD._additionalInit(self)
        AveragingSGD._additionalInit(self)
        
    @property
    def learning_rate(self):
        return clip(1./(self._decayProportion) * OracleSGD.learning_rate.__get__(self), 0, 1./self._curvature)    


class AdaptivelyAveragingOracle(AveragingOracle):
    """ This formula comes from a magical derivation... """

    @property
    def learning_rate(self):
        #self._num_updates += 1
        d = sqrt(self._decayProportion)
        #self._num_updates -= 1
        #print d, self.parameters[0], self._avg_params[0],
        #print  ((self.parameters ** 2 + abs((1-d) * self.parameters * (self.parameters -self._avg_params))) / (self.parameters ** 2 + self._noiseLevel ** 2 
        #                               / self.batch_size) / d / self._curvature)[0]
        return clip((self.parameters ** 2 +  abs((1-d**2) * self.parameters * (self.parameters -self._avg_params))) / (self.parameters ** 2 + self._noiseLevel ** 2 
                                       / self.batch_size) / d / self._curvature, 0, 1./self._curvature)
        
    
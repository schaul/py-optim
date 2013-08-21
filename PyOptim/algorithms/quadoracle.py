from sgd import SGD
from benchmarks.stoch_1d import StochQuad
from averaging import AveragingSGD
from scipy import clip

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



class _AveragingOracle(OracleSGD, AveragingSGD):
    """ The oracle learning rates are boosted in inverse proportion to the averaging decay, but with a maximum of 1/h. 
    This appears to work very well. """
    
    def _additionalInit(self):
        OracleSGD._additionalInit(self)
        AveragingSGD._additionalInit(self)
        
    @property
    def learning_rate(self):
        return clip(1./(self._decayProportion) * OracleSGD.learning_rate.__get__(self), 0, 1./self._curvature)    



class AveragingOracle(OracleSGD, AveragingSGD):
    """ This formula comes from a magical derivation:
    Not only does the oracle need to know the distance to the optimum for the current parameters, it also needs that info
    for the current averaged parameters. """
    def _additionalInit(self):
        OracleSGD._additionalInit(self)
        AveragingSGD._additionalInit(self)
        
    @property
    def learning_rate(self):        
        return self._calcOptimalRate(self._decayProportion)
        
    def _calcOptimalRate(self, decay):
        return (self.parameters **2 
                - (1-decay) * self.parameters *(self.parameters - self._avg_params))\
                / (self.parameters **2 +  self._noiseLevel **2) / decay / self._curvature



class AdaptivelyAveragingOracle(AveragingOracle):
    """ Averaging rate is adaptive, as for vSGD (but with another oracle step). """
    
    def _additionalInit(self):
        AveragingOracle._additionalInit(self)
        self._tau = 1
            
    @property    
    def _decayProportion(self):
        lr = self._calcOptimalRate(1./self._tau)
        self._tau *= (1-lr)
        self._tau += 1
        return 1./self._tau
        
    
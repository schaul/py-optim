"""
Wrapper functions to make simple stationary functions into non-stationary ones.
"""

from stoch_1d import StochFun
from scipy import randn
from pybrain.utilities import setAllArgs



    
class OptimumJumper(StochFun):
    """ Every so many time-steps, the optimum jumps in a random direction, by a certain amount """
    
    jumpdist = 1
    jumpdist_std = .1
    jumptime = 200
    # no jump initially?
    startatzero = True
    
    def __init__(self, basefun, **kwargs):
        """ Provide a stationary 1D base function. """
        setAllArgs(self, kwargs)
        self._basefun = basefun
        self._optimum = 0
        if not self.startatzero:
            self._jump()
        else:
            self._timer = self.jumptime
            
        # monkey-patching wrapper
        tmp = self._basefun._newSample
        def _oneStep(*args, **kwargs):
            """ the timer is decremented only when _newSamples() is called """
            #print self, self._timer, self._optimum
            if 'override' not in kwargs or kwargs['override'] == False:
                self._timer -= 1
                if self._timer <= 0:
                    self._jump()
            return tmp(*args, **kwargs)                    
        self._basefun._newSample = _oneStep
    
    def _newSample(self, *args, **kwargs): return self._basefun._newSample(*args, **kwargs)    
    
    @property
    def _lastseen(self):
        return self._basefun._lastseen
    
    @property
    def optimum(self):
        return self._optimum + self._basefun.optimum
                    
    def _jump(self):
        self._optimum += (0.5-(randn() > 0)) * 2* self.jumpdist  
        self._optimum += randn() * self.jumpdist_std
        self._timer = self.jumptime    
        self.updateOracles()
        
    def registerOracle(self, oracle):
        if not hasattr(self, '_oracles'):
            self._oracles = []
        self._oracles.append(oracle)
        self.updateOracles()
        
    def updateOracles(self):
        if not hasattr(self, '_oracles'):
            return
        for o in self._oracles:
            o._noiseLevel = self._basefun.noiseLevel
            o._curvature = self._basefun.curvature
            o._optimum = self.optimum
                        
    def _shiftedXs(self, xs):
        return xs - self._optimum
    
    # wrap all the other base function methods
    def _f(self, xs): return self._basefun._f(self._shiftedXs(xs))
    def _df(self, xs): return self._basefun._df(self._shiftedXs(xs))
    def _ddf(self, xs): return self._basefun._ddf(self._shiftedXs(xs))
    def expectedLoss(self, xs): return self._basefun.expectedLoss(self._shiftedXs(xs))
    
    def __str__(self):
        return "OptJumpWrapped("+str(self._basefun)+","+str(self.jumptime)+")"
    
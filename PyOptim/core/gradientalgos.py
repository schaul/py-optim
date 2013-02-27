from scipy import mean
from numpy.matlib import repmat
from pybrain.utilities import setAllArgs


class GradientBasedOptimizer(object):
    """ Parent class for a number gradient descent variants. 
    """
    
    # how many samples per step
    batch_size = 1
    
    # callback after each update
    callback = lambda * _: None
    
    # target value (stop after the loss is lower)
    loss_target = None
    
    def __init__(self, provider, init_params, **kwargs):
        self.provider = provider
        self.parameters = init_params.copy()
        self.paramdim = len(init_params)
        self._num_updates = 0
        setAllArgs(self, kwargs)
        self._additionalInit()
        
    def _additionalInit(self):
        """ Abstract: initializations for subclasses. """
    
    def _updateParameters(self):
        """ Abstract: implemented by subclasses. 
        Only affects parameter vector. """
    
    def _computeStatistics(self):
        """ Auxiliary computations that affect state variables,
        but not parameter vector """
        
    def _collectGradients(self):
        """ Obtain the gradient information for the specified number of samples
        from the provider object. """
        self.provider.nextSamples(self.batch_size)
        self._last_gradients = self.provider.currentGradients(self.parameters)    
    
    @property
    def _last_gradient(self):
        """ Makes minibatches transparent. """
        return mean(self._last_gradients, axis=0)
    
    def oneStep(self):
        """ Provided is a matrix, where each row is a sample, 
        and each column corresponds to one parameter. """
        self._collectGradients()
        self._computeStatistics()
        self._updateParameters()
        self._num_updates += 1
        self.callback(self)
        
    @property
    def bestParameters(self):
        return self.parameters
    
    def run(self, maxsteps=None):
        while not self.terminate(maxsteps):
            self.oneStep()
    
    def terminate(self, maxsteps):
        """ Termination criteria """
        if maxsteps is not None:
            if self._num_updates >= maxsteps:
                return True
        if self.loss_target is not None:
            l = self.provider.currentLosses(self.bestParameters)
            if mean(l) <= self.loss_target:
                return True
        return False

    def __str__(self):
        return self.__class__.__name__



class BbpropHessians(GradientBasedOptimizer):
    """ Algorithms that utilize bbprop to get a direct estimate of
    the diagonal Hessians. """
    
    def _collectGradients(self):
        GradientBasedOptimizer._collectGradients(self)        
        self._last_diaghessians = abs(self.provider.currentDiagHess(self.parameters))
          

class FiniteDifferenceHessians(GradientBasedOptimizer):
    """ Algorithms that utilize a finite-difference approximation 
    to the curvatures. """    
    
    def _fdDirection(self):
        """ Abstract """
    
    def _collectGradients(self):
        GradientBasedOptimizer._collectGradients(self)
        v = self._fdDirection()
        tmp = self.provider.currentGradients(self.parameters + v)
        if self.batch_size > 1:
            v = repmat(v, self.batch_size, 1)
        self._last_diaghessians = abs(self._last_gradients - tmp) / v  



from scipy import mean
from pybrain.utilities import setAllArgs



class GradientBasedOptimizer(object):
    """ Parent class for a number gradient descent variants. 
    """
    
    # how many samples per step
    batch_size = 1
    
    # callback after each update
    callback = lambda *_: None
    
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
        return mean(self._last_gradients, axis=1)
    
    def oneStep(self):
        """ Provided is a matrix, where each row is a sample, 
        and each column corresponds to one parameter. """
        self._collectGradients()
        self._computeStatistics()
        self._updateParameters()
        self._num_updates += 1
        self.callback(self)
    
    def run(self, maxsteps=None):
        while not self.terminate(maxsteps):
            self.oneStep()
    
    def terminate(self, maxsteps):
        """ Termination criteria """
        if maxsteps is not None:
            if self._num_updates >= maxsteps:
                return True
        if self.loss_target is not None:
            l = self.provider.currentLosses(self.parameters)
            if mean(l) <= self.loss_target:
                return True
        return False





    
    
    
class SampleProvider(object):
    """ Unified interface for interacting with a model: 
    given a data sample and a parameter vector, it produces 
    gradients, loss values, and potentially other terms like
    diagonal Hessians. 
    
    The samples are iteratively generated, either from a dataset, or from a 
    function, individually or in minibatches, shuffled or not.
    """
    
    batch_size = 1
    
    #optional function that generates diagonal Hessian approximations
    diaghess_fun = None
    
    def __init__(self, paramdim, loss_fun, gradient_fun, **kwargs):
        self.paramdim = paramdim
        self.loss_fun = loss_fun
        self.gradient_fun = gradient_fun        
        setAllArgs(self, kwargs)
        self.nextSamples()
    
    def nextSamples(self, how_many=None):
        """"""
        if how_many is None:
            how_many = self.batch_size
        self._provide(how_many)
    
    def _provide(self, number):
        """ abstract """
        
    def currentGradients(self, params):
        return self.gradient_fun(params)
        
    def currentLosses(self, params):
        return self.loss_fun(params)
        
    def currentDiagHess(self, params):
        if self.diaghess_fun is not None:
            return self.diaghess_fun(params)        
    
class FunctionWrapper(SampleProvider):
    """ Specialized case for a function that can generate samples on the fly. """
    
    def __init__(self, dim, stochfun, **kwargs):
        self.stochfun = stochfun
        SampleProvider.__init__(self, dim, loss_fun=stochfun._f,
                                gradient_fun=stochfun._df,
                                diaghess_fun=stochfun._ddf)
        stochfun._retain_sample=True
        
    def _provide(self, number):
        assert number == 1, 'so far only single new samples...'
        self.stochfun._newSample(self.paramdim, override=True)
        

class DatasetWrapper(SampleProvider):
    """ Specialized case for datasets """
    
    

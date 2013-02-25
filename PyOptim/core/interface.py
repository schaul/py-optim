from scipy import mean
from pybrain.utilities import setAllArgs



class GradientBasedOptimizer(object):
    """ Parent class for a number gradient descent variants. 
    """
    
    # how many samples per step
    batch_size = 1
    
    # callback after each update
    callback = lambda *_: None
    
    def __init__(self, provider, **kwargs):
        self.provider = provider
        self.paramdim = provider.numParameters
        self._num_updates = 0
        setAllArgs(self, **kwargs)
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
        self._last_gradients = self.provider.getGradients()    
    
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
        self.callback()
    
    





class ModelWrapper(object):
    """ Unified interface for interacting with a model: 
    given a data sample and a parameter vector, produce 
    gradients, loss values, and potentially other terms like
    diagonal Hessians. """
    
    diaghess_fun = None
    
    def __init__(self, sample_provider, loss_fun, gradient_fun, **kwargs):
        self.loss_fun = loss_fun
        self.gradient_fun = gradient_fun        
        setAllArgs(self, **kwargs)


    
    
class SampleProvider(object):
    """ A wrapper around a dataset that can iterate over it, sample randomly,
    or generate new points on the fly, where every sample has a unique identifier
    (the seed in the generative case).
    """
    
        
    
    def _provide(self, sample_id):
        """ abstract. """
        
    
    def provideNext(self, how_many=1, sample_id=None):
        """"""
        
    
    
    
    
class DatasetWrapper(SampleProvider):
    """ Specialized case for datasets """
    
    
class FunctionWrapper(SampleProvider):
    """ Specialized case for a function that can generate samples on the fly. """
    
    
    
    
    

    
    
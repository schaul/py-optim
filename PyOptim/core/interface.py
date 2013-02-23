from pybrain.utilities import setAllArgs 



class GradientBasedOptimizer(object):
    """ Parent class for a number gradient descent variants. 
    
    It has multiple ways of being invoked?
    """
    
    
    
    
    def __init__(self, provider, num_parameters, **kwargs):
        self.provider = provider
        self.num_parameters = num_parameters
        self._additionalInit(**kwargs)
        
    def _additionalInit(self, **kwargs):
        setAllArgs(self, **kwargs)
        
    
    
    def oneStep(self):
        """ Provided is a matrix, where each row is a sample, 
        and each column corresponds to one parameter. """
    
    
    





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
    
    
    
    
    

    
    
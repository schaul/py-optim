from random import shuffle
from pybrain.utilities import setAllArgs
    
    
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
    
    record_samples = False
    
    def __init__(self, dim, stochfun, **kwargs):
        self.stochfun = stochfun
        self._seen = []
        SampleProvider.__init__(self, dim, loss_fun=stochfun._f,
                                gradient_fun=stochfun._df,
                                diaghess_fun=stochfun._ddf, **kwargs)
        stochfun._retain_sample = True
        
    def _provide(self, number):
        assert number == 1, 'so far only single new samples...'
        self.stochfun._newSample(self.paramdim, override=True)
        if self.record_samples:
            self._seen.append(self.stochfun._lastseen)
            
    def __str__(self):
        return self.stochfun.__class__.__name__+" n=%s curv=%s "%(self.stochfun.noiseLevel, self.stochfun.curvature)

class DatasetWrapper(FunctionWrapper):
    """ Specialized case for datasets """
    
    shuffling = True
    
    def __init__(self, dataset, stochfun, **kwargs):
        self.dataset = dataset
        assert len(dataset) > 0, 'Must be non-empty'
        dim = len(dataset[0])
        self._indices = range(len(self.dataset))
        self.reset()
        FunctionWrapper.__init__(self, dim, stochfun, **kwargs)
        
    def reset(self):
        self._counter = 0                
        
    def getIndex(self):
        tmp = self._counter % len(self.dataset)
        if tmp == 0 and self.shuffling:
            shuffle(self._indices)
        return self._indices[tmp] 
        
    def _provide(self, number):
        assert number == 1, 'so far only single new samples...'
        i = self.getIndex()
        self.stochfun._lastseen = self.dataset[i]#:i+number]
        self._counter += number
        
from random import shuffle
from scipy import reshape, array
from numpy.matlib import repmat
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
        
    def nextSamples(self, how_many):
        """ Obtain a certain number of samples. """
        self.batch_size = how_many
        self._provide()
    
    def _provide(self):
        """ abstract """
        
    def currentGradients(self, params):
        if self.batch_size > 1:
            params = repmat(params, 1, self.batch_size)
            res = self.gradient_fun(params)            
            return reshape(res, (self.batch_size, self.paramdim))
        else:
            return self.gradient_fun(params)
        
    def currentLosses(self, params):
        if self.batch_size > 1:
            params = repmat(params, 1, self.batch_size)
            res = self.loss_fun(params)
            return reshape(res, (self.batch_size, self.paramdim))
        else:
            return self.loss_fun(params)
        
    def currentDiagHess(self, params):
        if self.diaghess_fun is None:
            return
        if self.batch_size > 1:
            params = repmat(params, 1, self.batch_size)
            res = self.diaghess_fun(params)
            return reshape(res, (self.batch_size, self.paramdim))
        else:
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
        
    def _provide(self):
        self.stochfun._newSample(self.paramdim*self.batch_size, override=True)
        if self.record_samples:
            ls = self.stochfun._lastseen
            if self.batch_size == 1:
                self._seen.append(ls)
            else:
                for l in reshape(ls, (self.batch_size, self.paramdim)):
                    self._seen.append(l)
                    
    def reset(self):
        """ Nothing to be done. """
            
    def __str__(self):
        return self.stochfun.__class__.__name__+" n=%s curv=%s "%(self.stochfun.noiseLevel, self.stochfun.curvature)

class DatasetWrapper(FunctionWrapper):
    """ Specialized case for datasets """
    
    shuffling = True
    
    def __init__(self, dataset, stochfun, **kwargs):
        self.dataset = dataset
        assert len(dataset) > 0, 'Must be non-empty'
        dim = dataset[0].size
        self._indices = range(len(self.dataset))
        self.reset()
        FunctionWrapper.__init__(self, dim, stochfun, **kwargs)
        
    def reset(self):
        self._counter = 0                
        
    def getIndex(self):
        tmp = self._counter % len(self.dataset)
        if tmp + self.batch_size >= len(self.dataset):
            # dataset is not a multiple of batchsizes
            tmp = 0
        if tmp == 0 and self.shuffling:
            shuffle(self._indices)
        return self._indices[tmp] 
        
    def _provide(self):
        i = self.getIndex()
        if self.batch_size == 1:
            x = self.dataset[i]
        else:
            x = array(self.dataset[i:i+self.batch_size])
        self.stochfun._lastseen = reshape(x, (1, self.batch_size * self.paramdim))
        self._counter += self.batch_size
        
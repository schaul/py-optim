"""
A few example functions of stochastic loss.

Each of them is one-dimensional, but can be evaluated on an array of points simultaneously.
"""

from scipy import randn, zeros_like, sign, array, zeros, exp, mean, argmin, \
                    ravel, reshape, ones_like, sqrt, pi, rand, log
from scipy.special import erf
from numpy.matlib import repmat
from pybrain.utilities import setAllArgs
from numpy.random import seed, randint
            

class StochFun(object):
    """ One-dimensional stochastic loss functions. 
    """
    
    # properties of the function
    noiseLevel = 1.
    curvature = 1    
    sparsity = 1.
    
    # some outlier samples?
    outlierProb = 0.
    outlierLevel = 100
    
    optimum = 0. # default value where the optimum lies
    
    # for plotting
    fmax = 4 
    dfmax = 1.2
    
    # a mean to retain the current sample
    _retain_sample = False
    
    def __init__(self, **args):
        setAllArgs(self, args)
        self._lastseen = None
        
    def _f(self, xs):
        """ One fixed noise value, for many parameters. Implemented by subclasses """
    
    def _df(self, xs):
        """ One fixed noise value, for many parameters. Implemented by subclasses """
   
    def _newSample(self, nsamples, override=False):
        if self._lastseen is None or override or not self._retain_sample:
            self._lastseen = randn(1, nsamples) * self.noiseLevel        
   
    def _noise(self, shape):
        if len(shape) > 1:
            nsynch, nsamples = shape
        else:
            nsamples = shape[0]
            nsynch = 1
        self._newSample(nsamples)
        res = repmat(self._lastseen, nsynch, 1)
        if self.outlierProb > rand():
            print 'OUT', nsynch, nsamples
            res[:,randint(1,nsamples)] *= self.outlierLevel            
        return res
        
    def _mask(self, shape):
        if len(shape) > 1:
            nsynch, nsamples = shape
        else:
            nsamples = shape[0]
            nsynch = 1
        return repmat(rand(1, nsamples) < self.sparsity, nsynch, 1)
    
    # these expectations should be overridden if a closed form is available.
    ESamples= 500
    def expectedLoss(self, xs, seeded=None):
        """ True loss value, in expectation, for each value in xs. """
        if seeded is not None:
            tmp = int(abs(randn()*1e6))
            seed(seeded)
        self._newSample(len(xs), override=True)
        rxs = repmat(xs, self.ESamples, 1)
        res = mean(self._f(rxs), axis=0)
        if seeded is not None:
            seed(tmp)
        return res
        
    def expectedGradient(self, xs):
        """ The same for the gradient """        
        rxs = repmat(xs, self.ESamples, 1).T
        return mean(self._df(rxs), axis=1)
        
    def expectedHessian(self, xs):
        """ Finite-difference approximation of the expected gradient """        
        rxs = repmat(xs, self.ESamples, 1).T
        return mean(self._ddf(rxs), axis=1)
        
        
    fd_eps = 0.5
    def expectedHessianFD(self, xs):
        """ Finite-difference approximation of the expected gradient """
        tmp = zeros(len(xs)*2)
        tmp[:len(xs)] = xs-self.fd_eps
        tmp[len(xs):] = xs+self.fd_eps
        g = self.expectedGradient(tmp)
        return (-g[:len(xs)]+g[len(xs):])/self.fd_eps/2
    
    
    def empMinLossRate(self, xs, bsize=1):
        """ What is the step-size/learning rate that empirically leads to the minimum expected loss. """
        candrates = exp(array(range(1,101))/25.-3)
        self.ESamples /= 10
        outcomes = zeros((len(xs),len(candrates),self.ESamples))
        for i, r in enumerate(candrates):
            for j in range(self.ESamples):
                for _ in range(bsize):
                    outcomes[:,i,j] += xs/float(bsize) - r * self._df(xs)/float(bsize)
        ls = self.expectedLoss(ravel(outcomes))
        ls = reshape(ls, (len(xs),len(candrates),self.ESamples))
        self.ESamples *= 10
        
        avgresults = mean(ls, axis=2)
        bestindices = argmin(avgresults, axis=1)
        bestrates = array([candrates[i] for i in bestindices])
        return bestrates
    
    def signalToNoiseRatio(self, xs):
        """ What is the one-sample signal-to-noise ratio. """         
        rxs = repmat(xs, self.ESamples, 1).T
        gs = self._df(rxs)
        g2s = mean(gs **2, axis=1)
        gs = mean(gs, axis=1)
        return gs**2/g2s
    
        
    def maxLogGain(self, numsamples, x0=1):
        """ Maximal gain in loss, given a number of sample gradients
        (just the order of magnitude, for plotting scales) """
        return 10
    
class StochAbs(StochFun):
    """ Absolute value, Gaussian noise on position of kink. """
        
    def _f(self, xs):
        return abs(xs + self._noise(xs.shape))  * self.curvature
            
    def _df(self, xs):
        return sign(xs + self._noise(xs.shape)) * self.curvature
    
    def _ddf(self, xs):
        return zeros_like(xs)

    def expectedLoss(self, xs):
        return (xs * erf(xs/sqrt(2*self.noiseLevel**2))
                + 
                sqrt(2/pi)*self.noiseLevel * exp(-xs**2/2/self.noiseLevel**2)
                ) * self.curvature
    
    def __str__(self):
        return "Abs %.2f" % self.noiseLevel
    

    def maxLogGain(self, numsamples, x0=1):
        """ Maximal gain in loss, given a number of sample gradients
        (just the order of magnitude, for plotting scales) """
        return max(0,log(numsamples)-log(self.noiseLevel)+2*log(abs(x0-self.optimum)))


class StochRectLin(StochFun):
    """ Rectified linear (linear near zero) """        
    kink = -1e-0
    optimum = -1e100
    
    def _f(self, xs):
        tmp = xs + self._noise(xs.shape)-self.kink
        return tmp*(tmp>0)  * self.curvature    
            
    def _df(self, xs):
        tmp = xs + self._noise(xs.shape)-self.kink
        return (tmp>0)  * self.curvature
        
    def _ddf(self, xs):
        return zeros_like(xs)

    def __str__(self):
        return "RectLin %.2f" % self.noiseLevel
    
    
    def maxLogGain(self, numsamples, x0=1):
        return max(0,log(numsamples)-log(self.noiseLevel))

class StochGauss(StochFun):
    """ Upside-down Gaussian bump: concave function with Hessian changing all the time. """        
    fmax =1.1
    
    def _f(self, xs):
        tmp = xs + self._noise(xs.shape)
        return (1-exp(-tmp**2/2))  * self.curvature    
                
    def _df(self, xs):
        tmp = xs + self._noise(xs.shape)
        return tmp * (exp(-tmp**2/2))  * self.curvature   
        
    def _ddf(self, xs):
        tmp = xs + self._noise(xs.shape)
        return (1- tmp**2) *exp(-tmp**2/2)  * self.curvature  

    def expectedLoss(self, xs):
        a = 1./sqrt(self.noiseLevel**2+1)
        return(1-exp(-xs**2/2*(a**2))*a)  * self.curvature    
    
    def __str__(self):
        return "Gauss %.2f" % self.noiseLevel


class StochRectLinFlat(StochRectLin):
    """ Rectified linear (flat near zero) """        
    
    def _f(self, xs):
        tmp = xs + self._noise(xs.shape)-self.kink
        return tmp*(tmp<0)  * self.curvature +3    
            
    def _df(self, xs):
        tmp = xs + self._noise(xs.shape)-self.kink
        return - 1* (tmp<0)  * self.curvature
            
    def __str__(self):
        return "RectLinFlat %.2f" % self.noiseLevel

    
class StochQuad(StochFun):
    """ Absolute value, Gaussian noise on position of kink. """
        
    fmax = 6
        
    def _f(self, xs):
        return 0.5* (xs + self._noise(xs.shape))**2 * self.curvature
                
    def _df(self, xs):
        return (xs + self._noise(xs.shape)) * self.curvature    
    
    def _ddf(self, xs):
        return ones_like(xs) * self.curvature
    
    def __str__(self):
        return "Quad %.2f" % self.noiseLevel
    
    def expectedLoss(self, xs):
        return (0.5 * xs **2 + 0.5 * self.noiseLevel **2) * self.curvature
    
    def maxLogGain(self, numsamples, x0=1):
        return max(0,2+log(numsamples)-2*log(self.noiseLevel)+2*log(abs(x0-self.optimum)))
    
    
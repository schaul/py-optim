from scipy import mean, ones_like, clip
from sgd import SGD
from core.gradientalgos import BbpropHessians, FiniteDifferenceHessians


class vSGD(SGD, BbpropHessians):
    """ vSGD: SGD with variance-adapted learning rates. """

    # how to initialize the running averages
    slow_constant = 2
    init_samples = 3
    
    # avoiding numerical instability
    epsilon = 1e-9
    outlier_level = 2

    def _additionalInit(self):
        # default setting
        if self.slow_constant is None:
            self.slow_constant = max(1, int(self.paramdim / 10.))
        
        # get a few initial samples to work with
        tmp = self.batch_size
        self.batch_size = self.init_samples * tmp
        self._collectGradients()
        self._num_updates += self.init_samples
        self.batch_size = tmp
        
        # initialize statistics
        grads = self._last_gradients
        self._gbar = mean(grads, axis=0)
        self._vbar = (mean(grads ** 2, axis=0) + self.epsilon) * self.slow_constant
        hess = clip(self._last_diaghessians, 1, 1 / self.epsilon) 
        self._hbar = mean(hess, axis=0) * self.slow_constant
        
        # time constants
        self._taus = (ones_like(self.parameters) + self.epsilon) * self.init_samples
        
    
    def _detectOutliers(self):
        """ Binary vector for which dimension saw an outlier gradient. """
        var = (self._vbar - self._gbar ** 2) / self.batch_size
        return (self._last_gradient - self._gbar) ** 2 > self.outlier_level ** 2 * var 


    def _computeStatistics(self):
        gs = self._last_gradients 
        hs = abs(mean(self._last_diaghessians, axis=0)) + self.epsilon
        bs = self.batch_size
        
        # slow down updates if the last sample was an outlier
        if self.outlier_level is not None:
            self._taus[self._detectOutliers()] += 1
        
        # update statistics
        fract = 1. / self._taus
        self._gbar *= (1. - fract)
        self._gbar += fract * mean(gs, axis=0)
        self._vbar *= (1. - fract)
        self._vbar += fract * mean(gs ** 2, axis=0) + self.epsilon            
        self._hbar *= (1. - fract)
        self._hbar += fract * hs
        
        # update time constants based on the variance-part of the learning rate                 
        vpart = self._gbar ** 2 / (1. / bs * self._vbar + 
                                   (bs - 1.) / bs * self._gbar ** 2)
        self._taus *= (1. - vpart)
        self._taus += 1 + self.epsilon
                                    
    
    @property
    def learning_rate(self):
        # effective batchsize
        bs = self.batch_size
        return self._gbar ** 2 / (1. / bs * self._vbar + 
                                  (bs - 1.) / bs * self._gbar ** 2) / self._hbar
                                  
                                  
class vSGDfd(FiniteDifferenceHessians, vSGD):
    """ vSGD with finite-difference estimate of diagonal Hessian """
    
    def _fdDirection(self):
        if self._num_updates == 0:
            return self._last_gradient
        else:
            return self._gbar
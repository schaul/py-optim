from scipy import mean, ones_like, clip, logical_or, median
from sgd import SGD
from core.gradientalgos import BbpropHessians, FiniteDifferenceHessians


class vSGD(SGD, BbpropHessians):
    """ vSGD: SGD with variance-adapted learning rates,
    as described in Schaul, Zhang & LeCun 2012. """

    # how to initialize the running averages
    slow_constant = 2
    init_samples = 10
    
    # avoiding numerical instability
    epsilon = 1e-9
    outlier_level = 1

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
        
        # mean gradient vector
        self._gbar = mean(self._last_gradients, axis=0)
        # mean squared gradient
        self._vbar = (mean(self._last_gradients ** 2, axis=0) + self.epsilon) * self.slow_constant
        
        # mean diagonal Hessian
        hs = clip(mean(self._last_diaghessians, axis=0), 1, 1/self.epsilon)
        self._hbar = hs * self.slow_constant
        
        # time constants
        self._taus = (ones_like(self.parameters) + self.epsilon) * self.init_samples
        
        # for debugging
        self._print_quantities = [('tau', self._taus), 
                                  ('p', self.parameters), 
                                  ('v', self._vbar), 
                                  ('g', self._gbar),
                                  ('vpa', self._gbar**2/self._vbar), 
                                  ('h', self._hbar),
                                  ]
        
    
    def _detectOutliers(self):
        """ Binary vector for which dimension saw an outlier gradient. """
        var = (self._vbar - self._gbar ** 2) / self.batch_size
        return (self._last_gradient - self._gbar) ** 2 > self.outlier_level ** 2 * var 


    def _computeStatistics(self):
        grads = self._last_gradients 
        hs = mean(self._last_diaghessians, axis=0) + self.epsilon
        
        # slow down updates if the last sample was an outlier
        if self.outlier_level is not None:
            self._taus[self._detectOutliers()] += 1
        
        # update statistics
        fract = 1. / self._taus
        self._gbar *= 1. - fract
        self._gbar += fract * mean(grads, axis=0)
        self._vbar *= 1. - fract
        self._vbar += fract * mean(grads ** 2, axis=0) + self.epsilon            
        self._hbar *= 1. - fract
        self._hbar += fract * hs
        
        # update time constants based on the variance-part of the learning rate  
        if self.batch_size > 1:                
            self._vpart = self._gbar ** 2 / (1. / self.batch_size * self._vbar + 
                                       (self.batch_size - 1.) / self.batch_size * self._gbar ** 2)
        else:
            self._vpart = self._gbar ** 2 / self._vbar
        self._taus *= 1. - self._vpart
        self._taus += 1 + self.epsilon
        del hs, fract
    
    @property
    def learning_rate(self):
        return self._vpart / self._hbar
    
    def _printStuff(self):
        print self._num_updates, 
        for n, a in self._print_quantities:
            print n, round(median(a),4), '\t',
        print
        
                                  
                                  
class vSGD_original(vSGD):
    """ The original version from 2012 had not outlier detection. """
    outlier_level = None    
                                  
                                  
class vSGDfd(FiniteDifferenceHessians, vSGD):
    """ vSGD with finite-difference estimate of diagonal Hessian,
    as described in Schaul & LeCun 2013. """
            
    @property
    def learning_rate(self):            
        return self._vpart * (self._hbar / self._vhbar)
    
    def _additionalInit(self):
        vSGD._additionalInit(self)
        h2s = clip(mean(self._last_diaghessians **2, axis=0), 1, 1/self.epsilon)
        self._vhbar = h2s * self.slow_constant**2     
        self._print_quantities.extend([('vh', self._vhbar),
                                       ('hpa', self._hbar/self._vhbar),
                                       ])   
    
    def _computeStatistics(self):
        vSGD._computeStatistics(self)
        fract = 1. / self._taus
        self._vhbar *= (1. - fract)
        self._vhbar += fract * mean(self._last_diaghessians ** 2, axis=0) + self.epsilon
        del fract        
    
    def _fdDirection(self):
        if self._num_updates == 0:
            return self._last_gradient+self.epsilon
        else:
            return self._gbar+self.epsilon
        
    def _detectOutliers(self):
        """ Binary vector for which dimension saw an outlier gradient:
        considers oultier curvature as well. """
        hs = mean(self._last_diaghessians, axis=0) 
        var = (self._vhbar - self._hbar ** 2) / self.batch_size
        res = logical_or(vSGD._detectOutliers(self),
                         (hs - self._hbar) ** 2 > self.outlier_level ** 2 * var)
        del hs, var
        return res


from scipy import sqrt, ones
from sgd import SGD


class AdaGrad(SGD):
    """ ADAGRAD algorithm with element-wise adaptive learning rates. """    

    init_lr = 1e-2
    
    def _additionalInit(self):
        self._acc_grad_var = ones(self.paramdim)

    def _computeStatistics(self):
        self._acc_grad_var += self._last_gradient**2
            
    @property
    def learning_rate(self):
        return self.init_lr/sqrt(self._acc_grad_var)
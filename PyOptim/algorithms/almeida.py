from scipy import ones, clip
from amari import Amari


class Almeida(Amari):
    """ Almeida's adaptive learning rates method, for the diagonal case. """    
       
    minimal_lr = 1e-10
    maximal_lr = 1e10
       
    def _additionalInit(self):
        Amari._additionalInit(self)
        self._previous_gradient = ones(self.paramdim)
            
    def _computeStatistics(self):
        Amari._computeStatistics(self)
        self._previous_gradient = self._last_gradient.copy()
   
    @property
    def learning_rate(self):
        return clip(self.init_lr/self._acc_grad_var*self._previous_gradient, self.minimal_lr, self.maximal_lr)
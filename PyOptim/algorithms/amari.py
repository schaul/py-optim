from scipy import ones
from sgd import SGD


class Amari(SGD):
    """ Amari's natural gradient algorithm, simplified to the diagonal case. """    

    init_lr = 1e-3
    time_const = 1000

    def _additionalInit(self):
        self._acc_grad_var = ones(self.paramdim)

    def _computeStatistics(self):
        self._acc_grad_var *= (1 - 1. / self.time_const)
        self._acc_grad_var += 1. / self.time_const * self._last_gradient ** 2
            
    @property
    def learning_rate(self):
        return self.init_lr / self._acc_grad_var

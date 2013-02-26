from scipy import power
from amari import Amari


class RMSProp(Amari):
    """ Root-mean-square-normalized SGD (Hinton 2012). """    

    exponent = -0.5

    @property
    def learning_rate(self):
        return self.init_lr * power(self._acc_grad_var, self.exponent)

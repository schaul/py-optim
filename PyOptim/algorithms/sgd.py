from core.interface import GradientBasedOptimizer


class SGD(GradientBasedOptimizer):
    """ Stochastic gradient descent, with a single global learning rate 
    (by default, a fixed value) """
    
    learning_rate = 1e-3
    
    def _updateParameters(self):
        self.parameters -= self.learning_rate * self._last_gradient
        
        
    
class AnnealingSGD(SGD):
    """ SGD with decaying learning rates """
    init_lr = 1e-3
    lr_decay = None
    
    @property
    def learning_rate(self):
        return self.init_lr / (1. + self._num_updates * self.lr_decay)  
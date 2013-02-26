from scipy import zeros
from sgd import SGD


class MomentumSGD(SGD):
    """ SGD with momentum. """    

    momentum = 0.9

    def _additionalInit(self):
        self._last_update = zeros(self.paramdim)

    def _updateParameters(self):
        self._last_update *= self.momentum
        self._last_update += self.learning_rate * self._last_gradient
        self.parameters -= self._last_update
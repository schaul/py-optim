from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised import BackpropTrainer
from pybrain.datasets import SupervisedDataSet
from core.datainterface import ModuleWrapper
from algorithms.sgd import SGD
from scipy import mean


class XORDataSet(SupervisedDataSet):
    """ A dataset for the XOR function."""
    def __init__(self):
        SupervisedDataSet.__init__(self, 2, 1)
        self.addSample([0,0],[0])
        self.addSample([0,1],[1])
        self.addSample([1,0],[1])
        self.addSample([1,1],[0])


def printy(s):
    if s._num_updates % 10 == 0:
        print s._num_updates, #s.bestParameters, 
        print mean(s.provider.currentLosses(s.bestParameters))
    
def testOldTraining():
    d = XORDataSet()
    n = buildNetwork(d.indim, 4, d.outdim, recurrent=True, bias=True)
    t = BackpropTrainer(n, learningrate = 0.01, momentum = 0.99, verbose = True)
    t.trainOnDataset(d, 1000)
    t.testOnData(verbose= True)

def testNewTraining(hidden = 15):
    d = XORDataSet()
    n = buildNetwork(d.indim, hidden, d.outdim, recurrent=False)
    provider = ModuleWrapper(d, n, shuffling=False)
    algo = SGD(provider, n.params.copy(), callback=printy, learning_rate=0.2)
    algo.run(200)
    
if __name__ == '__main__':
    #testOldTraining()
    testNewTraining()
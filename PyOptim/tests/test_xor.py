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
        self.addSample([0, 0], [0])
        self.addSample([0, 1], [1])
        self.addSample([1, 0], [1])
        self.addSample([1, 1], [0])


def printy(s):
    if s._num_updates % (100 / s.batch_size) == 0:
        print s._num_updates * s.batch_size, #s.bestParameters, 
        s.provider.nextSamples(4)
        print mean(s.provider.currentLosses(s.bestParameters))
        #s.provider.nextSamples(1)
        
    
def testOldTraining(hidden=15):
    d = XORDataSet()
    n = buildNetwork(d.indim, hidden, d.outdim, recurrent=False)
    t = BackpropTrainer(n, learningrate=0.01, momentum=0., verbose=False)
    t.trainOnDataset(d, 250)
    t.testOnData(verbose=True)

def testNewTraining(hidden=15):
    d = XORDataSet()
    n = buildNetwork(d.indim, hidden, d.outdim, recurrent=False)
    provider = ModuleWrapper(d, n, shuffling=False)
    algo = SGD(provider, n.params.copy(), callback=printy, learning_rate=0.01)
    algo.run(1000)
    
def testBatchTraining(hidden=15):
    d = XORDataSet()
    n = buildNetwork(d.indim, hidden, d.outdim, recurrent=False)
    provider = ModuleWrapper(d, n, shuffling=False)
    algo = SGD(provider, n.params.copy(), callback=printy, learning_rate=0.01, batch_size=4)
    algo.run(250)
    
if __name__ == '__main__':
    testOldTraining()
    print '\n' * 3
    print 'SGD'
    testNewTraining()
    print '\n' * 3
    print 'Batch'
    testBatchTraining()

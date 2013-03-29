from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised import BackpropTrainer
from pybrain.datasets import SupervisedDataSet
from core.datainterface import ModuleWrapper
from algorithms.sgd import SGD
from algorithms.vsgd import vSGDfd
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
    if ((s._num_updates * s.batch_size < 100 
         and s._num_updates % (20 / s.batch_size) == 0)
        or s._num_updates % (100 / s.batch_size) == 0):
        print s._num_updates * s.batch_size, #s.bestParameters, 
        s.provider.nextSamples(4)
        print mean(s.provider.currentLosses(s.bestParameters))
        #s.provider.nextSamples(1)
        
    
def testOldTraining(hidden=15, n=None):
    d = XORDataSet()
    if n is None:
        n = buildNetwork(d.indim, hidden, d.outdim, recurrent=False)
    t = BackpropTrainer(n, learningrate=0.01, momentum=0., verbose=False)
    t.trainOnDataset(d, 250)
    t.testOnData(verbose=True)

def testNewTraining(hidden=15, n=None):
    d = XORDataSet()
    if n is None:
        n = buildNetwork(d.indim, hidden, d.outdim, recurrent=False)
    provider = ModuleWrapper(d, n, shuffling=False)
    algo = SGD(provider, n.params.copy(), callback=printy, learning_rate=0.01, momentum=0.99)
    algo.run(1000)
    
def testNewTraining2(hidden=15, n=None):
    d = XORDataSet()
    if n is None:
        n = buildNetwork(d.indim, hidden, d.outdim, recurrent=False)
    provider = ModuleWrapper(d, n, shuffling=False)
    algo = vSGDfd(provider, n.params.copy(), callback=printy)
    algo.run(1000)
    
def testBatchTraining(hidden=15, n=None):
    d = XORDataSet()
    if n is None:
        n = buildNetwork(d.indim, hidden, d.outdim, recurrent=False)
    provider = ModuleWrapper(d, n, shuffling=False)
    algo = SGD(provider, n.params.copy(), callback=printy, learning_rate=0.04, batch_size=4)
    algo.run(250)
    
def testSome():
    
    net = buildNetwork(2, 15, 1, bias=True, recurrent=True)
    p0 = net.params.copy()
    testOldTraining(n=net)
    net._setParameters(p0)
    print '\n' * 2
    print 'Batch'
    testBatchTraining(n=net)
    net._setParameters(p0)
    print '\n' * 2
    print 'SGD'
    testNewTraining(n=net)
    net._setParameters(p0)
    print '\n' * 2
    print 'vSGD'
    testNewTraining2(n=net)
    net._setParameters(p0)
    
    
if __name__ == '__main__':
    testSome()

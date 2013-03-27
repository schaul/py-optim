from core.datainterface import ModuleWrapper
from algorithms import SGD
import pylab

            
def testPlot1():
    dim = 15
    from scipy import rand, dot
    from pybrain.datasets import SupervisedDataSet
    from pybrain import LinearLayer, FullConnection, FeedForwardNetwork
    from pybrain.utilities import dense_orth
    net = FeedForwardNetwork()
    net.addInputModule(LinearLayer(dim, name='in'))
    net.addOutputModule(LinearLayer(1, name='out'))
    net.addConnection(FullConnection(net['in'], net['out']))
    net.sortModules()
    
    ds = SupervisedDataSet(dim, 1)
    ds2 = SupervisedDataSet(dim, 1)
    R = dense_orth(dim)
    for _ in range(1000):
        tmp = rand(dim) > 0.5
        tmp2 = dot(tmp, R)
        ds.addSample(tmp, [tmp[-1]])
        ds2.addSample(tmp2, [tmp[-1]])
        
    f = ModuleWrapper(ds, net)
    f2 = ModuleWrapper(ds2, net)
    
    # tracking progress by callback
    ltrace = []
    def storer(a):
        ltrace.append(a.provider.currentLosses(a.bestParameters))
    
    x = net.params
    x *= 0.001
    
    algo = SGD(f, net.params.copy(), callback=storer, learning_rate=0.2)
    algo.run(1000)
    pylab.plot(ltrace, 'r-')
    
    del ltrace[:]
    
    algo = SGD(f2, net.params.copy(), callback=storer, learning_rate=0.2)
    algo.run(1000)
    pylab.plot(ltrace, 'g-')
    
    pylab.semilogy()
    pylab.show()

if __name__ == '__main__':
    testPlot1()

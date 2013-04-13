"""
Regression on the classic corn-fertilizer-insecticide toy dataset. 

Test design: Roy Lowrance.
"""

data = [(40, 6, 4),
        (44, 10, 4),
        (46, 12, 5),
        (48, 14, 7),
        (52, 16, 9),
        (58, 18, 12),
        (60, 22, 14),
        (68, 24, 20),
        (74, 26, 21),
        (80, 32, 24)]

#known optimal weight vector
optw = [31.98, 0.65, 1.11]

def prepare():
    """ Shape the dataset, and build the linear classifier """
    from pybrain import LinearLayer, FullConnection, FeedForwardNetwork
    from pybrain.datasets import SupervisedDataSet
    D = SupervisedDataSet(3, 1)
    for c, f, i in data:
        D.addSample([1, f, i], [c]) 
    
    net = FeedForwardNetwork()
    net.addInputModule(LinearLayer(D.indim, name='in'))
    net.addOutputModule(LinearLayer(1, name='out'))
    net.addConnection(FullConnection(net['in'], net['out']))
    net.sortModules()
    return D, net

def rmse(D, net):
    from scipy import sqrt
    s = 0.
    for inp, targ in D:
        net.reset()
        net.resetDerivatives()
        outp = net.activate(inp)
        s += (outp - targ)[0] ** 2
        outp = net.activate(inp)        
    net.reset()
    net.resetDerivatives()
    return sqrt(s / len(D))


def testKnown():
    """ Just verifying that things are correctly set up."""
    D, net = prepare()
    optw1 = [22.91, 1.82, 0.10]
    optw5 = [20.06, 1.71, 0.92]

    print 'rand', rmse(D, net)
    net._setParameters(optw1)
    print '1', rmse(D, net)
    net._setParameters(optw5)
    print '5', rmse(D, net)
    net._setParameters(optw)
    print 'opt', rmse(D, net)
    

def printy(s):
    if ((s._num_updates * s.batch_size < 1000 
         and s._num_updates % (100 / s.batch_size) == 0)
        or s._num_updates % (1000 / s.batch_size) == 0):
        print s._num_updates * s.batch_size, 
        s.provider.module._setParameters(s.bestParameters)
        print rmse(s.provider.dataset, s.provider.module), s.bestParameters
            

def testTrain():
    from algorithms.vsgd import vSGDfd
    from core.datainterface import ModuleWrapper
    D, net = prepare()
    print 'init', rmse(D,net), net.params
    print
    p0 = net.params.copy()
    for bs in range(1, 11):
        net._setParameters(p0.copy())
        provider = ModuleWrapper(D, net, shuffling=True)
        algo = vSGDfd(provider, net.params.copy(),batch_size=bs,
                      #callback=printy,
                      #verbose=True,
                      )
        # 1000 epochs
        algo.run(1000 * 
                 len(D)/bs)
        #print 
        net._setParameters(algo.bestParameters)
        print 'batch size',bs, 'RMSE', rmse(D, net), 'weights', net.params
        print

if __name__ == "__main__":
    #testKnown()
    testTrain()

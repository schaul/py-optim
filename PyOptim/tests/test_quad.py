from scipy import ones, mean

from core.datainterface import FunctionWrapper, DataFunctionWrapper
from benchmarks.stoch_1d import StochQuad
from algorithms.sgd import SGD
from algorithms.amari import Amari
from algorithms.almeida import Almeida
from algorithms.rmsprop import RMSProp
from algorithms.adagrad import AdaGrad
from algorithms.quadoracle import OracleSGD
from algorithms.momentum import MomentumSGD
from algorithms.averaging import AveragingSGD
        
def testWrapper(dim=5):
    f1 = FunctionWrapper(dim, StochQuad(noiseLevel=0.1))
    print f1.currentGradients(ones(dim))
    print f1.currentGradients(ones(dim))
    print
    print f1.currentGradients(ones(dim) + 0.1)
    f1.nextSamples()
    print
    print f1.currentGradients(ones(dim))
    print 
    
def printy(s):
    if s._num_updates % 2 == 0:
        print s._num_updates, s.bestParameters, 
        print mean(s.provider.currentLosses(s.bestParameters))
    
def testSGD(dim=3):
    f = FunctionWrapper(dim, StochQuad(noiseLevel=0.2))
    x0 = ones(dim)
    algo = SGD(f, x0, callback=printy, learning_rate=0.2, loss_target=0.01)
    algo.run(100)
    print
    
    
def testOracle(dim=3):
    f = FunctionWrapper(dim, StochQuad(noiseLevel=0.1))
    x0 = ones(dim)
    algo = OracleSGD(f, x0, callback=printy, loss_target=1e-5)
    algo.run(100)
    print mean(algo.parameters **2)
    
def testAlgos(dim=3):
    # generate a dataset
    f = StochQuad(noiseLevel=0.2)
    fw = FunctionWrapper(dim, f, record_samples=True)
    [fw.nextSamples(1) for _ in range(100)]
    ds = fw._seen
    dw = DataFunctionWrapper(ds, f, shuffling=False)
    
    x0 = ones(dim)
    for algoclass in [SGD, SGD, OracleSGD, Almeida, Amari, RMSProp, AdaGrad,
                      MomentumSGD, AveragingSGD]:
        dw.reset()
        print algoclass.__name__
        algo = algoclass(dw, x0, callback=printy)
        algo.run(16)
        
    
def testMinibatch(dim=4):
    f = StochQuad(noiseLevel=0.2)
    fw = FunctionWrapper(dim, f, record_samples=True)
    x0 = ones(dim)
    for mb in [1,3,15,250]:
        print 'minibatch', mb
        algo = SGD(fw, x0, callback=printy, batch_size=mb, learning_rate=0.1)
        algo.run(10)
        print
    [fw.nextSamples(1) for _ in range(2500)]
    dw = DataFunctionWrapper(fw._seen, f, shuffling=False)
    print 'Fixed samples'
    for mb in [1,3,15,250]:
        print 'minibatch', mb
        dw.reset()
        algo = SGD(dw, x0, callback=printy, batch_size=mb, learning_rate=0.1)
        algo.run(10)
        print
    
    
if __name__ == "__main__":
    #testWrapper()
    #testSGD()
    testOracle()
    #testAlgos()
    #testMinibatch()

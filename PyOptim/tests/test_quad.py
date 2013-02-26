from benchmarks.stoch_1d import StochQuad
from core.interface import FunctionWrapper
from scipy import ones

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
    print s._num_updates, s.parameters, s.provider.currentLosses(s.parameters)
    
def testSGD(dim=3):
    f = FunctionWrapper(dim, StochQuad(noiseLevel=0.2))
    x0 = ones(dim)
    from algorithms.sgd import SGD
    algo = SGD(f, x0, callback=printy, learning_rate=0.2, loss_target=0.01)
    algo.run(100)
    print
    
    
def testOracle(dim=3):
    from algorithms.quadoracle import OracleSGD
    f = FunctionWrapper(dim, StochQuad(noiseLevel=0.2))
    x0 = ones(dim)
    algo = OracleSGD(f, x0, callback=printy, loss_target=0.01)
    algo.run(100)
    print
    
if __name__ == "__main__":
    testWrapper()
    testSGD()
    testOracle()

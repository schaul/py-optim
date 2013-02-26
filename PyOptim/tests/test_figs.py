from tools.experiments import lossTraces
from core.datainterface import FunctionWrapper
from benchmarks.stoch_1d import StochQuad
from algorithms import SGD, AdaGrad, Amari, OracleSGD, RMSProp
from tools.plotting import plotWithPercentiles, algo_colors
import pylab

def testPlot1(dim=20):
    f = FunctionWrapper(dim, StochQuad(noiseLevel=0.2))
    ls = lossTraces(fwrap=f, aclass=SGD, dim=dim, maxsteps=100, algoparams={'learning_rate':0.2})
    pylab.plot(ls, 'b:')
    pylab.plot(pylab.mean(ls, axis=1), 'r-')
    pylab.semilogy()
    pylab.show()


def testPlot2(dim=100, maxsteps=250):
    f = FunctionWrapper(dim, StochQuad(noiseLevel=1, curvature=10))
    for aclass, aparams in [(SGD, {'learning_rate':0.1}),
                            (SGD, {'learning_rate':0.01}),
                            (AdaGrad, {'init_lr':0.3}),
                            (Amari, {'init_lr':0.1, 'time_const':100}),
                            (RMSProp, {'init_lr':0.1}),
                            (OracleSGD, {}),
                            ]:
        ls = lossTraces(fwrap=f, aclass=aclass, dim=dim, 
                        maxsteps=maxsteps, algoparams=aparams)
        plotWithPercentiles(ls, algo_colors[aclass], aclass.__name__)
    pylab.semilogy()
    pylab.xlim(0, maxsteps)
    pylab.legend()
    pylab.show()

    
    
if __name__ == "__main__":
    #testPlot1()
    testPlot2()
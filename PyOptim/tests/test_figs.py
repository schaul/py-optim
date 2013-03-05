from tools.experiments import lossTraces
from core.datainterface import FunctionWrapper, DataFunctionWrapper
from benchmarks.stoch_1d import StochQuad, StochAbs
from algorithms import SGD, AdaGrad, Amari, OracleSGD, RMSProp, vSGDfd, vSGD
from tools.plotting import plotWithPercentiles, algo_colors, plotHeatmap
import pylab

def testPlot1(trials=20):
    f = FunctionWrapper(trials, StochQuad(noiseLevel=0.2))
    ls = lossTraces(fwrap=f, aclass=SGD, dim=trials, maxsteps=100, algoparams={'learning_rate':0.2})
    pylab.plot(ls, 'b:')
    pylab.plot(pylab.mean(ls, axis=1), 'r-')
    pylab.semilogy()
    pylab.show()


def testPlot2(trials=51, maxsteps=10000):
    f = FunctionWrapper(trials, StochQuad(noiseLevel=100, curvature=1))
    for aclass, aparams in [#(SGD, {'learning_rate':0.1}),
                            #(SGD, {'learning_rate':0.01}),
                            #(AdaGrad, {'init_lr':0.3}),
                            #(Amari, {'init_lr':0.1, 'time_const':100}),
                            #(RMSProp, {'init_lr':0.1}),
                            #(OracleSGD, {}),
                            (vSGD, {'verbose':False}),
                            #(vSGDfd, {}),
                            ]:
        ls = lossTraces(fwrap=f, aclass=aclass, dim=trials,
                        maxsteps=maxsteps, algoparams=aparams)
        plotWithPercentiles(ls, algo_colors[aclass], aclass.__name__)
    pylab.semilogy()
    pylab.xlim(0, maxsteps)
    pylab.legend()
    pylab.show()

def testPlot3(trials=100, maxsteps=2 ** 10):
    fwrap = FunctionWrapper(trials, StochQuad(noiseLevel=1, curvature=1))
    ploti = 1
    variants = [(SGD, {'learning_rate':0.1}),
                (SGD, {'learning_rate':0.01}),
                (AdaGrad, {'init_lr':0.3}),
                (Amari, {'init_lr':0.1, 'time_const':100}),
                (RMSProp, {'init_lr':0.1}),
                (OracleSGD, {}),
                ]
    ratio = 1
    tot = len(variants)
    rows = int(pylab.sqrt(tot) / ratio)     
    cols = (tot + rows - 1) / rows 
    for aclass, aparams in variants:
        pylab.subplot(rows, cols, ploti); ploti += 1
        plotHeatmap(fwrap, aclass, aparams, trials, maxsteps)
        pylab.title(aclass.__name__)        
    pylab.show()
    
def testPlot4(trials=40, maxsteps=512):
    fun = StochQuad(noiseLevel=100., curvature=1)
    fwrap = FunctionWrapper(trials, fun, record_samples=True)
    fwrap.nextSamples(100000)
    fwrap = DataFunctionWrapper(fwrap._seen, fun, shuffling=False)
    
    for i, (aclass, aparams) in enumerate([(vSGD, {'batch_size':1}),
                                           (vSGDfd, {'batch_size':1}),
                                           ]):
        pylab.subplot(2, 1, 2)
        fwrap.reset()
        ls = lossTraces(fwrap=fwrap, aclass=aclass, dim=trials,
                        maxsteps=maxsteps, algoparams=aparams)
        plotWithPercentiles(ls, algo_colors[aclass], aclass.__name__)
        pylab.semilogy()
        pylab.xlim(0, maxsteps)
        pylab.legend()
    
        pylab.subplot(2, 2, i + 1)
        fwrap.reset()
        plotHeatmap(fwrap, aclass, aparams, trials, maxsteps)
        
    pylab.show()
    
    
if __name__ == "__main__":
    #testPlot1()
    testPlot2()
    #testPlot3()
    #testPlot4()

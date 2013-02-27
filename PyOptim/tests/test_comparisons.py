import pylab
from pybrain.utilities import crossproduct
from tools.plotting import plotHeatmap
from core.datainterface import FunctionWrapper
from matplotlib import rc
rc('text', usetex=False)

from algorithms import SGD, AdaGrad, Amari, OracleSGD, RMSProp, vSGD, MomentumSGD, vSGDfd
from benchmarks.stoch_1d import StochQuad, StochAbs, StochRectLin, StochGauss


algo_variants = [(SGD, {'learning_rate':1}),
                 (SGD, {'learning_rate':0.1}),
                 (SGD, {'learning_rate':0.01}),
                 (SGD, {'learning_rate':0.001}),
                 (None, None),
                 (MomentumSGD, {'learning_rate':0.1, 'momentum':0.5}),
                 (MomentumSGD, {'learning_rate':0.01, 'momentum':0.5}),
                 (MomentumSGD, {'learning_rate':0.1, 'momentum':0.995}),
                 (MomentumSGD, {'learning_rate':0.01, 'momentum':0.995}),
                 (None, None),
                 (AdaGrad, {'init_lr':1}),
                 (AdaGrad, {'init_lr':0.1}),
                 (AdaGrad, {'init_lr':0.01}),
                 (None, None),
                 (Amari, {'init_lr':1, 'time_const':100}),
                 (Amari, {'init_lr':0.1, 'time_const':100}),
                 (Amari, {'init_lr':0.01, 'time_const':100}),
                 (None, None),
                 (RMSProp, {'init_lr':1}),
                 (RMSProp, {'init_lr':0.1}),
                 (RMSProp, {'init_lr':0.01}),
                 (None, None),
                 (vSGD, {}),
                 (vSGDfd, {}),
                 (OracleSGD, {}),
                 ] 

fun_settings = [{'curvature':0.1, 'noiseLevel':10},
                {'noiseLevel':10},
                {},
                {'curvature':0.1},
                {'curvature':10},
                {'noiseLevel':0.1},
                {'curvature':10, 'noiseLevel':0.1},
                None
                ]

fun_classes = [StochQuad, StochGauss,
               StochAbs, StochRectLin,
               ]

fun_variants = crossproduct([fun_classes, fun_settings])



def plotAllCombinations(avariants, fvariants, trials, maxsteps):
    fwrap = FunctionWrapper(trials, StochQuad(noiseLevel=1, curvature=1))
    ploti = 1
    rows = len(avariants)     
    cols = len(fvariants)
    for aid, (aclass, aparams) in enumerate(avariants):
        if aclass is None:
            ploti += cols
            continue
        for fid, (fclass, fsettings) in enumerate(fvariants):
            if fsettings is None:
                ploti += 1
                continue        
            fwrap = FunctionWrapper(trials, fclass(**fsettings))     
            pylab.subplot(rows, cols, ploti); ploti += 1
            plotHeatmap(fwrap, aclass, aparams, trials, maxsteps)
            if aid == 0:
                pylab.title(fclass.__name__[5:])
            if fid == 0:
                pylab.ylabel(aclass.__name__[:5])        
    
def test1():
    plotAllCombinations(algo_variants[-3:], fun_variants[:], 100, 2**10)
    pylab.show()


if __name__ == '__main__':
    test1()

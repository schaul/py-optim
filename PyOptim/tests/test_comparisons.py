import pylab
from scipy import median
from tools.plotting import plotHeatmap
from tools.experiments import lossTraces
from core.datainterface import FunctionWrapper, DatasetWrapper

from algorithms import SGD, AdaGrad, Amari, OracleSGD, RMSProp, vSGD, MomentumSGD, vSGDfd
from benchmarks.stoch_1d import StochQuad, StochAbs, StochRectLin, StochGauss


algo_variants = {SGD: [{'learning_rate':1},
                       {'learning_rate':0.1},
                       {'learning_rate':0.1, 'batch_size':10},
                       {'learning_rate':0.01},
                       {'learning_rate':0.01, 'batch_size':10},
                       {'learning_rate':0.001},
                       ],
                 MomentumSGD: [{'learning_rate':0.1, 'momentum':0.5},
                               {'learning_rate':0.01, 'momentum':0.5},
                               {'learning_rate':0.1, 'momentum':0.995},
                               {'learning_rate':0.01, 'momentum':0.995},
                               ],
                 AdaGrad: [{'init_lr':1},
                           {'init_lr':0.1},
                           {'init_lr':0.01},
                           ],
                 Amari: [{'init_lr':1, 'time_const':100},
                         {'init_lr':0.1, 'time_const':100},
                         {'init_lr':0.01, 'time_const':100},
                         ],
                 RMSProp : [{'init_lr':1},
                            {'init_lr':0.1},
                            {'init_lr':0.01},
                            ],
                 vSGD: [{},
                        {'batch_size':10},
                        ],
                 vSGDfd: [{},
                          {'batch_size':10},
                          ],
                 OracleSGD: [{}],
                 } 

fun_settings = [#{'noiseLevel':100},
                {'curvature':10, 'noiseLevel':10},
                {'noiseLevel':10},
                {'curvature':0.1, 'noiseLevel':10},
                {},
                #{'curvature':0.1},
                #{'curvature':10},
                {'curvature':0.01},
                {'curvature':100},
                {'noiseLevel':0.1},
                {'curvature':10, 'noiseLevel':0.1},
                #{'noiseLevel':0.01},
                ]

fun_classes = [StochQuad, StochGauss,
               StochAbs, StochRectLin,
               ]




def plotAllCombinations(aclasses, avariants,
                        fclasses, fvariants,
                        trials, maxsteps, maxbatchsize=10):
    fundic = {}    
    ploti = 1
    rows = sum([len(avariants[ac]) for ac in aclasses]) + len(aclasses) - 1
    cols = len(fvariants) * len(fclasses) + len(fclasses) - 1
    f_mid = int(median(range(len(fvariants))))
    for ac_id, aclass in enumerate(aclasses):
        a_mid = int(median(range(len(avariants[aclass]))))
        for as_id, aparams in enumerate(avariants[aclass]):
            if as_id == 0 and ac_id > 0:
                ploti += cols
            
            for fc_id, fclass in enumerate(fclasses):
                if fc_id not in fundic:
                    # shared samples across all uses of one function
                    fun = fclass()
                    fwrap = FunctionWrapper(trials, fun, record_samples=True)
                    fwrap.nextSamples(maxbatchsize * maxsteps)
                    fundic[fc_id] = fwrap._seen
                data = fundic[fc_id]
                for fs_id, fsettings in enumerate(fvariants):
                    if fs_id == 0 and fc_id > 0:
                        ploti += 1
                    fun = fclass(**fsettings)
                    provider = DatasetWrapper(data, fun, shuffling=False)            
                    pylab.subplot(rows, cols, ploti); ploti += 1
                    plotHeatmap(provider, aclass, aparams, trials, maxsteps)
                    if ac_id == 0 and as_id == 0 and fs_id == f_mid:
                        pylab.title(fclass.__name__[5:])
                    if fs_id == 0 and as_id == a_mid:
                        pylab.ylabel(aclass.__name__[:5])
    pylab.subplots_adjust(left=0.1, bottom=0.01, right=0.99, top=0.9, wspace=0.05, hspace=0.05)        
    
def test1():
    plotAllCombinations([#SGD, 
                         vSGD,
                         vSGDfd, 
                         #OracleSGD, 
                         #Amari,
                         ], algo_variants,
                        fun_classes[:],
                        fun_settings, 25, 2 ** 8)
    pylab.show()


def _runsome():
    trials = 50
    maxsteps = 2000
    fwrap = FunctionWrapper(trials, StochQuad(noiseLevel=1, curvature=1))
    for aclass in [vSGD]: 
        for aparams in algo_variants[aclass]:
            for fclass in fun_classes[:2]:
                for fsettings in fun_settings[:2]:
                    fwrap = FunctionWrapper(trials, fclass(**fsettings))     
                    lossTraces(fwrap, aclass, algoparams=aparams, dim=trials, maxsteps=maxsteps, storesteps=[10])

def testSpeed():
    from pybrain.tests.helpers import sortedProfiling
            
    sortedProfiling('_runsome()')
    
    

if __name__ == '__main__':
    test1()
    #testSpeed()

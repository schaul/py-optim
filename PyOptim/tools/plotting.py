import pylab
from scipy import median, log2, mean, exp, zeros,isnan, sqrt, log10, clip
from pylab import cm
    
from tools import percentile
from experiments import lossTraces
from algorithms import SGD, AdaGrad, Amari, MomentumSGD, OracleSGD, RMSProp, vSGD, vSGDfd, AveragingSGD, AveragingOracle, AdaptivelyAveragingOracle

from matplotlib import rc
rc('text', usetex=False)

algo_colors = {SGD: 'b',
               MomentumSGD: 'g',
               AdaGrad: 'r',
               Amari: 'm',
               OracleSGD: 'k',
               RMSProp: 'y',
               vSGD: 'g',
               vSGDfd: 'c',
               AveragingSGD: 'r',
               AveragingOracle: 'y', 
               AdaptivelyAveragingOracle: 'm',
               }


def plotWithPercentiles(ltraces, color, name=None, plotall=False):
    m = median(ltraces, axis=1)
    lp = percentile(ltraces, 25, axis=1)
    up = percentile(ltraces, 75, axis=1)
    if plotall:
        for l in ltraces.T:
            pylab.plot(l, color + '-', alpha=0.3)
    pylab.plot(m, color + '-', label=name)
    pylab.fill_between(range(len(m)), lp, up, facecolor=color, alpha=0.1)
    
    
def plotHeatmap(fwrap, aclass, algoparams, trials, maxsteps):
    """ Visualizing performance across trials and across time 
    (iterations in powers of 2) """
    psteps = int(log2(maxsteps)) + 1
    storesteps = [0] + [2 ** x  for x in range(psteps)]
    ls = lossTraces(fwrap, aclass, dim=trials, maxsteps=maxsteps,
                    storesteps=storesteps, algoparams=algoparams,
                    minLoss=1e-10)
            
    initv = mean(ls[0])
    maxgain = exp(fwrap.stochfun.maxLogGain(maxsteps) + 1)
    maxneggain = (sqrt(maxgain))
    
    M = zeros((psteps, trials))
    for sid in range(psteps):
        # skip the initial values
        winfactors = clip(initv / ls[sid+1], 1. / maxneggain, maxgain)
        winfactors[isnan(winfactors)] = 1. / maxneggain
        M[sid, :] = log10(sorted(winfactors))
        
    pylab.imshow(M.T, interpolation='nearest', cmap=cm.RdBu, #@UndefinedVariable
                 aspect=psteps / float(trials) / 1,  
                 vmin= -log10(maxgain), vmax=log10(maxgain),
                 )   
    pylab.xticks([])
    pylab.yticks([])
    return ls
        

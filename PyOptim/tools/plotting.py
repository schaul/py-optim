from scipy import median, percentile
from algorithms import SGD, AdaGrad, Amari, MomentumSGD, OracleSGD, RMSProp, vSGD
import pylab


algo_colors = {SGD: 'b',
               MomentumSGD: 'g',
               AdaGrad: 'r',
               Amari: 'm',
               OracleSGD: 'k',
               RMSProp: 'y',
               vSGD: 'g',
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
    #pylab.plot(up, color + '+:')
    #pylab.plot(lp, color + '+:')
from scipy import ones, randn, array, reshape, ravel, mean


def lossTraces(fwrap, aclass, dim, maxsteps, storesteps=None, x0=None,
               initNoise=0.01, minLoss=1e-10, algoparams={}):
    """ Compute a number of loss curves, for the provided settings,
    stored at specific storestep points. """
    if not storesteps:
        storesteps = range(maxsteps + 1)
    
    # initial points, potentially noisy
    if x0 is None:
        x0 = ones(dim) + randn(dim) * initNoise
    
    # tracking progress by callback
    paramtraces = {'index':-1}
    def storer(a):
        lastseen = paramtraces['index']
        for ts in [x for x in storesteps if x > lastseen and x <= a._num_updates]:
            paramtraces[ts] = a.bestParameters.copy()
        paramtraces['index'] = a._num_updates
        
    # initialization    
    algo = aclass(fwrap, x0, callback=storer, **algoparams)
    print algo, fwrap, dim    
    
    # store initial step   
    algo.callback(algo)
    algo.run(maxsteps)

    # process learning curve
    del paramtraces['index']
    paramtraces = array([x for _, x in sorted(paramtraces.items())])
    oloss = mean(fwrap.stochfun.expectedLoss(ones(100) * fwrap.stochfun.optimum))
    ls = abs(fwrap.stochfun.expectedLoss(ravel(paramtraces)) - oloss) + minLoss
    ls = reshape(ls, paramtraces.shape)
    return ls

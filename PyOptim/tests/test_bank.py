"""
Classification on the UCI Bank dataset.
"""


from pybrain.datasets import SupervisedDataSet
from scipy import mean
import cPickle as cp
import os

from core.datainterface import ModuleWrapper
from algorithms.sgd import SGD
from algorithms.vsgd import vSGDfd
    
datapath = '../../../../mydata/bank'
ds_file = 'temp/bank_data_tmp.pkl'

def quickdump(filename,object):
    f = open(filename,'wb')
    cp.dump(object,f,-1)
    f.close()

def quickload(filename):
    f = open(filename,'rb')
    object = cp.load(f)
    f.close()
    return object

def getAllFilesIn(dir, tag='', extension='.pkl'):
    """ return a list of all filenames in the specified directory
    (with the given tag and/or extension). """
    allfiles = os.listdir(dir)
    res = []
    for f in allfiles:
        if f[-len(extension):] == extension and f[:len(tag)] == tag:
            res.append(dir + '/' + f)#[:-len(extension)])
    return res

def readData():
    D = None
    try:
        D = quickload(ds_file)        
    except Exception, e:
        print 'Oh-oh', e        
    if D is None:
        numericcols = [0,5,9,11,12,13,14]
        bincols = [4,6,7,16]
        othercols = [i for i in range(17) if i not in numericcols+bincols]
        otherfeatures = []
        
        # open the data file for pre-analysis
        f = open(datapath+'/bank.csv', 'r').readlines()
        for line in f[1:]:
            tokens = line.strip().split(';')
            for i in othercols:
                key = (i, tokens[i].strip('"'))
                if not key in otherfeatures:
                    otherfeatures.append(key)
            
        otherfeatures.sort()
        
        # now fill the dataset
        D = SupervisedDataSet(len(otherfeatures)+len(bincols)+len(numericcols)-1, 1)
        f = open(datapath+'/bank.csv', 'r').readlines()
        for line in f[1:]:
            tokens = line.strip().split(';')
            fvals = [float(tokens[i]) for i in numericcols]
            bvals = [tokens[i].strip('"')[0]=='y' for i in bincols]
            otherfs = []
            for i, n in otherfeatures:
                if tokens[i].strip('"') == n:
                    otherfs.append(1)
                else:
                    otherfs.append(0)            
            D.addSample(fvals+bvals[:-1]+otherfs, bvals[-1])
        print len(D)
    else:
        print 'already found'            
    quickdump(ds_file, D)
    return D



def printy(s, force=False):
    if (force or 
        (s._num_updates * s.batch_size < 100 
         and s._num_updates % (20 / s.batch_size) == 0)
        or s._num_updates % (100 / s.batch_size) == 0):
        print s._num_updates * s.batch_size, #s.bestParameters, 
        s.provider.nextSamples(4521)
        print mean(s.provider.currentLosses(s.bestParameters))
        #s.provider.nextSamples(1)
        
        
def testBank():
    D = readData()    
    print len(D), 'samples', D.indim, 'features'
    from pybrain import LinearLayer, FullConnection, FeedForwardNetwork, BiasUnit, SigmoidLayer
    net = FeedForwardNetwork()
    net.addInputModule(LinearLayer(D.indim, name='in'))
    net.addModule(BiasUnit(name='bias'))
    net.addOutputModule(SigmoidLayer(1, name='out'))
    net.addConnection(FullConnection(net['in'], net['out']))
    net.addConnection(FullConnection(net['bias'], net['out']))
    net.sortModules()
    p = net.params
    p *= 0.01
    provider = ModuleWrapper(D, net, shuffling=False)
    algo = SGD(provider, net.params.copy(), #callback=printy, 
           learning_rate=5.5e-5)
    #algo = vSGDfd(provider, net.params.copy(), #callback=printy
    #              )
    printy(algo, force=True)
    algo.run(len(D))
    printy(algo, force=True)
    algo.run(len(D))
    printy(algo, force=True)
    algo.run(len(D))
    printy(algo, force=True)
    algo.run(len(D))
    printy(algo, force=True)
    algo.run(len(D))
    printy(algo, force=True)


if __name__ == "__main__":
    testBank()
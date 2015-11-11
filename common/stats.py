from __future__ import division
from pylab import *
from utils import DataLog
try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    imported_mpi = True
except ImportError:
    imported_mpi = False

import utils
utils.backup(__file__)


class AbstractStat(object):
    '''A Stat is meant to encapsulate a single bit of data stored in a 
    floating numpy array. All data that might need to be stored gets put 
    into a utils.Bunch object and passed between each stat. This allows 
    the calculation of more complicated stats which rely on several bits 
    of data.'''
    def __init__(self):
        self.name = None #<---Must be set to a valid variable name
        self.collection = "reduce"
        self.parent = None

    def connect(self,parent):
        self.parent = parent

    def start(self,c,obj):
        '''The intention of start is to be first call after object is 
        createdAlso this gets called once during lifetime of stats 
        object'''
        pass

    def clear(self, c, obj):
        '''Called whenever statistics should be reset'''
        pass

    def add(self, c, obj):
        '''Called after each simulation step'''
        pass

    def report(self, c, obj):
        '''To ensure things work on the cluster, please always return a
        Numpy *double* array!
        There is some flexibility in what you return, either return a
        single array and the name and collection method will be inferred
        from self.name and self.collection respectively. Or return a
        list of tuples and give name, collection and data explicitly.'''
        #Called after a block of training has occurred
        raise NotImplementedError("Bad user!")

class CounterStat(AbstractStat):
    '''A simple stat that counts the number of add() calls.'''
    def __init__(self,name='num_steps'):
        self.name = name
        self.collection = "reduce"

    def start(self,c,obj):
        c[self.name] = 0.0 # Everything needs to be a float :-/

    def add(self,c,obj):
        c[self.name] += 1.0

    def report(self,c,obj):
        return array(c[self.name]) # And an array :-/

def _getvar(obj,var):
    if '.' in var:
        (obj_name,_,var) = var.partition('.')
        obj = obj.__getattribute__(obj_name)
        return _getvar(obj,var)
    return obj.__getattribute__(var)

class HistoryStat(AbstractStat):
    '''A stat that monitors the value of a variable at every step of
    the simulation'''
    def __init__(self, var='x',collection="gather",record_every_nth=1):
        self.var = var
        self.name = var+"_history"
        self.counter = var+"_counter"
        self.collection = collection
        self.record_every_nth = record_every_nth

    def start(self,c,obj):
        if 'history' not in c:
            c.history = utils.Bunch()
        c.history[self.counter] = 0

    def clear(self, c, obj):
        c.history[self.name] = []

    def add(self,c,obj):
        if not (c.history[self.counter] % self.record_every_nth):
            tmp = _getvar(obj,self.var)
            if callable(tmp):
                tmp=tmp()
            c.history[self.name].append(np.copy(tmp))
        c.history[self.counter] += 1

    def report(self,c,obj):
        try:
            return array(c.history[self.name])
        except ValueError as v:
            print 'Error in stats.py', v, self.name
            #~ import pdb
            #~ pdb.set_trace()


class StatsCollection:
    def __init__(self,obj,dlog=None):
        '''The StatsCollection object holds many statistics objects and
        distributes the calls to them. It also simplifies the collection
        of information when report() and cluster_report() are called.'''
        self.obj = obj
        self.c = utils.Bunch()
        self.disable = False
        self.methods = []
        if dlog is None:
            self.dlog = DataLog()
        else:
            self.dlog = dlog

    def start(self):
        '''The start() method is called once per simulation'''
        for m in self.methods:
            m.connect(self)
            m.start(self.c,self.obj)
            m.clear(self.c,self.obj)

    def clear(self):
        '''The clear() method is called at the start of an epoch that
        will be monitored'''
        for m in self.methods:
            m.clear(self.c,self.obj)

    def add(self):
        if self.disable:
            return
        for m in self.methods:
            m.add(self.c,self.obj)

    def _report(self):
        '''report() is called at the end of an epoch and returns a list
        of results in the form:
         [(name,collection,value)]
        where:
          name = name of statistic
          collection = how to communicate statistic when on a cluster
          value = the value observed.
        '''
        l = []
        for m in self.methods:
            val  = m.report(self.c,self.obj)
            if isinstance(val, list):
                for (name,collection,v) in val:
                    if v.size == 0:
                        continue
                    l.append((name,collection,v))
            else:
                if val.size==0:
                    continue
                l.append((m.name,m.collection,val))
        return l

    def single_report(self):
        l = self._report()
        for (name,coll, val) in l:
            self.dlog.append(name,val)

    def cluster_report(self,cluster):
        '''Same intent as single_report(), but communicate data across 
        the cluster. The cluster variable that is passed needs to have
        the following attribute:
        cluster.NUMBER_OF_CORES
        '''
        rank = comm.rank
        #Same logic from report()
        l = self._report()
        #Now we do cluster communication
        #~ print 'cluster reporting for node ', rank
        for (name,coll, val) in l:
            if coll is "reduce":
                #~ print 'Node',rank,'reducing',name
                temp = comm.reduce(val)
                temp = temp/cluster.NUMBER_OF_CORES
                #~ print 'Node',rank,'reduced',name

            if coll is "gather":                
                if rank == 0:
                    temp = empty((comm.size,)+(prod(val.shape),))
                else:
                    temp = None
                # for debugging
                #~ print 'Node',rank,'gathering',name
                #~ print name
                #~ print shape(val)
                #~ print shape(val.flatten())
                comm.Gather(val.flatten(), temp, root=0)
                if rank == 0:
                    temp = [temp[i].reshape(val.shape) for i in \
                                                       range(comm.size)]
                #~ print 'Node',rank,'gathered',name

            if coll is "gatherv": #Variable gather size
                #~ print 'Node',rank,'gatherving',name
                arrsizes = empty( cluster.NUMBER_OF_CORES, dtype=int )
                arrsize  = array( prod(val.shape) )
                comm.Allgather(sendbuf=[arrsize, MPI.LONG],
                               recvbuf=[arrsizes,MPI.LONG])
                if comm.rank==0:
                    temp = zeros(sum(arrsizes))
                else:
                    temp = zeros(0)
                comm.Gatherv([val.flatten(),(arrsize, None),MPI.DOUBLE],
                             [temp,         (arrsizes,None),MPI.DOUBLE],
                             root=0)
                #~ print 'Node',rank,'gathervd',name

            if coll is "root":
                if rank == 0:
                    temp = val
                else:
                    temp = array([])
            
            self.dlog.append(name,temp)
            del temp #Delete temp to ensure failing if coll is unknown

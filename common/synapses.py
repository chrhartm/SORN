from __future__ import division
from pylab import *
import numpy as np
import scipy.sparse as sp
import scipy.stats as st

import utils
utils.backup(__file__)

# Helper function for structural plasticity
def _find_new(W,avoid_self_connections):
    (i_max,j_max) = W.shape
    counter = 0
    while True:
        i = np.random.randint(i_max)
        j = np.random.randint(j_max)
        connected = W[i,j] > 0
        valid = (not avoid_self_connections) or (i!=j)
        if valid and not connected:
            break
        if valid and (counter >= 100):
            print("Leaving find_new early")
            break
        counter += 1
    return (i,j)

def create_matrix(shape,c):
    ''' Expecting a c with the following fields:
        c.use_sparse = True
        c.lamb = 10 (or inf to get full connectivity)
        c.avoid_self_connections = True
        c.eta_stdp = 0.001 (or 0.0 to disable)
        c.eta_istdp = 0.001 (or 0.0 to disable)

        c.sp_prob = 0.1 (or 0 to disable)
        c.sp_initial = 0.001
    '''
    if c.has_key("use_2D") and c.use_2D == True:
        return SparseSynapticMatrix2D(shape,c)
    elif c.use_sparse:
        return SparseSynapticMatrix(shape,c)
    else:
        return FullSynapticMatrix(shape,c)
        
class AbstractSynapticMatrix(object):
    """
    Interface for synaptic connection matrices
    """
    def __init__(self,shpae,c):
        """
        Initializes the variables of the matrix.
        
        Parameters:
            shape: array
                row/col --> size(to), size(from)
            c: bunch
                parameters as stated previously
        """
        raise NotImplementedError
    def prune_weights(self):
        """
        Prunes weights that are 0
        """
        raise NotImplementedError
    def struct_p(self):
        """
        Performs structural plasticity on the matrix
        defined as random instertion of a new connection
        """
        raise NotImplementedError
    def stdp(self,from_old,from_new,to_old=None,to_new=None):
        """
        Performs Spike-Timing-Dependent-Plasticity
        
        Parameters:
            from_old: array
                The state vector of the projecting population from the
                last time step
            from_new: array
                The state vector of the projecting population from
                this time step
            to_old: array
                The old state vector of the receiving population
                Default: from_old
            to_new: array
                The new state vector of the receiving population
                Default: to_new
        """
        raise NotImplementedError
    def istdp(self,y_old,x):
        """
        Performs Inhibitory Spike-Timing-Dependent-Plasticity as defined
        in Zheng 2013
        
        Parameters:
            y_old: array
                The old state of the inhibitory population
            x: array
                The new state of the excitatory population
        """
        raise NotImplementedError
    def istdp_pos(self,y_old,x):
        """
        Performs only the positive part of istdp (see this for details).
        """
        raise NotImplementedError
    def ss(self, target=None):
        """
        Performs synaptic scaling defined as normalizing all incoming 
        weights to a sum of 1 for each unit
        
        Parameters:
            target: float
                The target value to be normalized to (if not 1)
        """
        raise NotImplementedError
    def __mul__(self,x):
        """
        Multiplication of this matrix with a vector
        
        Parameters:
            x: array
                The vector to multiply with
        Returns:
            The product self*x
        """
        raise NotImplementedError
    def get_synapses(self):
         """
         Returns a dense copy of the connection matrix
         """
         raise NotImplementedError
    def set_synapses(self,W_new):
        """
        Sets the synapses.
        
        Parameters:
            W_new: matrix
                The new dense connection matrix
        """
        raise NotImplementedError
    def sane_after_update(self):
        """
        Checks if the incomming connections to each neuron sum to 1 and
        if the weights are in the range (0,1) 
        """
        raise NotImplementedError
        

class FullSynapticMatrix(AbstractSynapticMatrix):
    """
    Dense connection matrix class for SORN synapses.
    """
 
    def __init__(self,shape,c): 

        self.c = c
        if not self.c.has_key('eta_ss'):
            self.c.eta_ss = 1
        # M is a mask with 1s for all existing connections
        if c.lamb >= shape[0]:
            self.M = np.ones(shape)==True # cast to boolean
        else:
            self.M = np.random.rand(*shape) < (c.lamb/shape[0])
            # Iteratively creates connection matrices until all neurons 
            # get some input
            # TODO while(True) just has to blow up at some point
            assert(c.lamb > 0)
            while(True):
                num = np.sum(np.sum(self.M,1)==0)
                self.M[np.sum(self.M,1)==0] = np.random.rand(num,
                                             shape[1])<(c.lamb/shape[0])
                if c.avoid_self_connections:
                    np.fill_diagonal(self.M,False)
                if np.all(np.sum(self.M,1)>0):
                    break                
        self.W = np.random.rand(*shape)
        self.W[~self.M] = 0.0
        ess_tmp = self.c.eta_ss
        self.c.eta_ss = 1
        if self.c.has_key('eta_ds') and self.c.eta_ds > 0:
            for _ in range(2*int(1./self.c.eta_ds)):
                self.ss()
        self.ss()
        self.c.eta_ss = ess_tmp
        self.stdp_dw = np.zeros(3)
        if self.c.has_key('p_failure') and self.c.p_failure>0:
            self.masked = self.get_synapses()

    def prune_weights(self):
        c = self.c
        if c.has_key('no_lower_bound') and c.no_lower_bound:
            return
        self.W[self.W<0.0] = 0.0
        if c.has_key('upper_bound'):
            self.W[self.W > c.upper_bound] = c.upper_bound
        if c.has_key('no_prune') and c.no_prune:
            return
        self.M[self.W<=0.0] = False

    def struct_p(self):
        c = self.c
        if c.has_key('sp_prob') and np.random.rand() < c.sp_prob:
            (i,j) = _find_new(self.W,c.avoid_self_connections)

            self.W[i,j] = c.sp_initial
            self.M[i,j] = True

    def stdp(self,from_old,from_new,to_old=None,to_new=None):
        c = self.c
        if not c.has_key('eta_stdp'):
            return

        if to_old is None:
            to_old = from_old
        if to_new is None:
            to_new = from_new
        
        dw = (to_new[:,None]*from_old[None,:]
                         -to_old[:,None]*from_new[None,:])
        
        #~ # Compare with Andreea's implementation --> works
        #~ A = np.dot(from_old,to_new.T)
        #~ aux = c.eta_stdp*(A.T-A)
        #~ print sum(dw-aux)
        
        # Uses M to only change connections that actually exist
        if c.has_key('weighted_stdp') and c.weighted_stdp:
            dw = dw.astype(float)
            dw[dw>0] = (c.upper_bound-self.W[dw>0])
            dw[dw<0] = -self.W[dw<0]
        if c.has_key('bias') and c.bias:
            dw = dw.astype(float)
            dw[dw>0] *= np.sqrt(c.bias)
            dw[dw<0] /= np.sqrt(c.bias)
        if self.c.has_key('p_failure') and self.c.p_failure>0:
            dw[dw>0] *= (self.oldmasked[dw>0]>0)
            dw[dw<0] *= (self.masked[dw<0]>0)
            
        self.W[self.M] += c.eta_stdp*dw[self.M]
        self.prune_weights()
        self.stdp_dw = np.array([sum(dw),sum(dw>0),sum(dw<0)])

    def istdp(self,y_old,x):
        c = self.c
        if not c.has_key('eta_istdp'):
            return
        self.W[self.M] += -c.eta_istdp*((1-(x[:,None]*(1+1.0/c.h_ip)))\
                                        *y_old[None,:])[self.M]
        # can't use W < 0 because W gets 0 often
        self.W[self.W<=0] = 0.001 
        self.W *= self.M
        self.W[self.W>1.0] = 1.0
        
    def istdp_pos(self,y_old,x):
        c = self.c
        if not c.has_key('eta_istdp'):
            return
        self.W[self.M] += c.eta_istdp*((1-x[:,None])\
                                       *y_old[None,:])[self.M]
         # can't use W < 0 because W get's 0 often
        self.W[self.W<=0] = 0.001
        self.W *= self.M
        self.W[self.W>1.0] = 1.0

    def ss(self,target=None):
        if target is None:
            target = abs(self.W).sum(1)
        if self.c.has_key('eta_ds') and self.c.eta_ds > 0:
            (rows,cols) = shape(self.W)
            z_pre = target
            z_pre[z_pre < 1e-6] = 1e-6
            z_post = abs(self.W).sum(0)*cols/float(rows)
            z_post[z_post < 1e-6] = 1e-6
            W_new = self.W/(0.5*z_pre[:,None]+0.5*z_post[None,:])
            self.W = (1-self.c.eta_ds)*self.W+self.c.eta_ds*W_new
        else:
            z = target
            z[z < 1e-6] = 1e-6
            #~ self.W /= z[:,None]
            if self.c.eta_ss < 1:
                # W->(1-eta)*W+eta*(W/z) simplified:
                self.W *= 1.-self.c.eta_ss*(1.-1./z[:,None])
            else:
                self.W /= z[:,None]

    def __mul__(self,x):
        if self.c.has_key('p_failure') and self.c.p_failure>0:
            p = self.c.p_failure
            self.oldmasked = self.masked
            self.masked = self.get_synapses()
            nonzero = sum(self.masked>0)
            self.masked[self.masked>0] *= np.random.choice([0,1],
                                                      nonzero,p=[p,1-p])
            return self.masked.dot(x)
        else:
            return self.W.dot(x)

    def get_synapses(self):
        return self.W.copy()

    def set_synapses(self,W_new,scale=True):
        self.W = W_new.copy()
        self.M = W_new>0
        if scale:
            self.ss()
            self.prune_weights()

    def sane_after_update(self):
        eps = 1e-6
        Z = self.W.sum(1)
        if self.c.has_key('eta_ds') and self.c.eta_ds > 0:
            # eta_ds implies that it only converges to scaled version
            Z[:] = 1
        if any(abs(Z-1.0)>eps):
            print shape(self.W)
            print self.W.sum(0)
            print self.W.sum(1)
            ind = abs(Z-1.0)>eps
            print find(ind)
            print ("Difference from 1:",Z[ind]-1.0)
            self.ss()
            Z = self.W.sum(1)
            print ("Difference after trying to fix it:",Z[ind]-1.0)

        assert np.all(self.W >= 0.0)
        assert np.all(self.W <= 1.0)

        return True

class SparseSynapticMatrix(AbstractSynapticMatrix): 
    """
    A sparse implementation of the connection matrix.
    This uses the CSC format.
    """
    def __init__(self, shape, c):
        
        self.c = c
        if not self.c.has_key('eta_ss'):
            self.c.eta_ss = 1
        (M,N) = shape
        # c.lamb := number of outgoing synapses
        if c.lamb > M:
            p = 1.0
        else:
            p = c.lamb/(M+1e-16)
        # p := probability of a connection being set. 
        # (for both incomming and outgoing)
        # random variable for number of incomming connections
        rv = st.binom(N,p)
        # sample number of incomming connections 
        ns = rv.rvs(M)     
        assert(p>0)
        while(True):
            num = np.sum(ns==0)
            ns[ns==0] = rv.rvs(num)
            if all(ns>0):
                break
        W_dok = sp.dok_matrix( shape, dtype=np.float)
        
        if c.avoid_self_connections:
            j_s = range(N-1)
            ns -= 1
            ns[ns<=0] = 1
        else:
            j_s = range(N)
            
        for i in range(M):
            data = np.random.rand(ns[i])
            data /= sum(data)+1e-10
            np.random.shuffle(j_s)
            for ind in range(ns[i]):
                j = j_s[ind]
                if c.avoid_self_connections:
                    j += (j>=i)
                W_dok[i,j] = data[ind]

        self.W = W_dok.tocsc()
        ess_tmp = self.c.eta_ss
        self.c.eta_ss = 1
        self.ss()
        self.c.eta_ss = ess_tmp
        self.stdp_dw = np.zeros(3)
        if self.c.has_key('p_failure') and self.c.p_failure>0:
            self.mask = np.ones(self.W.data.shape[0])
        # Used for optimizing structural plasticity
        self.struct_p_count = 0 
        self.struct_p_list = []
        
        if not self.sane_after_update():
            print "Not sane in init"

    def prune_weights(self):
        c = self.c
        if c.has_key('no_lower_bound') and c.no_lower_bound:
            return
        if c.has_key('upper_bound'):
            self.W.data[self.W.data > c.upper_bound] = c.upper_bound
        # Delete very small weights
        self.W.data[self.W.data<1e-10] = 0.0
        if c.has_key('no_prune') and c.no_prune:
            return
        if (self.c.has_key('p_failure') and self.c.p_failure>0 and
            any(self.W.data==0)):
            
            W_tmp = self.W.copy()
            self.mask[self.mask==0] = -1
            self.mask[W_tmp.data==0] = 0
            W_tmp.data = self.mask
            W_tmp.eliminate_zeros()
            self.mask = W_tmp.data
            self.mask[self.mask==-1] = 0
            
        self.W.eliminate_zeros()

    # Structural Plasticity
    def struct_p(self):
        c = self.c
        if c.has_key('sp_prob') and np.random.rand() < c.sp_prob:
            (i,j) = _find_new(self.W,c.avoid_self_connections)
            self.struct_p_count += 1
            self.struct_p_list.append( (i,j) )
        if self.struct_p_count>10:
            # Change sparse matrix to DOK-matrix in order to change
            # connections
            W_dok = self.W.todok()
            if self.c.has_key('p_failure') and self.c.p_failure>0:
                self.mask[self.mask==0] = -1
                W = self.W.copy()
                W.data = self.mask
                W_dokfail = W.todok()
            for (i,j) in self.struct_p_list:
                W_dok[i,j] = c.sp_initial
                if self.c.has_key('p_failure') and self.c.p_failure>0:
                    W_dokfail[i,j] = -1
                
            self.W = W_dok.tocsc()
            if self.c.has_key('p_failure') and self.c.p_failure>0:
                self.mask = W_dokfail.tocsc().data
                self.mask[self.mask==-1] = 0
            self.struct_p_count = 0
            self.struct_p_list = []

    def stdp(self,from_old,from_new,to_old=None,to_new=None):
        c = self.c
        if not c.has_key('eta_stdp'):
            return
        if to_old is None:
            to_old = from_old
        if to_new is None:
            to_new = from_new

        N = self.W.shape[1]
        col = np.repeat(np.arange(N),np.diff(self.W.indptr))
        row = self.W.indices
        data = self.W.data
        dw = (to_new[row]*from_old[col] - \
                to_old[row]*from_new[col]) # Suitable for CSC
        if c.has_key('weighted_stdp') and c.weighted_stdp:
            dw = dw.astype(float)
            dw[dw>0] = (c.upper_bound-data[dw>0])
            dw[dw<0] = -data[dw<0]
        if c.has_key('bias') and c.bias:
            dw = dw.astype(float)
            dw[dw>0] *= np.sqrt(c.bias)
            dw[dw<0] /= np.sqrt(c.bias)
        if self.c.has_key('p_failure') and self.c.p_failure>0:
            # Synapses that were not active at last step did not trigger
            # activity at this step -> no potentiation
            dw[dw>0] *= self.oldmask[dw>0]
            # Synapses that were not active at this step cannot have
            # triggered depression at this step
            dw[dw<0] *= self.mask[dw<0]
        data += c.eta_stdp*dw
        self.prune_weights()
        self.stdp_dw = np.array([sum(dw),sum(dw>0),sum(dw<0)])

    def istdp(self,y_old,x):
        c = self.c
        if not c.has_key('eta_istdp'):
            return

        N = self.W.shape[1]
        col = np.repeat(np.arange(N),np.diff(self.W.indptr))
        row = self.W.indices

        self.W.data -= c.eta_istdp*\
                       ((1-(x[col]*(1+1.0/c.h_ip)))*y_old[row])
        # can't use W < 0 because W gets 0 often
        self.W.data[self.W.data<=0] = 0.001 
        self.W.data[self.W.data>1.0] = 1.0
        
    def istdp_pos(self,y_old,x):
        c = self.c
        if not c.has_key('eta_istdp') or c.eta_istdp <= 0.0:
            return

        N = self.W.shape[1]
        col = np.repeat(np.arange(N),np.diff(self.W.indptr))
        row = self.W.indices
        
        self.W.data += c.eta_istdp*((1-x[col])*y_old[row])
        # can't use W < 0 because W gets 0 often
        self.W.data[self.W.data<=0] = 0.001 
        self.W.data[self.W.data>1.0] = 1.0

    def ss(self,target=None):
        if target is None:
            target = abs(self.W).sum(1)
        if self.c.has_key('eta_ds') and self.c.eta_ds > 0:
            (rows,cols) = shape(self.W)
            col = np.repeat(np.arange(cols),np.diff(self.W.indptr))
            row = self.W.indices
            z_pre = array(target).T[0]
            z_pre[z_pre < 1e-6] = 1e-6
            z_post = array(abs(self.W).sum(0)*cols/float(rows))[0]
            z_post[z_post < 1e-6] = 1e-6
            data_new = self.W.data/(0.5*z_pre[row]+0.5*z_post[col])
            self.W.data = (1-self.c.eta_ds)*self.W.data+self.c.eta_ds\
                                                               *data_new
        else:
            z = target
            data = self.W.data
            z[z < 1e-6] = 1e-6
            if self.c.eta_ss < 1:
                # W->(1-eta)*W+eta*(W/z) simplified:
                data *= 1.-self.c.eta_ss*(1.-1./np.array(
                                z[self.W.indices]).reshape(data.shape))
            else:
                data /= np.array(z[self.W.indices]).reshape(data.shape)

    def __mul__(self,x):
        if self.c.has_key('p_failure') and self.c.p_failure>0:
            p = self.c.p_failure
            self.oldmask = self.mask
            self.mask = np.random.choice([0,1],self.W.data.shape[0],
                                   p=[p,1.-p])
            tmpdata = self.W.data.copy()
            self.W.data *= self.mask
            toreturn = self.W * x
            self.W.data = tmpdata.copy()
            return toreturn
        else:
            return self.W * x

    def get_synapses(self):
        return self.W.todense()

    def set_synapses(self,W_new):
        self.W = sp.csc_matrix(W_new)
        self.prune_weights()

    def sane_after_update(self):
        assert np.all(self.W.data >= 0.0)
        assert np.all(self.W.data <= 1.0)
        eps = 1e-6
        # This makes sure that every neuron gets some input (for ss)
        Z = self.W.sum(1)
        if any(abs(Z-1.0)>eps) and not (self.c.has_key('eta_ds') and 
                                        self.c.eta_ds > 0):
            ind = abs(Z-1.0)>eps
            print shape(self.W)
            print np.where(ind)
            print ("Difference from 1:",Z[ind]-1.0)
            self.ss()
            Z = self.W.sum(1)
            print ("Difference after trying to fix it:",Z[ind]-1.0)
        # double scaling iterative process
        if self.c.has_key('eta_ds') and self.c.eta_ds > 0:
            return True
        else:
            return not any(abs(Z-1.0)>eps)
        
"""
Sheet Topology by Dan
"""
class SparseSynapticMatrix2D(SparseSynapticMatrix): 
    """
    2D 1/R-connectivity sparse matrix
    Topology only effects initial connections and SP for now 
    single population (ee) only
    """
    def __init__(self, shape, c):
        """
        new way
        """
        self.c = c
        (M,N) = shape
        #check connection probability
        if c.lamb >= N:
            print "overconnection requested; compensating"
            if c.avoid_self_connections:
                c.lamb = N-1
            else:
                c.lamb = N
        #generate position arrays and distance matrix
        self.X = c.A * np.random.rand(N)
        self.Y = c.A * np.random.rand(N)
       
        print "position list:"
        print self.X
        print self.Y
        
        self.R=np.zeros(shape)
        if c.has_key("periodic") and c.periodic == True:
            for i in xrange(N):
        		for j in xrange(N):     			
        			dx = self.X[i]-self.X[j]
        			dy = self.Y[i]-self.Y[j]
        			#modify according to periodicity
        			dhalf = c.A / 2
        			if dx >= dhalf:
        				dx = dx - c.A
        			if dx < -dhalf:
        				dx = dx + c.A
        			if dy >= dhalf:
        				dy = dy - c.A
        			if dy < -dhalf:
        				dy = dy + c.A      			
        			R_2=dx*dx + dy*dy
        			self.R[i,j] = np.sqrt(R_2)
        else:
            for i in xrange(N):
        		for j in xrange(N):
        			dx = self.X[i]-self.X[j]
        			dy = self.Y[i]-self.Y[j]
        			R_2=dx*dx + dy*dy
        			self.R[i,j] = np.sqrt(R_2)
        print "separation matrix:"
        print self.R
        
        self.P=np.zeros(shape)
        for i in xrange(N):
            for j in xrange(N):  # for distribution, try 1/2 gaussian w/ 0.5 @ 250 um
                p = np.exp(self.R[i,j]*self.R[i,j]*np.log(0.5)/(250*250))
                self.P[i,j] = p
        print "connection probability matrix:"
        print self.P
        
        
        #initialize dictionary of keys
        W_dok = sp.dok_matrix( shape, dtype=np.float)
        
        # generate roughly mean Lam connections per neuron if equidistant
        count = 0
        while count <= N * c.lamb:
            # get a random potential connection
            i = np.random.randint(0,N)
            j = np.random.randint(0,N)
            # check self-connection and unoccupied
            if c.avoid_self_connections:
                while i==j or W_dok[i,j] != 0:
                	i = np.random.randint(0,N)
                	j = np.random.randint(0,N)
            else:
            	while W_dok[i,j] != 0:
            		i = np.random.randint(0,N)
            		j = np.random.randint(0,N)
            # check probability, generate connection if good
            if self.P[i,j] > 0. and np.random.rand() < self.P[i,j]:
                W_dok[i,j] = np.random.rand()
				#print W_dok[i,j]
                count +=1
                print count
       
        #finish it up
        self.W = W_dok.tocsc()
        
        print "connections generated:"
        print self.get_synapses()
        
        self.ss()
        #Used for optimizing structural plasticity
        self.struct_p_count = 0
        self.struct_p_list = []
    
        
        """
        old way
  
        self.c = c
        (M,N) = shape
        if c.lamb > M:
            p = 1.0
        else:
            p = c.lamb/(M+1e-16)
        rv = st.binom(M,p)
        ns = rv.rvs(N)
        # Just a weak attempt to get connected neurons (try to minimize 
        # the probability that a neuron has zero connections)
        for a in range(100):
            num = np.sum(ns==0)
            ns[ns==0] = rv.rvs(num)
            if all(ns>0):
                break

        W_dok = sp.dok_matrix( shape, dtype=np.float)
        
        if c.avoid_self_connections:
            i_s = range(M-1)
            ns -= 1
            ns[ns<=0] = 1
        else:
            i_s = range(M)

        for j in range(N):
            data = np.random.rand(ns[j])
            data /= sum(data)+1e-10
            np.random.shuffle(i_s)
            for ind in range(ns[j]):
                i = i_s[ind]
                if c.avoid_self_connections:
                    i += (i>=j)
                W_dok[i,j] = data[ind]

        self.W = W_dok.tocsc()
        
        self.ss()
        #Used for optimizing structural plasticity
        self.struct_p_count = 0 
        self.struct_p_list = []
        """

    # Structural Plasticity
    def struct_p(self):
    
        """
        new way
        """
        c = self.c
        W_dok = self.W.todok()
        (N,M) = shape(W_dok)
        
        # should a new connection be generated?
        if c.has_key('sp_prob') and np.random.rand() < c.sp_prob:
            done = 0
            while done == 0:
            	# get a random potential connection
				i = np.random.randint(0,N)
				j = np.random.randint(0,N)
				# check self-connection and unoccupied
				if c.avoid_self_connections:
					while i==j or W_dok[i,j] != 0:
						i = np.random.randint(0,N)
						j = np.random.randint(0,N)
				else:
					while W_dok[i,j] != 0:
						i = np.random.randint(0,N)
						j = np.random.randint(0,N)
				# check probability, generate connection if good
				if self.P[i,j] > 0. and np.random.rand() < self.P[i,j]:
					self.struct_p_count += 1
					self.struct_p_list.append((i,j))
					done = 1
                    
    	if self.struct_p_count>10:
            # Change sparse matrix to DOK-matrix in order to change connections
    		for (i,j) in self.struct_p_list:
    			W_dok[i,j] = c.sp_initial
			self.W = W_dok.tocsc()
			self.struct_p_count = 0
			self.struct_p_list = []

        """
        old way
        
        c = self.c
        if c.has_key('sp_prob') and np.random.rand() < c.sp_prob:
            (i,j) = _find_new(self.W,c.avoid_self_connections)
            self.struct_p_count += 1
            self.struct_p_list.append( (i,j) )
        if self.struct_p_count>10:
            # Change sparse matrix to DOK-matrix in order to change
            # connections
            W_dok = self.W.todok()
            for (i,j) in self.struct_p_list:
                W_dok[i,j] = c.sp_initial
            self.W = W_dok.tocsc()
            self.struct_p_count = 0
            self.struct_p_list = []
        """

#~ #Code relating to csr matices:
#~ ss:
    #~ z = W.sum(1)
    #~ z[z < 1e-6] = 1e-6
    #~ # Suitable for CSR normalization
    #~ W.data /= z.repeat(np.diff(W.indptr)).flat 
    #~ return W
#~ stdp:
    #~ row = repeat(arange(N),diff(W.indptr))
    #~ N = W.shape[0]
    #~ col = W.indices
    #~ W.data += c.eta_stdp*(to_new[row]*from_old[col] \
    #~                     - to_old[row]*from_new[col])
    #~ W.data[W.data<0.0] = 0.0
    #~ W.data[W.data>1.0] = 1.0
    #~ return W

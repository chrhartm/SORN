import utils
utils.backup(__file__)

import pylab as pl
import numpy.testing as tests
import itertools

class OptimalChannels(object):
    def __init__(self,p_uA_given_A=0.8,p_uA_given_B=0.1,
                 p_uB_given_A=None,p_uB_given_B=None,N_u=10):
        self.p_uA_given_A = p_uA_given_A
        self.p_uA_given_B = p_uA_given_B
        if p_uB_given_A is None:
            self.p_uB_given_A = p_uA_given_B
        else:
            self.p_uB_given_A = p_uB_given_A
        if p_uB_given_B is None:
            self.p_uB_given_B = p_uA_given_A
        else:
            self.p_uB_given_B = p_uB_given_B
        self.N_u = N_u

    def _p_A_given_N_A(self,N_A,p_A,N_B=None):
        if N_B == None:
            N_B = self.N_u-N_A
            
        p_B = 1-p_A
        p_N_A_given_A = (self.p_uA_given_A**(N_A) 
                            * (1-self.p_uA_given_A)**(self.N_u-N_A) 
                            * self.p_uB_given_A**(N_B) 
                            * (1-self.p_uB_given_A)**(self.N_u-N_B))
        p_N_A_given_B = (self.p_uA_given_B**(N_A) 
                            * (1-self.p_uA_given_B)**(self.N_u-N_A) 
                            * self.p_uB_given_B**(N_B) 
                            * (1-self.p_uB_given_B)**(self.N_u-N_B))
        return p_N_A_given_A*p_A / ((p_N_A_given_A*p_A)
                                     +(p_N_A_given_B*p_B))
    
    # Speeds shit up by 100x
    def _sample_posteriors_noloop(self,true_N_A,p_A,N_samples):
        true_N_B = self.N_u-true_N_A
        N_values = pl.shape(true_N_A)[0]
        posteriors = pl.zeros((N_samples,N_values))
        for (i,(t_N_A,t_N_B)) in enumerate(zip(true_N_A,true_N_B)):
            A_probs = pl.ones((N_samples,self.N_u))
            A_probs[:,:t_N_A] *= self.p_uA_given_A
            A_probs[:,t_N_A:] *= self.p_uA_given_B
            B_probs = pl.ones((N_samples,self.N_u))
            B_probs[:,:t_N_A] *= self.p_uB_given_A
            B_probs[:,t_N_A:] *= self.p_uB_given_B
            
            N_A = pl.sum(A_probs>pl.rand(N_samples,self.N_u),1)
            N_B = pl.sum(B_probs>pl.rand(N_samples,self.N_u),1)
            
            posteriors[:,i] = self._p_A_given_N_A(N_A,p_A,N_B)
        return pl.mean(posteriors,0)

    
    def _sample_posteriors(self,true_N_A,p_A,N_samples):
        true_N_B = self.N_u-true_N_A  
        N_values = pl.shape(true_N_A)[0]
        posteriors = pl.zeros((N_samples,N_values))
        for i in range(N_samples):
            for (j,(t_N_A,t_N_B)) in enumerate(zip(true_N_A,true_N_B)):
                A_given_A = pl.ones(t_N_A)*self.p_uA_given_A
                A_given_B = pl.ones(t_N_B)*self.p_uA_given_B
                A_probs = pl.hstack((A_given_A,A_given_B))
                B_given_A = pl.ones(t_N_A)*self.p_uB_given_A
                B_given_B = pl.ones(t_N_B)*self.p_uB_given_B
                B_probs = pl.hstack((B_given_A,B_given_B))

                N_A = pl.sum(A_probs>pl.rand(self.N_u))
                N_B = pl.sum(B_probs>pl.rand(self.N_u))

                posteriors[i,j] = self._p_A_given_N_A(N_A,N_B)
         
        return pl.mean(posteriors,0)
        
    def _exact_posteriors(self,true_N_A,p_A):
        from scipy.stats import binom
        true_N_B = self.N_u-true_N_A
        N_values = pl.shape(true_N_A)
        posteriors = pl.zeros((N_values))
        # Get all ints that sum to a fixed value (e.g. 1: [(1,0),(0,1)])
        combinations = {}
        N_u = int(self.N_u)
        for i in range(N_u+1):
            for j in range(N_u+1):
                try:
                    combinations[i+j] += [(i,j)]
                except KeyError:
                    combinations[i+j] = [(i,j)]
        # For each ambiguity, calculate the probabilities of active
        # channels
        for (i,(t_N_A,t_N_B)) in enumerate(zip(true_N_A,true_N_B)):
            # First calculate the probabilities of drawing specific
            # N_As from both binomials that contribute to N_A
            N_us = pl.arange(0,N_u+1)
            p_N_us_A1 = binom.pmf(N_us,t_N_A,self.p_uA_given_A)
            p_N_us_A2 = binom.pmf(N_us,t_N_B,self.p_uA_given_B)
            p_N_us_B1 = binom.pmf(N_us,t_N_A,self.p_uB_given_A)
            p_N_us_B2 = binom.pmf(N_us,t_N_B,self.p_uB_given_B)
            ps_N_A = pl.zeros(N_u+1)
            ps_N_B = pl.zeros(N_u+1)
            # Then combine the binomials with the combinations
            for j in range(N_u+1):
                for (k,l) in combinations[j]:
                    ps_N_A[j] += p_N_us_A1[k]*p_N_us_A2[l]
                    ps_N_B[j] += p_N_us_B1[k]*p_N_us_B2[l]
            # Some safety checks that probabilities actually sum to 1
            tests.assert_almost_equal(sum(ps_N_A),1.)
            tests.assert_almost_equal(sum(ps_N_B),1.)
            # Combine probabilities to get posterior
            for (N_A,N_B) in itertools.product(range(N_u+1),
                                               range(N_u+1)):
                p_A_given_N_AB = self._p_A_given_N_A(N_A,p_A,N_B)
                posteriors[i] += p_A_given_N_AB*ps_N_A[N_A]*ps_N_B[N_B]
        return posteriors
        
        
    def optimal_inference(self,p_A=0.3,N_As=None,N_samples=1000):
        if N_As is None:
            N_As = pl.linspace(0,self.N_u,self.N_u+1)
        #posteriors = self._sample_posteriors_noloop(N_As,p_A,N_samples)
        posteriors = self._exact_posteriors(N_As,p_A)
        return posteriors

    def plot(self,p_A=0.3,N_As=None):
        if N_As is None:
            pl.linspace(0,self.N_u,self.N_u+1)
        posteriors = sample_posteriors_noloop(N_As,p_A,1000)
        pl.figure()
        x = N_As/N_u
        pl.plot(x,pl.mean(posteriors,0),label='A')
        pl.plot(x,1-pl.mean(posteriors,0),label='B')
        #plot([0,1],[1-p_A,1-p_A],'k--',label='1 - p(A) and p(A)')
        #plot([0,1],[p_A,p_A],'k--')
        pl.plot([0.5,0.5],[0,1],'k-.',label='full ambiguity')
        pl.legend(loc='best')
        pl.title('p(on|correct)=%.1f   p(on|incorrect)=%.1f'
                 %(self.p_uA_given_A,self.p_uA_given_B))
        pl.xlabel('Fraction of A in ambiguous stimulus')
        pl.ylabel('Posterior probability')
    

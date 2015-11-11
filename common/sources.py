from __future__ import division
from pylab import *
import random
# there already is a sample in the namespace from numpy
from random import sample as smpl
import itertools
import utils
utils.backup(__file__)
import synapses

class AbstractSource(object):
    def __init__(self):
        """
        Initialize all relevant variables.
        """
        raise NotImplementedError
    def next(self):
        """
        Returns the next input
        """
        raise NotImplementedError
    def global_range(self):
        """
        Returns the maximal global index of all inputs
        """
        raise NotImplementedError
    def global_index(self):
        """
        Returns the current global (unique) index of the current input
        "character"
        """
        raise NotImplementedError
    def generate_connection_e(self,N_e):
        """
        Generates connection matrix W_eu from input to the excitatory
        population
        
        Parameters:
            N_e: int
                Number of excitatory units
        """
        raise NotImplementedError
    def generate_connection_i(self,N_e):
        """
        Generates connection matrix W_iu from input to the inhibitory
        population
        
        Parameters:
            N_e: int
                Number of excitatory units
        """
        raise NotImplementedError
    def update_W_eu(self,W_eu):
        """
        Modifies W_eu if necessary
        Called every step in SORN after next() was called but before 
        W_eu is used
        """
        pass


class CountingSource(AbstractSource):
    """
    Source for the counting task.
    Different of words are presented with individual probabilities.
    """
    def __init__(self, words,probs, N_u_e, N_u_i, avoid=False,
                permute_ambiguous=False):
        """
        Initializes variables.
        
        Parameters:
            words: list
                The words to present
            probs: matrix
                The probabilities of transitioning between word i and j
                It is assumed that they are summing to 1
            N_u: int
                Number of active units per step
            avoid: bool
                Avoid same excitatory units for different words
        """
        self.word_index = 0  #Index for word
        self.ind = 0         #Index within word
        self.words = words   #different words
        self.probs = probs   #Probability of transitioning from i to j
        self.N_u_e = int(N_u_e)  #Number active per step
        self.N_u_i = int(N_u_i)
        self.avoid = avoid
        self.permute_ambiguous = permute_ambiguous
        self.alphabet = unique("".join(words))
        self.N_a = len(self.alphabet)
        self.lookup = dict(zip(self.alphabet,range(self.N_a)))
        self.glob_ind = [0]
        self.glob_ind.extend(cumsum(map(len,words)))
        self.predict = self.predictability()
        self.reset()
    
    @classmethod
    def init_simple(cls, N_words, N_letters, word_length, max_fold_prob,
                    N_u_e, N_u_i, avoiding, words=None, seed=None):
        """
        Construct the arguments for the source to make it usable for the
        cluster
        
        Parameters:
            N_words: int
                Number of different words
            N_letters: int
                Number of letters to generate words from
            word_length: list
                Range of length (unimodal distribution)
            max_fold_prob: float
                maximal probability difference between words
            N_u_e: int
                Number of active excitatory units per step
            N_u_i: int
                Number of active inhibitory units per step
            avoid: bool
                Avoid same excitatory units for different words
        """
        newseed = np.random.randint(0,4000000000)
        if seed is not None:
            np.random.seed(seed=seed)
        letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
        assert(N_letters <= len(letters))
        letters = array([x for x in letters[:N_letters]])
        
        if words is None:
            words = []
            
            for i in range(N_words):
                word = letters[np.random.randint(0,N_letters,
                                np.random.randint(
                                    word_length[0], word_length[1]+1))]
                words.append(''.join(word))
        else:
            assert(N_words == len(words) and 
                   N_letters == len(unique(''.join(words))))
            
        probs = array([np.random.rand(N_words)*(max_fold_prob-1)+1]
                      *N_words)

        # Normalize
        probs /= sum(probs,1)
        
        if seed is not None:
            np.random.seed(seed=newseed)
             
        return CountingSource(words, probs, N_u_e, N_u_i,avoid=avoiding)

    def generate_connection_e(self,N_e):
        W = zeros((N_e,self.N_a))

        available = set(range(N_e))
        for a in range(self.N_a):
            temp = random.sample(available,self.N_u_e)
            W[temp,a] = 1
            if self.avoid:
                available = available.difference(temp)
                
        # The underscore has the special property that it doesn't 
        # activate anything:
        if '_' in self.lookup:
            W[:,self.lookup['_']] = 0

        c = utils.Bunch(use_sparse=False,
                        lamb=np.inf,
                        avoid_self_connections=False)
        ans = synapses.create_matrix((N_e,self.N_a),c)
        ans.W = W

        return ans
        
    def generate_connection_i(self,N_i):
        c = utils.Bunch(use_sparse=False,
                lamb=np.inf,
                avoid_self_connections=False)
        ans = synapses.create_matrix((N_i,self.N_a),c)
        W = zeros((N_i, self.N_a))
        if N_i>0:
            available = set(range(N_i))
            for a in range(self.N_a):
                temp = random.sample(available,self.N_u_i)
                W[temp,a] = 1
                #~ if self.avoid: # N_i is smaller -> broad inhibition?
                    #~ available = available.difference(temp)
            if '_' in self.lookup:
                W[:,self.lookup['_']] = 0
        ans.W = W

        return ans           
            
    def char(self):
        word = self.words[self.word_index]
        return word[self.ind]

    def index(self):
        character = self.char()
        ind = self.lookup[character]
        return ind

    def next_word(self):
        self.ind = 0
        w = self.word_index
        p = self.probs[w,:]
        self.word_index = find(rand()<=cumsum(p))[0]

    def next(self):
        self.ind = self.ind+1
        string = self.words[self.word_index]
        if self.ind >= len(string):
            self.next_word()
        ans = zeros(self.N_a)
        ans[self.index()] = 1
        return ans
        
    def reset(self):
        self.next_word()
        self.ind = -1

    def global_index(self):
        return self.glob_ind[self.word_index]+self.ind

    def global_range(self):
        return self.glob_ind[-1]
        
    def trial_finished(self):
        return self.ind+1 >= len(self.words[self.word_index])
        
    def update_W_eu(self,W_eu):
        if not self.permute_ambiguous:
            return
        # Init
        if not hasattr(self,'A_neurons'):
            self.A_neurons = where(W_eu.W[:,self.lookup['A']]==1)[0]
            self.B_neurons = where(W_eu.W[:,self.lookup['B']]==1)[0]
            self.N_u = int(len(self.A_neurons))
            assert(self.N_u == len(self.B_neurons))
            self.N_A = {}
            self.N_B = {}
            for letter in [word[0] for word in self.words]:
                letter_index = self.lookup[letter]
                self.N_A[letter] = sum(W_eu.W[self.A_neurons,letter_index]).round().astype(int)
                self.N_B[letter] = sum(W_eu.W[self.B_neurons,letter_index]).round().astype(int)
        
        # When new word presented, permute its input
        # careful! simple permutation goes crazy when inputs overlap
        if self.ind == 0: 
            letter = self.char()
            if letter in ['A','B']:
                return
            letter_index = self.index()
            # First set both to zero so that the second doesn't set 
            # units of the first back to zero (overlap case)
            W_eu.W[self.A_neurons,letter_index] *= 0
            W_eu.W[self.B_neurons,letter_index] *= 0
            W_eu.W[self.A_neurons[smpl(xrange(self.N_u),self.N_A[letter])],
                   letter_index] = 1
            W_eu.W[self.B_neurons[smpl(xrange(self.N_u),self.N_B[letter])],
                   letter_index] = 1

    def predictability(self):
        temp = self.probs
        for n in range(10):
            temp = temp.dot(temp)
        final = temp[0,:]
        #Let's assume that all words have unique initial letters
        probs = map(len, self.words)
        probs = array(probs)
        probs = (probs + self.probs.max(1)-1)/probs
        return sum(final*probs)

class RandomLetterSource(CountingSource):
    def __init__(self, N_letters, N_u_e, N_u_i, avoid=False):        
        letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
        assert(N_letters <= len(letters))
        words = [x for x in letters[:N_letters]]
        
        probs = ones((N_letters,N_letters))
        probs /= sum(probs,1)
 
        super(RandomLetterSource, self).__init__(words, probs, N_u_e, 
                                                 N_u_i,avoid=avoid)
        
class TrialSource(AbstractSource):
    """
    This source takes any other source and gives it a trial-like
    structure with blank periods inbetween stimulation periods
    The source has to implement a trial_finished method that is True
    if it is at the end of one trial
    """
    def __init__(self,source,blank_min_length,blank_var_length,
                 defaultstim,resetter=None):
        assert(hasattr(source,'trial_finished'))
        self.source = source
        self.blank_min_length = blank_min_length
        self.blank_var_length = blank_var_length
        self.reset_blank_length()
        self.defaultstim = defaultstim
        self.resetter = resetter
        self._reset_source()
        self.blank_step = 0
        
    def reset_blank_length(self):
        if self.blank_var_length > 0:
            self.blank_length = self.blank_min_length\
                                + randint(self.blank_var_length)
        else:
            self.blank_length = self.blank_min_length
        
    def next(self):
        if not self.source.trial_finished():
            return self.source.next()
        else:
            if self.blank_step >= self.blank_length:
                self.blank_step = 0
                self._reset_source()
                self.reset_blank_length()
                return self.source.next()
            else:
                self.blank_step += 1
                return self.defaultstim
            

    def _reset_source(self):
        if self.resetter is not None:
            getattr(self.source,self.resetter)()
            
    def global_range(self):
        return source.global_range()
        
    def global_index(self):
        if self.blank_step > 0:
            return -1
        return self.source.global_index()
        
    def generate_connection_e(self, N_e):
        return self.source.generate_connection_e(N_e)
        
    def generate_connection_i(self, N_i):
        return self.source.generate_connection_i(N_i)
        
    def update_W_eu(self,W_eu):
        self.source.update_W_eu(W_eu)
        
class AndreeaCountingSource(AbstractSource):
    """
    This was only for debugging purposes - it resembles her matlab code
    perfectly
    """
    def __init__(self,sequence,sequence_U,pop,train):
        self.pop = pop-1
        self.seq = sequence[0]-1
        self.seq_u = sequence_U[0]-1
        self.t = -1
        # change m,n,x to make them identical

        
        if train:
            self.seq[self.seq==2] = 80
            self.seq[self.seq==3] = 2
            self.seq[self.seq==4] = 3
            self.seq[self.seq==80] = 4
            self.seq_u[self.seq_u>13] = (self.seq_u[self.seq_u>13]%7)+7
            self.lookup = {'A':0,'B':1,'M':2,'N':3,'X':4}
        else:
            self.seq[self.seq==2] = 7
            self.seq[self.seq==9] = 2
            self.seq[self.seq==3] = 5
            self.seq[self.seq==10] = 3
            self.seq[self.seq==4] = 6
            self.seq[self.seq==11] = 4
            self.lookup = {'A':0,'B':1,'X':7,'M':5,'N':6,'C':2,'D':3,
                           'E':4}

            #~ self.lookup = {'A':0,'B':1,'X':2,'M':3,'N':4,'C':9,
                                #~ 'D':10,'E':11}
            
            self.alphabet = 'ABCDEMNX'
            self.words = ['AXXXXXM','BXXXXXN','CXXXXXN','CXXXXXM',
                                'DXXXXXN','DXXXXXM','EXXXXXN','EXXXXXM']
            self.glob_ind = [0]
            self.glob_ind.extend(cumsum(map(len,self.words))) 
        self.N_a = self.seq.max()+1
        
    def next(self):
        self.t += 1
        tmp = zeros((self.N_a)) 
        tmp[self.seq[self.t]] = 1
        return tmp
    def global_range(self):
        return self.seq_u.max()
    def global_index(self):
        return self.seq_u[self.t]
    def generate_connection(self,N_e):
        W = np.zeros((N_e,self.N_a))
        for i in range(self.N_a):
            if i <= 4: #--> A,B,X,M,N
                W[self.pop[:,i],i] = 1
            if i == 9:
                W[self.pop[0:2,0],i] = 1
                W[self.pop[2:10,1],i] = 1
            if i == 10:
                W[self.pop[0:5,0],i] = 1
                W[self.pop[5:10,1],i] = 1
            if i == 11:
                W[self.pop[0:8,0],i] = 1
                W[self.pop[8:10,1],i]  = 1
        self.W = W
        return W

class NoSource(AbstractSource):
    """
    No input for the spontaneous conditions
    
    Parameters:
        N_u: int
            Number of input units
    """
    def __init__(self,N_u=1):
        self.N_u = N_u
    def next(self):
        return np.zeros((self.N_u))

    def global_range(self):
        return 1

    def global_index(self):
        return -1

    def generate_connection_e(self,N_e):
        c = utils.Bunch(use_sparse=False,
                        lamb=np.inf, # cannot be 0
                        avoid_self_connections=False)
        tmpsyn = synapses.create_matrix((N_e,self.N_u),c)
        tmpsyn.set_synapses(tmpsyn.get_synapses()*0)
        return tmpsyn
        
    def generate_connection_i(self,N_i):
        c = utils.Bunch(use_sparse=False,
                        lamb=np.inf, # cannot be 0
                        avoid_self_connections=False)
        tmpsyn = synapses.create_matrix((N_i,self.N_u),c)
        tmpsyn.set_synapses(tmpsyn.get_synapses()*0)
        return tmpsyn
        
class RandomSource(AbstractSource):
    """
    Poisson input spike trains.
    """
    def __init__(self, firing_rate, N_neurons, connection_density,
                                                              eta_stdp):
        """
        Initialize the source
        
        Parameters:
            firing_rate: double
                The firing rate of all input neurons
            N_neurons: int
                The number of poisson input units
            connection_density: double
                Density of connections from input to excitatory pop
            eta_stdp: double
                STDP rate for the W_eu matrix                
        """
        self.rate = firing_rate
        self.N = N_neurons
        self.density = connection_density
        self.eta_stdp = eta_stdp
    def next(self):
        return rand(self.N)<=self.rate
    def global_range(self):
        return 1
    def global_index(self):
        return 0
    def generate_connection(self,N_e):
        c = utils.Bunch(use_sparse=False,
                        lamb=self.density*N_e,
                        avoid_self_connections=False,
                        #CHANGE should this be different?
                        eta_stdp = self.eta_stdp)
        tmp = synapses.create_matrix((N_e,self.N),c)
        # get correct connection density
        noone = True
        while(noone):
            tmp.set_synapses((rand(N_e,self.N)<self.density).astype(
                                                                 float))
            if sum(tmp.get_synapses()) > 0:
                noone = False                
        return tmp

#Useful for turning an index into a direction
mapping = {0:(+1,+0),
           1:(-1,+0),
           2:(+0,+1),
           3:(+0,-1),
           4:(+1,+1),
           5:(+1,-1),
           6:(-1,+1),
           7:(-1,-1)}

class DriftingSource(AbstractSource):
    """
    One of Philip's sources. Drifting objects in different directions
    """
    def __init__(self,c):
        self.X = c.X  #num horizontal pixels
        self.Y = c.Y  #num vertical pixels
        #number of directions of possible movement
        self.num_direction = c.num_direction  
        self.num_steps = c.num_steps
        self.choose_each_step = c.choose_each_step
        self.symbol = c.symbol # Number of bits for each symbol
        #~ self.N = c.N #number of neurons
        self.change_prob = c.change_prob #probability of changing dir

        self.direction = random.randrange(0,self.num_direction)
        self.all = list(itertools.product(range(c.X),range(c.Y)))
        self.active = {}

    def get_new_symbols(self,next):
        everything = set(self.all)
        active = set(next)
        remain = list(everything.difference(active))
        random.shuffle(remain)
        activated = remain[:self.choose_each_step]
        return activated

    def next(self):
        if random.random() < self.change_prob:
            #random.randint has different endpoint behaviour from 
            # np.random.randint!!
            self.direction = random.randrange(0,self.num_direction)
        temp = {}
        (dx,dy) = mapping[self.direction]
        for (x,y) in self.active.keys():
            count = self.active[(x,y)] - 1
            if count <= 0:
                continue
            nx = (x+dx)%self.X
            ny = (y+dy)%self.Y
            temp[(nx,ny)] = count
        activated = self.get_new_symbols(temp.keys())
        for k in activated:
            temp[k] = self.num_steps
        self.active = temp

        # Now convert that into representation
        ans = np.zeros((self.X,self.Y))
        for (x,y) in self.active.keys():
            ans[x,y] = 1
        ans.shape = self.X*self.Y
        return ans

    def generate_connection(self,N):
        W = np.zeros((N,self.X,self.Y))
        available = set(range(N))
        for a in range(self.X):
            for b in range(self.Y):
                temp = random.sample(available,self.symbol)
                W[temp,a,b] = 1
                available = available.difference(temp)
        W.shape = (N,self.X*self.Y)
        c = utils.Bunch(use_sparse=False,
                        lamb=np.inf,
                        avoid_self_connections=False)
        ans = synapses.create_matrix((N,self.X*self.Y),c)
        ans.W = W
        return ans

    def global_range(self):
        return self.num_direction

    def global_index(self):
        return self.direction

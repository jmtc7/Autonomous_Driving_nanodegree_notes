import numpy as np
import random
from math import sqrt, pi, exp

def gaussian_prob(obs, mu, sig):
    num = (obs - mu)**2
    denum = 2*sig**2
    norm = 1 / sqrt(2*pi*sig**2)
    return norm * exp(-num/denum)

class GNB():
    def __init__(self):
        self.classes = ['left', 'keep', 'right']

    def process_vars(self,vars):
        # could do something fancy in here, but right now
        # s, d, s_dot and d_dot alone give good results
        s, d, s_dot, d_dot = vars
        return s, d, s_dot, d_dot
        
    def train(self, X, Y):
        """
        X is an array of training data, each entry of which is a 
        length 4 array which represents a snapshot of a vehicle's
        s, d, s_dot, and d_dot coordinates.

        Y is an array of labels, each of which is either 'left', 'keep',
        or 'right'. These labels indicate what maneuver the vehicle was 
        engaged in during the corresponding training data snapshot. 
        """

        num_vars = 4

        # initialize an empty array of arrays. For this problem
        # we are looking at three labels and keeping track of 4 
        # variables for each (s,d,s_dot,d_dot), so the empty array
        # totals_by_label will look like this:

        # {
        #   "left" :[ [],[],[],[] ], 
        #   "keep" :[ [],[],[],[] ], 
        #   "right":[ [],[],[],[] ]  
        # }

        totals_by_label = {
            "left" : [], 
            "keep" : [],
            "right": [],
        }
        for label in self.classes:
            for i in range(num_vars):
                totals_by_label[label].append([])

        for x, label in zip(X,Y):

            # process the raw s,d,s_dot,d_dot snapshot if desired.
            x = self.process_vars(x)

            # add this data into the appropriate place in the 
            # totals_by_label data structure.
            for i,val in enumerate(x):
                totals_by_label[label][i].append(val)
        
        # Get the mean and standard deviation for each of the arrays
        # we've built up. These will be used as our priors in GNB
        means = []
        stds = []
        for i in self.classes:
            means.append([])
            stds.append([])
            for arr in totals_by_label[i]:
                mean = np.mean(arr)
                std = np.std(arr)
                means[-1].append(mean)
                stds[-1].append(std)

        self._means = means
        self._stds = stds

        
    def _predict(self, obs):
        """
        Private method used to assign a probability to each class.
        """
        probs = []
        obs = self.process_vars(obs)
        for (means, stds, lab) in zip(self._means, self._stds, self.classes):
            product = 1
            for mu, sig, o in zip(means, stds, obs):
                likelihood = gaussian_prob(o, mu, sig)
                product *= likelihood
            probs.append(product)
        t = sum(probs)
        return [p/t for p in probs]

    def predict(self, observation):
        probs = self._predict(observation)
        idx = 0
        best_p = 0
        for i, p in enumerate(probs):
            if p > best_p:
                best_p = p
                idx = i
        names = ['left','keep','right']
        return names[idx]
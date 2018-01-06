import numpy as np
from track import Track
import matplotlib.pyplot as plt
import time

class TrackerEngine:
    """
    Tracking engine to parse observations and do the matching.
    
    Attrs:
        tracks: List of track objects in the pool currently
        observations: List of list of observations in each frame
    
    """
    
    def __init__(self, beta, observations):
        """
        Initializes the tracker engine.
        
        Args:
            beta (scalar): Fixed threshold probability
            observations([[(mx1),]]): Variable sized list of list of observations, each an mx1 observation vector
            
        Returns:
            Nothing
        """
        self.beta = beta
        self.tracks = []
        self.observations = observations

        # Drawing canvas
        self.fig, self.ax = plt.subplots()
        self.fig.show() # Should we keep this active permanently?
        self.ax.set_xlim(0,10)
        self.ax.set_ylim(0,10)
        
    def likelihood_mat(self, tracks, obs):
        """
        Calculate the likelihood matrix of each measurement being generated by each observation
        
        Args:
            trackers [Track,]: Variable sized list of trackers in the current frame
            obs ([(mx1),]): Variable sized list of mx1 observation vectors for current frame
            
        Returns:
            likelihood_mat (num_trackers, num_obs): Likelihood matrix of  
        """
        l_mat = np.zeros((len(tracks), len(obs)))

        for i in range(len(tracks)):
            for j in range(len(obs)):
                l_mat[i,j] = tracks[i].likelihood(obs[j])
        
        return l_mat
        
    def match_mat_det(self, l_mat):
        """
        Matches trackers to indices based on the likelihood matrix.
        Matching strategies: {'greedy' | 'hungarian' | 'jdpa'}
        
        Args:
            likelihood_mat (num_trackers, num_obs): Likelihood of each tracker generating each detection
            
        Returns:
            match_mat (num_trackers, num_obs): Binary matrix specifying which tracker matches which detection
        """
        match_mat = np.zeros(l_mat.shape)

        if match_mat.shape[0] == 0:
            return match_mat

        for j in range(match_mat.shape[1]):

            t_idx = l_mat[:,j].argmax()
            score = l_mat[t_idx, j]

            if score>self.beta:
                match_mat[t_idx, j] = 1

        return match_mat

    def match_mat_tracker(self, l_mat):
        """
        Matches trackers to indices based on the likelihood matrix.
        Matching strategies: {'greedy' | 'hungarian' | 'jdpa'}
        
        Args:
            likelihood_mat (num_trackers, num_obs): Likelihood of each tracker generating each detection
            
        Returns:
            match_mat (num_trackers, num_obs): Binary matrix specifying which tracker matches which detection
        """
        match_mat = np.zeros(l_mat.shape)

        if match_mat.shape[0] == 0:
            return match_mat

        for i in range(match_mat.shape[0]):

            j = l_mat[i,:].argmax()
            score = l_mat[i, j]

            if score>self.beta:
                match_mat[i, j] = 1

        return match_mat
    
    def update_trackers(self, obs, match_mat):
        """
        Updates the trackers based on the matching with detections.
        If match found, runs the tracker's update function.
        Else a new tracker is instantiated.
        
        Args:
            obs: List of mx1 observations
            match_mat (num_trackers, num_obs): Binary matrix specifying which tracker matches which detection
            
        Returns:
            Nothing
        """
        
        for j in range(match_mat.shape[1]):
            total = match_mat[:,j].sum()
            
            if total>0:
                t_idx = match_mat[:,j].argmax()
                self.tracks[t_idx].update(obs[j])
                
            else:
                # Pop new tracker ontop of the queue of trackers
                t = Track()
                t.update(obs[j])
                
                self.tracks.append(t)

    def prune(self):
        """ Prunes low likelihood tracks (which haven't received any action in a while) """
        new_tracks = []

        for t in self.tracks:
            if t.lhood > self.beta:
                new_tracks.append(t)

        self.tracks = new_tracks

    def predict(self):
        """ Projects ahead each remaining track """
        for t in self.tracks:
            t.predict() 

    def draw(self, obs, i):
        """ Draws things on a canvas object """
        #time.sleep(1)

        for ob in obs:
            self.ax.scatter(ob[0].item(), ob[1].item(), color='black')
        
            self.fig.canvas.draw()
            time.sleep(0.1)

        for t in self.tracks:
            self.ax.scatter(t.x[0].item(), t.x[1].item(), color=t.color)
        
            self.fig.canvas.draw()

        # Also draw observations?
    
    def run(self):
        """
        1. Loops over the list of observations to get current frame detections
        2. Computes likelihood matrix
        3. Computes match matrix based on the likelihood matrix and choice of matching strategy
        4. Updates tracks based on the match matrix result. Should have an edge case for 0 track size.
        5. Stores x estimates in another array which can then be printed.
        """
        for i,obs in enumerate(self.observations):
            
            l_mat = self.likelihood_mat(self.tracks, obs)
            match_mat = self.match_mat_tracker(l_mat)
            
            self.update_trackers(obs, match_mat)

            self.draw(obs,i)

            self.prune()
            self.predict()

        # THis is just a stupid for loop 

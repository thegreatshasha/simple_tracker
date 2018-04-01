import numpy as np
from scipy.stats import multivariate_normal
import random
import uuid

class Track:
    """
    Track class is a kalman filter.
    
    Attrs:
        x (nx1): Estimate of current state
        P (nxn): Current state covariance
        F (nxn): Transition matrix
        H (mxn): Emission matrix
        Q (nxn): Process noise covariance
        R (mxm): Observation noise covariance
        color : Unique color associated with track. This helps in visualizing the track
    """
    
    def __init__(self, x=np.matrix('0. 0. 0. 0.').T, P=np.matrix(np.eye(4))*1000):
        """
        Initializes the track at a particular location.
            
        Args:
            x (nx1): Vector of initial belief of the gaussian
            Q (nxn): Covariance matrix. Should be large.
                
        Returns:
            Nothing
        """
        # Initialization of each track
        self.x = x
        self.P = P

        # Motion model parameters for 1d
        # Remove these hardcoded values asap
        #self.R = np.matrix(0.1)
        #self.Q = np.matrix('0.0033 0.005; 0.005 0.001')/10
        #self.H = np.matrix('1. 0.')
        #self.F = np.matrix('1. 1.; 0. 1.')

        # Motion model parameters for 2d
        self.R = np.matrix(np.eye(2))/10
        self.Q = np.matrix('''
                      0.33  0.    0.5   0.;
                      0.    0.33  0.     0.5;
                      0.    0.    1.     0.;
                      0.    0.    0.     1.
                      ''')*100
        self.H = np.matrix('1. 0. 0. 0.; 0. 1. 0. 0.')
        self.F = np.matrix('''
                      1. 0. 1. 0.;
                      0. 1. 0. 1.;
                      0. 0. 1. 0.;
                      0. 0. 0. 1.
                      ''')

        self.color = np.random.rand(3,)
        self.history = []
        self.id = str(uuid.uuid4())[:8]
        self.location = 'tracks/%s.csv'%self.id
        
    def predict(self):
        """
        Predicts the future based on current state.
        """
        #print('previously', self.P, self.F*self.P*self.F.T)
        self.x = self.F * self.x
        self.P = self.F*self.P*self.F.T + self.Q
        #print('after', self.P, self.Q)
        
    def update(self, y, t):
        """
        Updates the state estimate based on the current observation
        
        Args:
            y (mx1): Observation vector
            
        Returns:
            Nothing
        """
        K = self.P*self.H.T*(self.H*self.P*self.H.T + self.R).I # Kalman gain computation
        self.x = self.x + K*(y - self.H*self.x) # Update mean
        self.P = (np.eye(self.x.shape[0]) - K*self.H)*self.P # Update covariance

        self.lhood = self.likelihood(y)
        #import pdb; pdb.set_trace()
        self.history.append([t, y[0].item(), y[1].item()])

    def likelihood(self, y):
        """
        Returns the likelihood of generating a certain observation based on the current belief
        
        Args:
            y (mx1): Observation vector
            
        Returns:
            likelihood (scalar): Probability of generating y based on the current belief
        """
        obs_mean = np.asarray(self.H*self.x).reshape(-1)
        obs_cov = self.H*self.P*self.H.T + self.R
        #print(obs_mean.shape, obs_cov.shape)

        var = multivariate_normal(obs_mean, obs_cov)
        likelihood = var.pdf(np.asarray(y).reshape(-1))
        
        return likelihood

    def serialize(self):
        hist_array = np.asarray(self.history)
        np.savetxt(self.location, hist_array, delimiter=",")


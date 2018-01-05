import numpy as np
from scipy.stats import multivariate_normal
import random

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
    
    def __init__(self, x=np.matrix('0. 0.').T, P=np.matrix(np.eye(2))*1000):
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

        # Motion model parameters
        # Remove these hardcoded values asap
        self.R = np.matrix(0.1)
        self.Q = np.matrix('0.0033 0.005; 0.005 0.001')/10
        self.H = np.matrix('1. 0.')
        self.F = np.matrix('1. 1.; 0. 1.')

        self.color = 'red'#random.choice(['r','g','b'])
        
    def predict(self):
        """
        Predicts the future based on current state.
        """
        print('previously', self.P, self.F*self.P*self.F.T)
        self.x = self.F * self.x
        self.P = self.F*self.P*self.F.T + self.Q
        print('after', self.P)
        
    def update(self, y):
        """
        Updates the state estimate based on the current observation
        
        Args:
            y (mx1): Observation vector
            
        Returns:
            Nothing
        """
        K = self.P*self.H.T*(self.H*self.P*self.H.T + self.R).I # Kalman gain computation
        self.x = self.x + K*(y - self.H*self.x) # Update mean
        self.P = (np.eye(2) - K*self.H)*self.P # Update covariance

        self.lhood = self.likelihood(y)
        
    def likelihood(self, y):
        """
        Returns the likelihood of generating a certain observation based on the current belief
        
        Args:
            y (mx1): Observation vector
            
        Returns:
            likelihood (scalar): Probability of generating y based on the current belief
        """
        obs_mean = self.H*self.x
        obs_cov = self.H*self.P*self.H.T + self.R

        var = multivariate_normal(obs_mean, obs_cov)
        likelihood = var.pdf(y)

        return likelihood

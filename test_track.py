import unittest
import numpy as np
import matplotlib.pyplot as plt
from track import Track

class TrackerTest(unittest.TestCase):
    """Tests for `track.py`."""

    def test_likelihood(self):
        """ Tests that the likelihood function of track works and produces the correct result """
        t = Track(np.matrix('0. 0.').T, np.matrix('1. 0.; 0. 1.')) # Pass mean and covariance
        
        y = np.matrix(0.5)
        
        print(t.likelihood(y))
    
    def _test_kalman(self):
        """ Tests that the covariance actually decreases for sensible choice of initialization conditions """
        t = Track()

        # Observations
        ys = np.matrix([0.15, 0.19, 0.30, 0.42, 0.45, 0.64, 0.72, 0.83, 0.90, 1.02, 1.12, 1.22, 1.34, 1.36, 1.53, 1.58, 1.67, 1.80, 1.90, 1.97]).T

        xs = []
        vs = []

        for y in ys:
            t.predict()
            t.update(y)

            xs.append(t.x[0].item())
            vs.append(t.x[1].item())

        # Plot the position esimate
        plt.figure(figsize=(10,10))
        plt.plot(range(ys.shape[0]), ys.A1, 's')
        plt.plot(range(ys.shape[0]), xs)

        # Plot the velocity estimate
        plt.figure(figsize=(10,10))
        plt.plot(range(ys.shape[0]), vs)

        plt.show()

if __name__ == '__main__':
    unittest.main()
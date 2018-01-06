import unittest
import numpy as np
from engine import TrackerEngine
from track import Track
import matplotlib.pyplot as plt
import time

class ToyDataset:

    def __init__(self, num_particles=2, eta=0.5):
        self.num_particles = num_particles

        self.F = np.matrix('''
                      1. 0. 1. 0.;
                      0. 1. 0. 1.;
                      0. 0. 1. 0.;
                      0. 0. 0. 1.
                      ''')

        self.H = np.matrix('1. 0. 0. 0.; 0. 1. 0. 0.')

        self.particles = [np.matrix(np.random.rand(4,1)) for i in range(num_particles)]

        self.observations = []

        self.eta = eta

    def update(self):
        
        detections = []

        for i in range(len(self.particles)):
            self.particles[i] = self.F * self.particles[i] 
            pos = self.H * self.particles[i] + self.eta*np.matrix(np.random.rand(2,1))

            detections.append(pos)

        self.observations.append(detections)

    def generate(self, n):
        for i in range(n):
            self.update()

        return self.observations

class TrackerEngineTest2D(unittest.TestCase):
    """Tests for `primes.py`."""

    def test_single_bead_2d(self):
        dg = ToyDataset(num_particles=5, eta=0.5)
        detections = dg.generate(n=15)
        # detections = [
        #                 [np.matrix('''0.5 0.5''').T, np.matrix('''4.5 4.5''').T],
        #                 [np.matrix('''1. 1.''').T, np.matrix('''4. 4.''').T],
        #                 [np.matrix('''1.5 1.5''').T, np.matrix('''3.5 3.5''').T],
        #                 [np.matrix('''2.0 2.0''').T, np.matrix('''3.0 3.0''').T],
        #                 [np.matrix('''2.5 2.5''').T, np.matrix('''2.5 2.5''').T],
        #                 [np.matrix('''3. 3.''').T, np.matrix('''2.0 2.0''').T],
        #                 [np.matrix('''3.5 3.5''').T, np.matrix('''1.5 1.5''').T],
        #                 [np.matrix('''4. 4.''').T, np.matrix('''1. 1.''').T],
        #                 [np.matrix('''4.5 4.5''').T, np.matrix('''0.5 0.5''').T],
        #              ]
        e = TrackerEngine(0.0001, detections)
        e.run()

        print(e.tracks)
        for t in e.tracks:
            print(t.P, t.x)

        print(len(e.tracks))

        raw_input('your mom')

    def _test_match_hungarian(self):
        detections = []
        e = TrackerEngine(0.0001, detections)
        likelihood_mat = np.matrix('0.01 1.2 0.13; 0.05 0.5 0.16')
        match_mat = e.match_mat_hungarian(likelihood_mat)
        print(likelihood_mat)
        print(match_mat)


if __name__ == '__main__':
    unittest.main()
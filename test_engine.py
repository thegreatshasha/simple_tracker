import unittest
import numpy as np
from engine import TrackerEngine
from track import Track

class TrackerEngineTest(unittest.TestCase):
    """Tests for `primes.py`."""

    def test_likelihood_mat(self):
        """ Provide mock tracker objects and detections and check that the correct likelihood value is generates """
        t1 = Track(np.matrix('0. 0.').T, np.matrix('1. 0.; 0. 1.'))
        t2 = Track(np.matrix('2. 2.').T, np.matrix('1. 0.; 0. 1.'))
        tracks = [t1, t2]
        #tracks = []

        detections = [[np.matrix(0.), np.matrix(1.), np.matrix(2.)]]

        e = TrackerEngine(0.1, detections)
        
        print('likelihood is')
        print(e.likelihood_mat(tracks, detections[0]))
    
    def _test_matching_greedy(self):
        """ Tests the greedy matching strategy inside the algorithm """
        detections = [[np.matrix(0.), np.matrix(1.), np.matrix(2.)]]

        e = TrackerEngine(0.1, detections)
        likelihood_mat = np.matrix('0.01 1.2 0.13; 0.05 0.5 0.16')
        print(likelihood_mat)

        print(e.match_mat(likelihood_mat))

    def _test_update_trackers(self):
        """ Test that the update after the matching works fine """
        """ Testing with two detections, two trackers. One matching, one not """
        t1 = Track(np.matrix('0. 0.').T, np.matrix('1. 0.; 0. 1.'))
        t2 = Track(np.matrix('2. 2.').T, np.matrix('1. 0.; 0. 1.'))
        tracks = [t1, t2]

        detections = [[np.matrix(0.), np.matrix(1.), np.matrix(2.)]]
        match_mat = np.matrix('0 0 0; 0 0 1')

        e = TrackerEngine(0.1, detections)
        #e.tracks = tracks
        
        e.update_trackers(detections[0], match_mat)
        
        for t in e.tracks:
            print(t.x[0].item())
        # Check that the length of the trackers has increased same as the no of empty columns

    def _test_single_bead_static(self): # most important, plot trajectory first
        """ Single stationary bead tracking. See if one of the shadow trackers actually stays with the static observation. Also plot trajectories. """
        detections = [[np.matrix(1.5)], [np.matrix(1.5)], [np.matrix(1.5)]]
        e = TrackerEngine(0.1, detections)
        e.run()

    
#     def test_single_bead_moving(self):
#         """ Tests that a single bead moving at constant velocity of 1m/s is tracked by the engine. Plot trajectories. """
#         pass
    
#     def test_two_static_beads(self):
#         """ Test that two stationary beads are tracked by the engine. Plot trajectories. """
#         pass
    
#     def test_two_moving_beads(self):
#         """ Test that two moving beads are tracked by the engine. Plot trajectories. """
#         pass
    
#     def test_two_particles_crossing_instant(self):
#         """ Three particles crossing each other at the same time for an instant"""
#         pass
    
#     def test_two_particle_stay_parallel(self):
#         """ Two particles come together, stay together for a while and then separate. Smooth trajectories. """
#         pass

#       def test_grid_tracking(self):
#           """ Grid tracking """
#           pass

# Coming soon
#     def test_matching_hungarian(self):
#         """ Tests the hungarian matching strategy """
#         pass
    
#     def test_matching_jpda(self):
#         """ Tests the jpda matching strategy """
#         pass

if __name__ == '__main__':
    unittest.main()
from engine import TrackerEngine
import numpy as np
detections = np.load('data.npy')

#print(detections)
e = TrackerEngine(0.0001, detections)
e.run()

print(e.tracks)
for t in e.tracks:
	print(t.P, t.x)

print(len(e.tracks))
raw_input('asdgasgd')
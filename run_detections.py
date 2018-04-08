from engine import TrackerEngine
import numpy as np
import glob

detections = np.load('data_backward.npy')
img_fls = sorted(glob.glob('test_images/*'), reverse=True)

#print(detections)
e = TrackerEngine(0.00001, detections, img_fls)
e.run()

print(e.tracks)
for t in e.tracks:
	print(t.P, t.x)

print(len(e.tracks))
raw_input('asdgasgd')
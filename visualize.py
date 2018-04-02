import numpy as np
import glob
from skimage.io import imread
import matplotlib.pyplot as plt
import random
from tqdm import tqdm

def query(track_data, t):
	return track_data[np.isin(track_data[:,0], range(t+1-10, t+1))][:,[1,2]]

track_fls = glob.glob('tracks/*.csv')
track_fls_random = random.sample(track_fls, 30)
img_fls = sorted(glob.glob('test_images/*'))
det_fls = sorted(glob.glob('detections_csv/*'))
detections = np.load('data.npy')
print()


colors = [(np.random.random(size=3) * 256) for i in range(len(track_fls_random))]


for t, img_fl in tqdm(enumerate(img_fls)):
	img = imread(img_fl)
	det = np.loadtxt(det_fls[t], delimiter=',')

	plt.figure(figsize=(10,10))
	plt.imshow(img)
	plt.scatter(det[:,0], det[:,1], s=16, c='black', marker='x')
	
	for j, t_fl in enumerate(track_fls_random):
		track_data = np.loadtxt(t_fl, delimiter=',')
		xy = query(track_data, t)

		#print(xy)

		
		plt.scatter(xy[:,0], xy[:,1], s=5)
	
	plt.savefig('tracks_outputs/%d'%t)
	plt.close()


print(track_fls)
import numpy as np
csv_dir = 'detections_csv/*'

num = 100
from glob import glob

out_files = sorted(glob(csv_dir))
print(out_files)

det_array = []

for out_fl in out_files:
	data = np.loadtxt(open(out_fl, "rb"), delimiter=",", skiprows=1)

	#data = np.loadtxt(out_fl, delimiter='.')

	data_mat = []

	for x, y in data:
		data_mat.append(np.matrix([x, y]).T)

	det_array.append(data_mat)

np.save('data.npy', det_array)

#print(det_array)
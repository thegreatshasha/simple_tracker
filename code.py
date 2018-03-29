import numpy as np
csv_dir = 'csv_final/*'

num = 30
from glob import glob

out_files = sorted(glob(csv_dir))[:num]
print(out_files)

det_array = []

for out_fl in out_files:
	data = np.loadtxt(out_fl)

	data_mat = []

	for x, y, theta in data:
		data_mat.append(np.matrix([x, y]).T)

	det_array.append(data_mat)

np.save('data.npy', det_array)

#print(det_array)
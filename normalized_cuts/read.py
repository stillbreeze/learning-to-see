import h5py
import numpy as np

with h5py.File('eig.h5', 'r') as hf:
	eigVal=np.array(hf.get('eigVal'))
	eigVec=np.array(hf.get('eigVec'))
print eigVal
print eigVec
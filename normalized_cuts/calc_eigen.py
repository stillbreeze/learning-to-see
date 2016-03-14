from __future__ import division
import cv2
import numpy as np
from math import sqrt, pow, exp
from scipy.sparse import csr_matrix
import h5py

if __name__ == "__main__":
	with h5py.File('weight_matrix.h5', 'r') as hf:
		W=np.array(hf.get('weight_matrix'))
	d=np.sum(W,1)
	D=np.diag(d)
	print W.shape
	print d.shape
	print D.shape
	# eigVal,eigVec=np.linalg.eig()
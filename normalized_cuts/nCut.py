import h5py
import numpy as np
import calc_ncut as ncut
import scipy.optimize


def NcutRecursive():
	with h5py.File('eig.h5', 'r') as hf:
		eigVal=np.array(hf.get('eigVal'))
		eigVec=np.array(hf.get('eigVec'))

	with h5py.File('weight_matrix.h5', 'r') as hf:
		W=np.array(hf.get('weight_matrix'))

	d=np.sum(W,1)
	D=np.diag(d)

	D=D.astype("float32",copy=False)
	W=W.astype("float32",copy=False)

	eigVecSmall=eigVec[:,1]
	eigVecSmall=eigVecSmall.real
	thresh=np.mean(eigVecSmall)

	thresh=scipy.optimize.fmin(ncut.ncutValue, thresh, args=(eigVecSmall, D, W, d))[0]

	A=np.where(eigVecSmall>thresh)
	B=np.where(eigVecSmall<=thresh)

if __name__ == "__main__":
	NcutRecursive()
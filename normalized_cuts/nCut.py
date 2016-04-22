import h5py
import numpy as np
import calc_ncut as ncut
import calc_eigen as eig
import scipy.optimize
import cv2
import pickle



def NcutRecursive(W,segFinal,imgSeg,threshcut,thresharea,first):

	if not first:
		with h5py.File('eig.h5', 'r') as hf:
			eigVec=np.array(hf.get('eigVec'))
		first+=1
	else:
		print 'calculating eigen vectors'
		try:
			eigVec=eig.eigen(W)
			with h5py.File('eig'+str(first)+'.h5', 'w') as hf:
				hf.create_dataset('eigVec', data=eigVec)
			first+=1
		except:
			print 'no eigen vectors found'
			return 1
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

	A=A[0]
	B=B[0]

	print A
	print '\n'
	print B

	print thresh
	cut=ncut.ncutValue(thresh,eigVecSmall,D,W,d)
	print cut
	if (cut > threshcut) or (A.shape[0]<thresharea) or (B.shape[0]<thresharea):
		print 'Finished recursion'
		return 1
	segFinal.append((A.tolist(),B.tolist()))
	res=NcutRecursive(np.take(W[A],A,1),segFinal,np.take(imgSeg,A),threshcut,thresharea,first)
	res=NcutRecursive(np.take(W[B],B,1),segFinal,np.take(imgSeg,B),threshcut,thresharea,first)
	print 'Writing segments'
	with open('segFinal.pickle','wb') as f:
		pickle.dump(segFinal,f)

if __name__ == "__main__":
	img=cv2.imread('./colors.jpg',0)
	imgSeg=cv2.resize(img,(50,50))

	with h5py.File('weight_matrix.h5', 'r') as hf:
		W=np.array(hf.get('weight_matrix'))
	segFinal=[]
	first=0
	threshcut=0.5
	thresharea=20
	NcutRecursive(W,segFinal,imgSeg,threshcut,thresharea,first)
import cv2
import numpy as np
from matplotlib import pyplot as plt
from numpy import linalg as LA


# Read,resize images. Returns image matrices
def read(left,right,width,height):
    img1 = cv2.imread(left,0)
    img2 = cv2.imread(right,0)
    img1 = cv2.resize(img1,(width,height))
    img2 = cv2.resize(img2,(width,height))
    return img1,img2

# SIFT-KNN based keypoint matcher. Returns high confidence corresponding keypoints from each stereo image
def getMatches(img1,img2):
    sift = cv2.xfeatures2d.SIFT_create()

    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)


    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=500)

    flann = cv2.FlannBasedMatcher(index_params,search_params)
    matches = flann.knnMatch(des1,des2,k=2)

    good = []
    pts1 = []
    pts2 = []

    for i,(m,n) in enumerate(matches):
        if m.distance < 0.8*n.distance:
            good.append(m)
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)

    return pts1,pts2

# Calculate fundamental matrix, and in turn, return epilines in second image corresponding to every set of coordinates in first image 
def getEpilines(pts1,pts2,width,height):
    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)
    F, mask = cv2.findFundamentalMat(pts1,pts2,cv2.FM_LMEDS)

    pts = np.array([(i,j) for i in range(width) for j in range(height)])

    epilines = cv2.computeCorrespondEpilines(pts.reshape(-1,1,2), 2,F)
    epilines = epilines.reshape(-1,3)
    return epilines

# Calculates window-based cost function and minimizes the disparity between pixels. Returns minimum intensity disparity for each point on first image
def calcCost(img1,img2,i1,j1,j2,window_size,height):
    w_size = window_size/2
    min_intensity_disparity = 99999
    window1 = img1[i1-w_size:i1+1+w_size,j1-w_size:j1+1+w_size]
    for i2 in range(0+(window_size/2),height-(window_size/2)):
        window2 = img2[i2-w_size:i2+1+w_size,j2-w_size:j2+1+w_size]

        # Comment/Uncomment next 2 lines for changing cost function 

        # intensity_disparity = np.sum(np.fabs(np.subtract(window2,window1)))
        intensity_disparity = LA.norm((np.subtract(window2,window1)),'fro')

        if intensity_disparity < min_intensity_disparity:
            min_intensity_disparity = intensity_disparity

    return min_intensity_disparity

# Prepare disparity map by minimizing cost function along the horizontal epiline
def prepDisparityMap(img1,img2,window_size,epilines,width,height):
    b_line=epilines[::,1]
    c_line=epilines[::,2]
    y = -c_line/b_line
    y = y.reshape((width,height))
    disparity_map=np.zeros_like(img1)

    for i in range(0+(window_size/2),width-(window_size/2)):
        for j in range(0+(window_size/2),height-(window_size/2)):
            if y[i][j] > height-(window_size/2)-1:
                y[i][j] = height-(window_size/2)-1
            elif y[i][j] < (window_size/2):
                y[i][j] = window_size/2
            rastor_search_line = int(round(y[i][j]))
            min_intensity_disparity = calcCost(img1.astype('int32'),img2.astype('int32'),i,j,rastor_search_line,window_size,height)
            disparity_map[i][j] = min_intensity_disparity
    return disparity_map


if __name__ == "__main__":

# Image reading parameters
    # -------------------------------------------------
    left = 'tsukuba-left.ppm'
    right = 'tsukuba-right.ppm'
    width,height = 75,75
    img1,img2 = read(left,right,width,height)

# Keypoint matching for fundamental matrix and epiline generation for all coordinates of image
    # -------------------------------------------------
    pts1,pts2 = getMatches(img1,img2)

    epilines = getEpilines(pts1,pts2,width,height)

# Construction of disparity map
    # -------------------------------------------------
    window_size = 2
    disparity_map = prepDisparityMap(img1,img2,window_size,epilines,width,height)

# Heat map visualization
    # -------------------------------------------------
    imgplot = plt.imshow(disparity_map)
    imgplot.set_cmap('nipy_spectral')
    plt.colorbar()
    # plt.savefig('phone@3.png')
    plt.show()
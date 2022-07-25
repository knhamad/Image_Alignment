import numpy as np
from skimage.morphology import (erosion, dilation, closing, opening,
                                area_closing, area_opening)
from skimage.measure import label, regionprops, regionprops_table
import cv2 

square = np.ones((3,3))

def segment(imgWithKeypoints):
    imgWithKeypoints[imgWithKeypoints<70]=0
    imgWithKeypoints[imgWithKeypoints>=70]=1
    return imgWithKeypoints

def MO(LabeledImg):
    multi_dilated = multi_dil( LabeledImg, 2)
    area_closed = area_closing(multi_dilated, 50000)
    opened = cv2.morphologyEx(area_closed, cv2.MORPH_OPEN, square,iterations = 5)
    multi_eroded = multi_ero(opened, 2)
    return multi_eroded

def CCA(LabeledImgAfterMO):
    label_im = label(LabeledImgAfterMO)
    return label_im

def createKeyPoints(CCAimg):
    num_of_components=np.unique(CCAimg)
    keypoints=np.zeros((len(num_of_components)-1,2))
    for i in range(len(num_of_components)-1):
        r,c=np.where(CCAimg==num_of_components[i+1])
        # keypoints[i,:]=np.array([c[0],r[0]])
        keypoints[i,:]=np.array([int(np.mean(c)),int(np.mean(r))])
    return keypoints

def Homography(k1,k2,RANSAC_Thresh=5):
    h, mask = cv2.findHomography(k2, k1, cv2.RANSAC,RANSAC_Thresh)
    return h

def warp(fixed,moving,Homography):
    return cv2.warpPerspective(moving, Homography, (fixed.shape[1],fixed.shape[0]))

def mutual_information(hgram):
    """ Mutual information for joint histogram
    """
    # Convert bins counts to probability values
    pxy = hgram / float(np.sum(hgram))
    px = np.sum(pxy, axis=1) # marginal for x over y
    py = np.sum(pxy, axis=0) # marginal for y over x
    px_py = px[:, None] * py[None, :] # Broadcast to multiply marginals
    # Now we can do the calculation using the pxy, px_py 2D arrays
    nzs = pxy > 0 # Only non-zero pxy values contribute to the sum
    return np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))

def multi_dil(im, num, element=square):
    for i in range(num):
        im = dilation(im, element)
    return im

def multi_ero(im, num, element=square):
    for i in range(num):
        im = erosion(im, element)
    return im
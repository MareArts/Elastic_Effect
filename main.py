import numpy as np
import cv2 
from Elastic import elastic


#file read
o_img = cv2.imread('izone_oy.png')
elMat = elastic(o_img, alpha=5000, sigma=8, random_state=None)

cv2.namedWindow('origin',0)
cv2.imshow('origin', o_img)
cv2.namedWindow('elastic',0)
cv2.imshow('elastic', elMat)

cv2.waitKey(0)

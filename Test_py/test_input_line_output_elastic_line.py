import numpy as np
import cv2 
import sys
from Elastic import elastic 
from Elastic import test_elastic_pts 


rows = 500
cols = 500
newMat_3ch = np.zeros((rows, cols, 3), dtype = "uint8") #3channel


step = 20
x = np.linspace(start=0, stop=rows-1, num=step)
y = np.linspace(start=0, stop=cols-1, num=step)

v_xy = []
h_xy = []
for i in range(step):
    v_xy.append( [int(x[i]), 0, int(x[i]), rows-1] )
    h_xy.append( [0, int(y[i]), cols-1, int(y[i])] )

for i in range(step):
    [x1, y1, x2, y2] = v_xy[i]
    [x1_, y1_, x2_, y2_] = h_xy[i]

    cv2.line(newMat_3ch, (x1,y1), (x2, y2), (0,0,255),1 )
    cv2.line(newMat_3ch, (x1_,y1_), (x2_, y2_), (255,0,0),1 )
    
cv2.namedWindow('newMat_3ch',0)
cv2.imshow('newMat_3ch', newMat_3ch)


xy1 = []
for i, v in enumerate(x):
    xy1.append( [int(v), y[10]])
print(xy1)

elMat, el_pts = test_elastic_pts(newMat_3ch, alpha=500, sigma=8, pts=xy1, random_state=None)

x1=y1=0
for i, point in enumerate(el_pts):
    x2, y2 = int(point[0]), int(point[1])
    cv2.circle(elMat,(x2,y2),3,(0,0,255))
    if i>0:
        cv2.line(elMat, (x1,y1), (x2, y2), (0,0,255),1 )
    x1 = int(x2)
    y1 = int(y2)
cv2.namedWindow('elastic',0)
cv2.imshow('elastic', elMat)



#elastic
#elMat = elastic(newMat_3ch, alpha=5000, sigma=8, random_state=None)
# cv2.namedWindow('elMat',0)
# cv2.imshow('elMat', elMat)
cv2.waitKey(0)



sys.exit('test1')

import numpy as np
import cv2 
from Elastic import test_elastic_pts


#file read
o_img = cv2.imread('izone_oy.png')

#MAKE LINE
w = o_img.shape[1]
h = o_img.shape[0]
sx1 = int(w/4)
ex1 = int(w/4*3)
xstep = int((ex1 - sx1)/10)
sy1 = int(h/4)
ey1 = int(h/4*3)
ystep = int((ey1 - sy1)/10)

x = np.linspace(start=sx1, stop=ex1, num=20)
y = np.linspace(start=sy1, stop=ey1, num=20)

xy1 = []
for i, v in enumerate(x):
    xy1.append( [v, sy1])


#draw line on image and check
pts = np.array(xy1, np.int32)
#pts = pts.reshape((-1, 1, 2))
cv2.polylines(o_img, [pts], True, (0,255,0), 1)
cv2.namedWindow('o_img',0)
cv2.imshow('o_img', o_img)

#elastic
elMat, el_pts = test_elastic_pts(o_img, alpha=5000, sigma=8, pts=xy1, random_state=None)

#check elastic line
#now this is not correct!!
x1=y1=0
for i, point in enumerate(el_pts):
    x2, y2 = int(point[0]), int(point[1])
    cv2.circle(elMat,(x2,y2),3,(0,0,255))
    if i>0:
        cv2.line(elMat, (x1,y1), (x2, y2), (0,0,255),2 )
    x1 = int(x2)
    y1 = int(y2)
cv2.namedWindow('elastic',0)
cv2.imshow('elastic', elMat)


cv2.waitKey(0)
